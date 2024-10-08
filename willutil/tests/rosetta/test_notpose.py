import numpy as np
import pytest
import willutil as wu

# ic.configureOutput(includeContext=True, contextAbsPath=False)


def main():
    test_bug1()
    test_notpose_from_coords()
    test_notpose_has()
    test_notpose()
    test_notpose_sc_coords()
    ic("test_notpose.py DONE")


def test_notpose_from_coords():
    pdb = wu.readpdb(wu.tests.testdata.test_data_path("pdb/1pgx.pdb1.gz"))
    ncaco = pdb.ncaco()
    ic(ncaco.shape)

    nopo = wu.NotPose(coords=ncaco, seq=pdb.sequence())
    nopo2 = wu.NotPose(pdb=pdb)
    assert np.allclose(nopo.ncaco, nopo2.ncaco)
    assert nopo.sequence() == nopo2.sequence()
    assert nopo.secstruct() == nopo2.secstruct()
    assert nopo.size() == nopo2.size()
    assert nopo.pdb_info().name() == "NONAME"
    assert nopo2.pdb_info().name().endswith("pdb/1pgx.pdb1.gz")
    for ir in range(1, nopo.size() + 1):
        assert np.allclose(nopo.residue(ir).xyz("C"), nopo.residue(ir).xyz("C"))
    assert nopo.chain(70) == "A"

    nopo.extract(chain="A")


def test_notpose_has():
    fname = wu.tests.testdata.test_data_path("pdb/1pgx.pdb1.gz")
    nopo = wu.NotPose(fname)
    assert nopo.residue(4).has("CA")
    assert nopo.residue(40).has("CB")


def _get_sc_coords(pose, which_resi=None, recenter_input=False, **kw):
    pytest.importorskip("pyrosetta")
    kw = wu.Bunch(kw, _strict=False)
    if which_resi is None:
        which_resi = list(range(1, pose.size() + 1))
    resaname, resacrd = list(), list()
    for ir in which_resi:
        r = pose.residue(ir)
        if not r.is_protein():
            raise ValueError("non-protein residue %s at position %i" % (r.name(), ir))
        anames, crd = list(), list()
        for ia in range(r.natoms()):
            anames.append(r.atom_name(ia + 1))
            xyz = r.xyz(ia + 1)
            crd.append([xyz.x, xyz.y, xyz.z])
        resaname.append(anames)
        hcrd = np.ones((len(anames), 4), dtype="f4")
        hcrd[:, :3] = np.array(crd)
        resacrd.append(hcrd)
    if recenter_input:
        bb = get_bb_coords(pose, which_resi, **kw.sub(recenter_input=False))
        cen = np.mean(bb.reshape(-1, 4)[:, :3], 0)
        for xyz in resacrd:
            xyz[:, :3] -= cen
    return resaname, resacrd


def test_notpose_sc_coords():
    fname = wu.tests.testdata.test_data_path("pdb/1pgx.pdb1.gz")
    nopo = wu.NotPose(fname)
    names, coords = _get_sc_coords(nopo)
    for n, c in zip(names, coords):
        assert len(n) == len(c)
        assert len(set(n)) == len(n)


def test_notpose():
    pyro = pytest.importorskip("pyrosetta")

    pyro.init("-mute all")
    fname = wu.tests.testdata.test_data_path("pdb/1pgx.pdb1.gz")

    nopo = wu.NotPose(fname)
    pose = pyro.pose_from_file(fname)
    assert len(pose.secstruct()) == len(nopo.secstruct())
    assert nopo.size() == pose.size()
    assert pose.sequence() == nopo.sequence()
    for ir in range(1, 1 + pose.size()):
        npo = pose.residue(ir).nheavyatoms()
        nnopo = nopo.residue(ir).nheavyatoms()
        if pose.residue(ir).name() == "PRO":
            npo -= 1
        if pose.residue(ir).name() == "LYS":
            continue
        if ir == 70:  # OXT
            nnopo += 1
        # ic(ir, pose.residue(ir).name(), pose.residue(ir).nheavyatoms(), nopo.residue(ir).nheavyatoms())
        # ic(nnopo)
        assert npo == nnopo

    nlysclose, nname = 0, 0
    for ir in range(1, 1 + pose.size()):
        for ia in range(1, 1 + nopo.residue(ir).natoms()):
            if pose.residue(ir).is_virtual(ia):
                continue
            # ic(ir, ia, pose.residue(ir).name(), pose.residue(ir).atom_name(ia), nopo.residue(ir).atom_name(ia))
            # ic(list(pose.residue(ir).xyz(ia)), nopo.residue(ir).xyz(ia))
            if not pose.residue(ir).atom_name(ia).strip() == nopo.residue(ir).atom_name(ia).strip():
                nname += 1
                continue
            if not np.allclose(pose.residue(ir).xyz(ia), nopo.residue(ir).xyz(ia)):
                assert pose.residue(ir).name() == "LYS"
                assert np.allclose(pose.residue(ir).xyz(ia), nopo.residue(ir).xyz(ia), atol=4)
                nlysclose += 1
    assert nlysclose == 11
    assert nname == 1


if __name__ == "__main__":
    main()
