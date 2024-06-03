import pytest
import numpy as np
import willutil as wu


def main():
    _test_symmetrize_frames()
    assert 0

    test_subframes()
    test_frames_asym_of()
    test_frames_asym_remove_sameaxis()
    test_remove_if_same_axis()
    test_sym()
    test_sym_frames()


def _test_symmetrize_frames():
    from opt_einsum import contract as einsum

    R = np.load("/home/sheffler/project/symmmotif_HE/R.npy")
    T = np.load("/home/sheffler/project/symmmotif_HE/T.npy")
    x = np.load("/home/sheffler/project/symmmotif_HE/xyzorig_in.npy")
    x = x[0]
    symmsub = np.load("/home/sheffler/project/symmmotif_HE/symmsub.npy")
    symmRs = np.load("/home/sheffler/project/symmmotif_HE/symmRs.npy")
    N = 5
    R = np.stack([R[:N], R[100 : 100 + N], R[200 : 200 + N], R[300 : 300 + N]])
    T = np.stack([T[:N], T[100 : 100 + N], T[200 : 200 + N], T[300 : 300 + N]])
    ic(x.shape)
    x = np.stack([x[:N], x[100 : 100 + N], x[200 : 200 + N], x[300 : 300 + N]])

    RT = wu.hconstruct(R, T)
    # wu.showme(RT)

    crd = np.stack([[0, 0, 0, 1], [2, 0, 0, 1], [0, 2, 0, 1]])
    crd = wu.hxform(RT, crd)
    ic(crd.shape)
    wu.dumppdb("test.pdb", crd)

    ic(R.shape, x.shape)
    wu.dumppdb("xyzorig.pdb", x + T[:, :, None, :])
    wu.dumppdb("RT_x.pdb", einsum("srij,sraj->srai", R, x) + T[:, :, None, :])

    assert 0


@pytest.mark.skip
def test_subframes():
    frames = wu.sym.frames("tet")
    subframes = wu.sym.subframes(frames, "C3", asym=[100, 10, 1])
    ic(frames.shape)
    ic(subframes.shape)


def test_frames_asym_of():
    f = wu.sym.frames("icos", asym_of="c5")
    assert len(f) == 12
    f = wu.sym.frames("icos", asym_of="c3")
    assert len(f) == 20

    f = wu.sym.frames("icos", asym_of="c2")
    assert len(f) == 30
    f = wu.sym.frames("oct", asym_of="c4")
    assert len(f) == 6

    f = wu.sym.frames("oct", asym_of="c3")
    assert len(f) == 8
    f = wu.sym.frames("oct", asym_of="c2")
    assert len(f) == 12

    f = wu.sym.frames("tet", asym_of="c3")
    assert len(f) == 4
    f = wu.sym.frames("tet", asym_of="c2")
    assert len(f) == 6


def test_frames_asym_remove_sameaxis():
    syms = "tet oct icos".split()
    csyms = "c2 c3 c4 c5".split()
    config = [
        ("tet  c2".split(), (4, 6, 6, 12)),
        ("tet  c3".split(), (2, 4, 4, 12)),
        ("oct  c2".split(), (7, 12, 12, 24)),
        ("oct  c3".split(), (4, 8, 8, 24)),
        ("oct  c4".split(), (3, 6, 6, 24)),
        ("icos c2".split(), (16, 30, 30, 60)),
        ("icos c3".split(), (8, 20, 20, 60)),
        ("icos c5".split(), (4, 12, 12, 60)),
    ]
    for i, ((sym, csym), (n1, n2, n3, n4)) in enumerate(config):
        # print(i, sym, csym)
        cart = wu.sym.axes(sym, csym)
        # cart = [0, 0, 10]

        f = wu.sym.frames(sym, bbsym=csym, asym_of=csym, axis=[0, 0, 1])
        assert len(f) == n1
        # print(i, sym, csym, len(f))
        f[:, :, 3] += 10 * wu.homog.hdot(f, cart)
        # print(f[0, :, 2])
        # assert 0

        # order frames along Z!!!

        # wu.viz.showme(f, spheres=0.5, name=f'test_{sym}_{csym}_bbsym_asymof')
        # wu.viz.showme(f[1:-1], spheres=1, name=f'test_{sym}_{csym}_bbsym_asymof')

        f = wu.sym.frames(sym, asym_of=csym, axis=[0, 0, 1])
        assert len(f) == n2
        # print(i, sym, csym, len(f))
        f[:, :, 3] += 10 * wu.homog.hdot(f, cart)
        # wu.viz.showme(f, spheres=0.4, name=f'test_{sym}_{csym}_asymof')

        f = wu.sym.frames(sym, bbsym=csym, axis=[0, 0, 1])
        assert len(f) == n3
        # print(i, sym, csym, len(f))
        f[:, :, 3] += 10 * wu.homog.hdot(f, cart)
        # wu.viz.showme(f, spheres=0.3, name=f'test_{sym}_{csym}_bbsym')

        f = wu.sym.frames(sym, axis=[0, 0, 1], axis0=wu.sym.axes(sym, csym))
        assert len(f) == n4
        # print(i, sym, csym, len(f))
        f[:, :, 3] += 10 * wu.homog.hdot(f, cart)
        # wu.viz.showme(f, spheres=0.2, name=f'test_{sym}_{csym}_all')


def test_remove_if_same_axis():
    f = wu.sym.frames("tet")
    assert f.shape == (12, 4, 4)
    assert wu.sym.frames("tet", bbsym="c2").shape == (6, 4, 4)
    assert wu.sym.frames("tet", bbsym="c3").shape == (4, 4, 4)

    f = wu.sym.frames("oct")
    assert f.shape == (24, 4, 4)
    assert wu.sym.frames("oct", bbsym="c2").shape == (12, 4, 4)
    assert wu.sym.frames("oct", bbsym="c3").shape == (8, 4, 4)
    assert wu.sym.frames("oct", bbsym="c4").shape == (6, 4, 4)

    f = wu.sym.frames("icos")
    assert f.shape == (60, 4, 4)
    assert wu.sym.frames("icos", bbsym="c2").shape == (30, 4, 4)
    assert wu.sym.frames("icos", bbsym="c3").shape == (20, 4, 4)
    assert wu.sym.frames("icos", bbsym="c5").shape == (12, 4, 4)


def test_sym():
    assert wu.sym.symframes.tetrahedral_frames.shape == (12, 4, 4)
    assert wu.sym.symframes.octahedral_frames.shape == (24, 4, 4)
    assert wu.sym.symframes.icosahedral_frames.shape == (60, 4, 4)
    x = np.concatenate(
        [
            wu.sym.symframes.tetrahedral_frames,
            wu.sym.symframes.octahedral_frames,
            wu.sym.symframes.icosahedral_frames,
        ]
    )
    assert np.all(x[..., 3, 3] == 1)
    assert np.all(x[..., 3, :3] == 0)
    assert np.all(x[..., :3, 3] == 0)


def test_sym_frames():
    assert len(wu.sym.tetrahedral_axes_all[2] == 6)
    assert len(wu.sym.tetrahedral_axes_all[3] == 4)
    assert len(wu.sym.tetrahedral_axes_all["3b"] == 4)
    assert len(wu.sym.octahedral_axes_all[2] == 12)
    assert len(wu.sym.octahedral_axes_all[3] == 8)
    assert len(wu.sym.octahedral_axes_all[4] == 6)
    assert len(wu.sym.icosahedral_axes_all[2] == 30)
    assert len(wu.sym.icosahedral_axes_all[3] == 20)
    assert len(wu.sym.icosahedral_axes_all[5] == 12)


if __name__ == "__main__":
    main()
