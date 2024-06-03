import willutil as wu
from willutil.tests.testdata import load_test_data, test_data_path


def respairdat10_plus_xmap_rots():
    dset = load_test_data("pdb/respairdat10_plus_xmap_rots.nc")
    return wu.pdb.PdbData(dset)


def respairdat10():
    dset = load_test_data("pdb/respairdat10.nc")
    return wu.pdb.PdbData(dset)


def pdbcontents():
    return load_test_data("pdb/1pgx.pdb1.gz")


def pdbfname():
    return test_data_path("pdb/1pgx.pdb1.gz")


def pdbfnames():
    return [test_data_path("pdb/1pgx.pdb1.gz"), test_data_path("pdb/1qys.pdb1.gz")]


def three_PDBFiles():
    return [wu.pdb.readpdb(test_data_path(f"pdb/{code}.pdb1.gz")) for code in ["3asl", "1pgx", "1coi"]]
    # return [load_test_data(f'pdb/{code}.pdb1.gz.pickle') for code in ['3asl', '1pgx', '1coi']]


def pdbfile():
    return wu.pdb.readpdb(test_data_path("pdb/3asl.pdb1.gz"))
    # return load_test_data('pdb/3asl.pdb1.gz.pickle')


def pdb1pgx():
    return wu.pdb.readpdb(test_data_path("pdb/1pgx.pdb1.gz"))
    # return load_test_data('pdb/1pgx.pdb1.gz.pickle')


def pdb1coi():
    return wu.pdb.readpdb(test_data_path("pdb/1coi.pdb1.gz"))
    # return load_test_data('pdb/1coi.pdb1.gz.pickle')


def pdb1qys():
    return wu.pdb.readpdb(test_data_path("pdb/1qys.pdb1.gz"))
    # return load_test_data('pdb/1qys.pdb1.gz.pickle')


def ncac():
    return wu.pdb.readpdb(pdbfname()).ncac()
    # pdb = pdb.subset(atomnames=['N', 'CA', 'C'], chains=['A'])
    # xyz = np.stack([pdb.df['x'], pdb.df['y'], pdb.df['z']]).T.reshape(-1, 3, 3)
    # return xyz


def ncaco():
    return wu.pdb.readpdb(pdbfname()).ncaco()
    # pdb = pdb.subset(het=False, atomnames=['N', 'CA', 'C', 'O'], chains=['A'])
    # xyz = np.stack([pdb.df['x'], pdb.df['y'], pdb.df['z']]).T.reshape(-1, 4, 3)
    # return xyz
