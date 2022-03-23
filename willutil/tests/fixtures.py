import willutil as wu
from willutil.tests.testdata import load_test_data, test_data_path

def respairdat10_plus_xmap_rots():
   dset = load_test_data('pdb/respairdat10_plus_xmap_rots.nc')
   return wu.pdb.PdbData(dset)

def respairdat10():
   dset = load_test_data('pdb/respairdat10.nc')
   return wu.pdb.PdbData(dset)

def pdbcontents():
   return load_test_data('pdb/1pgx.pdb1.gz')

def pdbfname():
   return test_data_path('pdb/1pgx.pdb1.gz')

def pdbfnames():
   return [test_data_path('pdb/1pgx.pdb1.gz'), test_data_path('pdb/1qys.pdb1.gz')]

def three_PDBFiles():
   return [load_test_data(f'pdb/{code}.pdb1.gz.pickle') for code in ['3asl', '1pgx', '1coi']]

def pdbfile():
   return load_test_data('pdb/3asl.pdb1.gz.pickle')
