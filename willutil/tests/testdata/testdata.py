import os
import willutil as wu
from willutil.pdb import PdbData

test_data_dir = os.path.dirname(__file__)
test_pdb_dir = os.path.join(test_data_dir, 'pdb')

def test_data_path(fname):
   return os.path.join(test_data_dir, fname)

def load_test_data(fname):
   return wu.load(os.path.join(test_data_dir, fname))

def save_test_data(stuff, fname):
   return wu.save(stuff, os.path.join(test_data_dir, fname))

class TestFixtures:
   def __init__(self):
      pass

   def respairdat10_plus_xmap_rots():
      dset = load_test_data('pdb/respairdat10_plus_xmap_rots.nc')
      return PdbData(dset)

   def respairdat10():
      dset = load_test_data('pdb/respairdat10.nc')
      return PdbData(dset)

   def pdbcontents():
      return load_test_data('pdb/1pgx.pdb1.gz')

   def pdbfname():
      return test_data_path('pdb/1pgx.pdb1.gz')

   def pdbfnames():
      return [test_data_path('pdb/1pgx.pdb1.gz'), test_data_path('pdb/1qys.pdb1.gz')]

   def three_PDBFiles():
      return load_test_data('pdb/three_PDBFiles.pickle')

   def pdbfile():
      return load_test_data('pdb/pdbfile.pickle')

fixtures = TestFixtures
