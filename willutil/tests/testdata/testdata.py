from willutil.storage import load
from willutil.pdb import PdbData
import os

test_dat_dir = os.path.dirname(__file__)
test_pdb_dir = os.path.join(test_dat_dir, 'pdb')

def respairdat10_plus_xmap_rots():
   dset = load(os.path.join(test_pdb_dir, 'respairdat10_plus_xmap_rots.nc'))
   return PdbData(dset)

def respairdat10():
   dset = load(os.path.join(test_pdb_dir, 'respairdat10.nc'))
   return PdbData(dset)

def pdbcontents():
   return load(os.path.join(test_pdb_dir, '1pgx.pdb1.gz'))

def pdbfile():
   return os.path.join(test_pdb_dir, '1pgx.pdb1.gz')

def pdbfiles():
   return [
      os.path.join(test_pdb_dir, '1pgx.pdb1.gz'),
      os.path.join(test_pdb_dir, '1qys.pdb1.gz'),
   ]
