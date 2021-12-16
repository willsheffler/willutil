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
