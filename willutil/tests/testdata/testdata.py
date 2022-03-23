import os
import willutil as wu

test_data_dir = os.path.dirname(__file__)
test_pdb_dir = os.path.join(test_data_dir, 'pdb')

def test_data_path(fname):
   return os.path.join(test_data_dir, fname)

def load_test_data(fname):
   return wu.load(os.path.join(test_data_dir, fname))

def save_test_data(stuff, fname):
   return wu.save(stuff, os.path.join(test_data_dir, fname))
