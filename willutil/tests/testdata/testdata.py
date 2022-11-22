import os
import willutil as wu

test_data_dir = os.path.dirname(__file__)
test_pdb_dir = os.path.join(test_data_dir, 'pdb')

def test_data_path(fname):
   return os.path.join(test_data_dir, fname)

def load_test_data(fname):
   try:
      return wu.load(os.path.join(test_data_dir, fname))
   except FileNotFoundError as e:
      if os.path.dirname(fname) == 'pdb':
         return make_pdb_pickle(fname)
      raise e

def save_test_data(stuff, fname):
   return wu.save(stuff, os.path.join(test_data_dir, fname))

def make_pdb_pickle(fname):
   print('Creating PDBFile pickle', fname)
   pdbfname = os.path.join(test_data_dir, fname.replace('.pickle', ''))
   assert os.path.exists(pdbfname)
   pdb = wu.pdb.readpdb(pdbfname, True)
   save_test_data(pdb, fname)
   return pdb
