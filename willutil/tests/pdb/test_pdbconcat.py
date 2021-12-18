import willutil as wu

def test_pdb_concat(three_PDBFiles):
   print(three_PDBFiles)
   assert 0

def main():
   # pat = os.path.join(wu.tests.test_data_dir, 'pdb/*.pdb1.gz')
   # wu.tests.save_test_data(wu.pdb.load_pdbs(pat), 'pdb/three_PDBFiles.pickle')

   pdbfiles = wu.tests.three_PDBFiles()
   test_pdb_concat(pdbfiles)

if __name__ == '__main__':
   main()