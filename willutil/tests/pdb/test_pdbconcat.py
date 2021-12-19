import pytest
import willutil as wu

@pytest.mark.skip
def test_pdb_concat(three_PDBFiles):
   print(three_PDBFiles)

def main():
   # pat = os.path.join(wu.tests.test_data_dir, 'pdb/*.pdb1.gz')
   # wu.tests.save_test_data(wu.pdb.load_pdbs(pat), 'pdb/three_PDBFiles.pickle')

   pdbfiles = wu.tests.fixtures.three_PDBFiles()
   test_pdb_concat(pdbfiles)

if __name__ == '__main__':
   main()