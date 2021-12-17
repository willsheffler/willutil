import os
import willutil as wu

def test_pdbread(pdbfile, pdbcontents):
   pdb1 = wu.pdb.readpdb(pdbfile)
   pdb2 = wu.pdb.readpdb(pdbcontents)
   assert all(pdb1.df == pdb2.df)
   assert pdb1.meta.cryst == pdb2.meta.cryst
   assert pdb1.sequence == pdb2.sequence
   assert pdb1.sequence == 'ELTPAVTTYKLVINGKTLKGETTTKAVDAETAEKAFKQYANDNGVDGVWTYDDATKTFTVTEMVTEVPVAZZ'

def test_load_pdbs(pdbfiles):
   seqs = [
      'ELTPAVTTYKLVINGKTLKGETTTKAVDAETAEKAFKQYANDNGVDGVWTYDDATKTFTVTEMVTEVPVAZZ',
      'DIQVQVNIDDNGKNFDYTYTVTTESELQKVLNELZDYIKKQGAKRVRISITARTKKEAEKFAAILIKVFAELGYNDINVTFDGDTVTVEGQL',
   ]
   pdbs = wu.pdb.load_pdbs(pdbfiles)
   for i in range(len(pdbs)):
      assert pdbs[i].sequence == seqs[i]

def test_find_pdb_files():
   pat = os.path.join(wu.tests.test_dat_dir, 'pdb/*.pdb1.gz')
   files = wu.pdb.find_pdb_files(pat)
   found = set(os.path.basename(f) for f in files)
   check = {'1qys.pdb1.gz', '1coi.pdb1.gz', '1pgx.pdb1.gz'}
   assert check.issubset(found)

def main():
   import willutil.tests.testdata as tdat
   test_pdbread(tdat.pdbfile(), tdat.pdbcontents())
   test_load_pdbs(tdat.pdbfiles())
   test_find_pdb_files()

if __name__ == '__main__':
   main()