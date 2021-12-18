import os, glob
import numpy as np
import willutil as wu

def test_pdbread(pdbfname, pdbcontents):
   pdb1 = wu.pdb.readpdb(pdbfname)
   pdb2 = wu.pdb.readpdb(pdbcontents)
   assert all(pdb1.df == pdb2.df)
   assert pdb1.cryst1 == pdb2.cryst1
   assert pdb1.seq == pdb2.seq
   assert pdb1.seq == 'ELTPAVTTYKLVINGKTLKGETTTKAVDAETAEKAFKQYANDNGVDGVWTYDDATKTFTVTEMVTEVPVAZZ'
   # print(pdb1.df.head(61))
   # for c in pdb1.df.columns:
   # print(pdb1.df[c][60])
   # types = [type(_) for _ in pdb1.df.loc[0]]
   # for i in range(len(pdb1.df)):
   # assert types == [type(_) for _ in pdb1.df.loc[i]]
   # print(pdb1.df)

def test_load_pdbs(pdbfnames):
   seqs = [
      'ELTPAVTTYKLVINGKTLKGETTTKAVDAETAEKAFKQYANDNGVDGVWTYDDATKTFTVTEMVTEVPVAZZ',
      'DIQVQVNIDDNGKNFDYTYTVTTESELQKVLNELZDYIKKQGAKRVRISITARTKKEAEKFAAILIKVFAELGYNDINVTFDGDTVTVEGQL',
   ]
   pdbs = wu.pdb.load_pdbs(pdbfnames, cache=False, pbar=False)
   assert set(pdbs.keys()) == set(pdbfnames)
   for i, fname in enumerate(pdbs):
      assert pdbs[fname].seq == seqs[i]

def test_find_pdb_files():
   pat = os.path.join(wu.tests.test_data_dir, 'pdb/*.pdb1.gz')
   files = wu.pdb.find_pdb_files(pat)
   found = set(os.path.basename(f) for f in files)
   check = {'1qys.pdb1.gz', '1coi.pdb1.gz', '1pgx.pdb1.gz'}
   assert check.issubset(found)

def test_pdbfile(pdbfile):
   assert pdbfile.nres == 85
   a = pdbfile.subfile('A')
   b = pdbfile.subfile('B')
   assert a.nres + b.nres == pdbfile.nres
   assert np.all(a.df.ch == b'A')
   assert np.all(b.df.ch == b'B')

def main():
   from willutil.tests import fixtures
   test_pdbread(fixtures.pdbfname(), fixtures.pdbcontents())
   test_load_pdbs(fixtures.pdbfnames())
   test_find_pdb_files()
   test_pdbfile(fixtures.pdbfile())

   # with wu.Timer():
   #    ps = wu.pdb.load_pdbs('/home/sheffler/data/rcsb/divided/as/*.pdb?.gz', skip_errors=True,
   #                          maxsize=50_000)
   #    for fn, pf in ps.items():
   #       if pf.nres < 2:
   #          continue
   #       print(pf.nchain, pf.nres, pf.code)

if __name__ == '__main__':
   main()