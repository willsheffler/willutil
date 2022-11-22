import os, glob
import numpy as np, pandas as pd
import willutil as wu

# ic.configureOutput(includeContext=True, contextAbsPath=False)

def main():
   from willutil.tests import fixtures as f
   test_pdb_renumber(f.pdb1pgx())
   test_pdb_multimodel(f.pdb1coi())
   test_pdb_mask(f.pdb1pgx())
   test_pdb_bbcoords(f.pdb1pgx())
   # assert 0, 'MAIN'
   test_pdbread(f.pdbfname(), f.pdbcontents())
   test_load_pdbs(f.pdbfnames())
   test_find_pdb_files()
   test_pdbfile(f.pdbfile())
   ic('TEST_PDBREAD DONE')

def test_pdb_multimodel(pdb1coi):
   bb = pdb1coi.bb()
   bb0 = pdb1coi.subfile(modelidx=0).bb()
   bb1 = pdb1coi.subfile(modelidx=1).bb()
   bb2 = pdb1coi.subfile(modelidx=2).bb()
   assert bb.shape == (87, 5, 3)
   assert bb0.shape == (29, 5, 3)
   assert bb1.shape == (29, 5, 3)
   assert bb2.shape == (29, 5, 3)

def test_pdb_renumber(pdb1pgx):
   p = pdb1pgx
   assert p.ri[0] == 8
   p.renumber_from_0()
   assert p.ri[0] == 0

def test_pdb_mask(pdb1pgx):
   pdb = pdb1pgx
   # print(pdbfile.df)
   # print(pdbfile.seq)
   # print(pdbfile.code)
   camask = pdb.camask()
   cbmask = pdb.cbmask(aaonly=False)
   assert np.all(np.logical_and(cbmask, camask) == cbmask)
   nca = np.sum(pdb.df.an == b'CA')
   ncb = np.sum(pdb.df.an == b'CB')
   assert np.sum(pdb.camask()) == nca
   assert np.sum(pdb.cbmask()) == ncb
   ngly = np.sum((pdb.df.rn == b'GLY') * (pdb.df.an == b'CA'))
   assert nca - ncb == ngly
   assert nca - np.sum(pdb.cbmask()) == ngly
   p = pdb.subfile(het=False)
   assert p.sequence() == pdb.sequence().replace('Z', '')

   seq = p.sequence()
   cbmask = pdb.cbmask(aaonly=True)
   # ic(len(seq), sum(cbmask))
   assert len(seq) == np.sum(camask)
   for s, m in zip(pdb.seq, cbmask):
      assert m == (s != 'G')
   # isgly = np.array(list(seq)) == 'G'
   # wgly = np.where(isgly)[0]
   # ic(wgly)
   # ic(cbmask[wgly])

def test_pdb_bbcoords(pdb1pgx):
   pdb = pdb1pgx
   bb = pdb.bb()
   assert np.all(2 > wu.hnorm(bb[:, 0] - bb[:, 1]))
   assert np.all(2 > wu.hnorm(bb[:, 1] - bb[:, 2]))
   assert np.all(2 > wu.hnorm(bb[:, 2] - bb[:, 3]))
   cbdist = wu.hnorm(bb[:, 1] - bb[:, 4])
   mask = pdb.cbmask()
   hascb = bb[:, 4, 0] < 9e8
   assert np.all(hascb == mask)
   assert np.all(2 > cbdist[hascb])

def firstlines(s, num, skip):
   count = 0
   for line in s.splitlines():
      if not line.startswith('ATOM'):
         continue
      count += 1
      if count > skip:
         print(line)
      if count == num + skip:
         break

# COLUMNS        DATA TYPE       CONTENTS
# --------------------------------------------------------------------------------
#  1 -  6        Record name     "ATOM  "
#  7 - 11        Integer         Atom serial number.
# 13 - 16        Atom            Atom name.
# 17             Character       Alternate location indicator.
# 18 - 20        Residue name    Residue name.
# 22             Character       Chain identifier.
# 23 - 26        Integer         Residue sequence number.
# 27             AChar           Code for insertion of residues.
# 31 - 38        Real(8.3)       Orthogonal coordinates for X in Angstroms.
# 39 - 46        Real(8.3)       Orthogonal coordinates for Y in Angstroms.
# 47 - 54        Real(8.3)       Orthogonal coordinates for Z in Angstroms.
# 55 - 60        Real(6.2)       Occupancy.
# 61 - 66        Real(6.2)       Temperature factor (Default = 0.0).
# 73 - 76        LString(4)      Segment identifier, left-justified.
# 77 - 78        LString(2)      Element symbol, right-justified.
# 79 - 80        LString(2)      Charge on the atom.

def test_pdbread(pdbfname, pdbcontents):
   pd.set_option("display.max_rows", None, "display.max_columns", None)

   foo = (
      #    5    10   15   20   25   30   35   40   45   50   55   60   65   70   75   80
      #    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |
      #hhhhhhiiiii_aaaaLrrr_CiiiiI___xxxxxxxxyyyyyyyyzzzzzzzzoooooobbbbbb      ssssEEcc
      'HETATM12345 ATOM RES C 1234   1236.8572215.5813376.721440.50547.32      SEGIPBCH\n' +
      'ATOM1234567 ATOM RES C 1234   1236.8572215.5813376.721440.50547.32      SEGIPBCH\n')
   pdb = wu.pdb.readpdb(foo)
   assert all(pdb.df.columns == ['het', 'ai', 'an', 'rn', 'ch', 'ri', 'x', 'y', 'z', 'occ', 'bfac', 'elem', 'mdl'])
   assert pdb.df.shape == (2, 13)
   assert all(pdb.df.ai == (12345, 1234567))

   # num, skip = 1, 0
   # firstlines(pdbcontents, num, skip)

   pdb2 = wu.pdb.readpdb(pdbcontents)
   pdb1 = wu.pdb.readpdb(pdbfname)
   assert all(pdb1.df == pdb2.df)
   assert pdb1.cryst1 == pdb2.cryst1
   assert pdb1.seq == pdb2.seq

   assert pdb1.seq == 'ELTPAVTTYKLVINGKTLKGETTTKAVDAETAEKAFKQYANDNGVDGVWTYDDATKTFTVTEMVTEVPVA'
   # print(pdbcontents)
   # for c in pdb1.df.columns:
   # print(pdb1.df[c][60])
   types = [type(_) for _ in pdb1.df.loc[0]]
   for i in range(len(pdb1.df)):
      assert types == [type(_) for _ in pdb1.df.loc[i]]

def test_load_pdbs(pdbfnames):
   seqs = [
      'ELTPAVTTYKLVINGKTLKGETTTKAVDAETAEKAFKQYANDNGVDGVWTYDDATKTFTVTEMVTEVPVA',
      'DIQVQVNIDDNGKNFDYTYTVTTESELQKVLNELZDYIKKQGAKRVRISITARTKKEAEKFAAILIKVFAELGYNDINVTFDGDTVTVEGQL',
   ]
   pdbs = wu.pdb.load_pdbs(pdbfnames, cache=False, pbar=False)
   assert set(pdbs.keys()) == set(pdbfnames)
   for i, fname in enumerate(pdbs):
      assert pdbs[fname].seqhet == seqs[i]

def test_find_pdb_files():
   pat = os.path.join(wu.tests.test_data_dir, 'pdb/*.pdb1.gz')
   files = wu.pdb.find_pdb_files(pat)
   found = set(os.path.basename(f) for f in files)
   check = {'1qys.pdb1.gz', '1coi.pdb1.gz', '1pgx.pdb1.gz'}
   assert check.issubset(found)

def test_pdbfile(pdbfile):
   # print(pdbfile.df)
   # ic(pdbfile.nreshet)
   assert pdbfile.nreshet == 85
   a = pdbfile.subfile('A')
   b = pdbfile.subfile('B')
   assert a.nres + b.nres == pdbfile.nres
   assert np.all(a.df.ch == b'A')
   assert np.all(b.df.ch == b'B')

if __name__ == '__main__':
   main()
