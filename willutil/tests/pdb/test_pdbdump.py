import tempfile, glob
import numpy as np
import willutil as wu

def main():

   # ic(wu.pdb.pdbdump.dump_pdb_nchain_nres_natom((100, 5)))
   # assert wu.pdb.pdbdump.dump_pdb_nchain_nres_natom((100, 5)) == (1, 100, 5)
   # assert 0
   test_dump_pdb_nchain_nres_natom()
   from willutil.tests import fixtures as f
   test_pdbdump(f.pdb1pgx())
   test_pdbdump_ncac(f.pdb1pgx())
   test_pdbdump_sequential()
   ic('test_pdbdump.py DONE')

def test_dump_pdb_nchain_nres_natom():
   assert wu.pdb.pdbdump.dump_pdb_nchain_nres_natom((10, 11, 12)) == (10, 11, 12)
   assert wu.pdb.pdbdump.dump_pdb_nchain_nres_natom((11, ), nchain=3, nresatom=5) == (3, 11, 5)
   assert wu.pdb.pdbdump.dump_pdb_nchain_nres_natom((11, ), nres=3, nresatom=5) == (11, 3, 5)
   assert wu.pdb.pdbdump.dump_pdb_nchain_nres_natom((11, ), nchain=3, nres=5) == (3, 5, 11)
   assert wu.pdb.pdbdump.dump_pdb_nchain_nres_natom((11, 13), nchain=4) == (4, 11, 13)
   assert wu.pdb.pdbdump.dump_pdb_nchain_nres_natom((11, 13), nres=4) == (11, 4, 13)
   assert wu.pdb.pdbdump.dump_pdb_nchain_nres_natom((11, 13), nresatom=4) == (11, 13, 4)
   assert wu.pdb.pdbdump.dump_pdb_nchain_nres_natom(nchain=3, nres=5, nresatom=7) == (3, 5, 7)

   assert wu.pdb.pdbdump.dump_pdb_nchain_nres_natom(shape=(20, ), nchain=5) == (5, 4, 1)
   assert wu.pdb.pdbdump.dump_pdb_nchain_nres_natom(shape=(20, ), nres=5) == (1, 5, 4)
   assert wu.pdb.pdbdump.dump_pdb_nchain_nres_natom(shape=(20, ), nresatom=5) == (1, 4, 5)

   ic(wu.pdb.pdbdump.dump_pdb_nchain_nres_natom(shape=(4, 15), nchain=4, nresatom=5))
   assert wu.pdb.pdbdump.dump_pdb_nchain_nres_natom(shape=(4, 15), nchain=4, nresatom=3) == (4, 5, 3)

def test_pdbdump_sequential():
   with tempfile.TemporaryDirectory() as d:
      xyz = wu.hrandpoint(10)
      wu.dumppdb(f'{d}/bar', xyz)
      wu.dumppdb(f'{d}/bar', xyz)
      wu.dumppdb(f'{d}/bar', xyz)
      wu.dumppdb(f'{d}/bar', xyz)
      assert len(glob.glob(f'{d}/bar*.pdb')) == 4
      wu.dumppdb(f'{d}/foo%04i_fooo.pdb', xyz)
      wu.dumppdb(f'{d}/foo%04i_fooo.pdb', xyz)
      wu.dumppdb(f'{d}/foo%04i_fooo.pdb', xyz)
      assert len(glob.glob(f'{d}/foo*_fooo.pdb')) == 3

def test_pdbdump(pdb1pgx):
   with tempfile.TemporaryDirectory() as d:
      # fname = f'{d}/xyz.pdb'
      fname = f'xyz.pdb'
      xyz, mask = pdb1pgx.atomcoords()
      xyz, mask = xyz[10:20], mask[10:20]
      wu.dumppdb(fname, xyz, mask, nchain=1)
      newpdb = wu.readpdb(fname)
      # ic(newpdb.df)
      newxyz, mask = newpdb.atomcoords()
      assert mask.dtype == bool
      assert np.all(mask[:, :4] == 1)
      assert np.sum(mask[:, 4] == 1) == 8
      # ic(xyz[:, 4])
      # ic(newxyz[:, 4])
      assert np.allclose(xyz, newxyz, atol=0.002)
      xyz = pdb1pgx.bb()[:13]
      wu.dumppdb(fname, xyz, nchain=1)
      newpdb = wu.readpdb(fname)
      newxyz, mask = newpdb.atomcoords('n ca c'.split())
      assert np.all(mask[:, :3] == 1)
      assert np.allclose(xyz[:, :3], newxyz, atol=0.002)

def test_pdbdump_ncac(pdb1pgx):
   pdb = pdb1pgx.subset(het=False)
   xyz, mask = pdb.atomcoords()[:10]
   with tempfile.TemporaryDirectory() as d:
      fname = f'{d}/xyz.pdb'
      fname = f'xyz.pdb'
      wu.dumppdb(fname, xyz, mask)

if __name__ == '__main__':
   main()
