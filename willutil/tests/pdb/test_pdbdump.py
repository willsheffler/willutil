import tempfile, glob
import numpy as np
import willutil as wu

def main():
   from willutil.tests import fixtures as f
   test_pdbdump(f.pdb1pgx())
   test_pdbdump_ncac(f.pdb1pgx())
   test_pdbdump_sequential()
   ic('test_pdbdump.py DONE')

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
      xyz, mask = pdb1pgx.coords()
      xyz, mask = xyz[10:20], mask[10:20]
      wu.dumppdb(fname, xyz, mask)
      newpdb = wu.readpdb(fname)
      newxyz, mask = newpdb.coords()
      assert mask.dtype == bool
      assert np.all(mask[:, :4] == 1)
      assert np.sum(mask[:, 4] == 1) == 8
      # ic(xyz[:, 4])
      # ic(newxyz[:, 4])
      assert np.allclose(xyz, newxyz, atol=0.002)
      xyz = pdb1pgx.bb()[:13]
      wu.dumppdb(fname, xyz)
      newpdb = wu.readpdb(fname)
      newxyz, mask = newpdb.coords('n ca c'.split())
      assert np.all(mask[:, :3] == 1)
      assert np.allclose(xyz[:, :3], newxyz, atol=0.002)

def test_pdbdump_ncac(pdb1pgx):
   pdb = pdb1pgx.subset(het=False)
   xyz, mask = pdb.coords()[:10]
   with tempfile.TemporaryDirectory() as d:
      fname = f'{d}/xyz.pdb'
      fname = f'xyz.pdb'
      wu.dumppdb(fname, xyz, mask)

if __name__ == '__main__':
   main()
