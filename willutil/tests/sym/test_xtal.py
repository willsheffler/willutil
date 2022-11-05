import itertools, tempfile
import numpy as np
import pytest
import willutil as wu

def main():
   # test_hxtal_viz(spacegroup='I 41 3 2', headless=False)
   # test_hxtal_viz(spacegroup='I 21 3', headless=False)
   # test_hxtal_viz(spacegroup='P 2 3', headless=False)
   test_xtal_cellframes()
   test_xtal_cryst1_I_21_3(False)  #, dump_pdbs=True)
   test_xtal_cryst1_P_2_3(False)
   # test_symelem(headless=False)

   # _test_hxtal_viz_gyroid(headless=False)
   ic('test_xtal.py DONE')

def test_symelem(headless=True):
   elem1 = wu.sym.SymElem(2, [1, 0, 0], [0, 0, 0])
   elem2 = wu.sym.SymElem(2, [1, 0, 0], [0, 10, 0])

   x = wu.hrand()
   e2 = wu.hxform(x, elem1)
   assert np.allclose(e2.coords, wu.hxform(x, elem1.coords))

   # x = wu.hrand()
   # e2 = wu.hxform(x, elem1)
   # assert np.allclose(e2.coords, wu.hxform(x, elem1.coords))

   wu.showme(elem1, headless=headless)
   wu.showme(wu.hxform(wu.hrot([0, 1, 0], 120, [0, 0, 1]), elem1), headless=headless)
   # wu.showme([elem1], fancover=0.8)

def test_xtal_cellframes():
   xtal = wu.sym.Xtal('P 2 3')
   assert xtal.nsub == 12
   assert len(xtal.cellframes(cellsize=1, cells=1)) == xtal.nsub
   assert len(xtal.cellframes(cellsize=1, cells=None)) == 1
   assert len(xtal.cellframes(cellsize=1, cells=2)) == 8 * xtal.nsub
   assert len(xtal.cellframes(cellsize=1, cells=3)) == 27 * xtal.nsub
   assert len(xtal.cellframes(cellsize=1, cells=4)) == 64 * xtal.nsub
   assert len(xtal.cellframes(cellsize=1, cells=5)) == 125 * xtal.nsub

def test_xtal_cryst1_P_2_3(*args, **kw):
   helper_test_xtal_cryst1('P 2 3', *args, **kw)

def test_xtal_cryst1_I_21_3(*args, **kw):
   helper_test_xtal_cryst1('I 21 3', *args, **kw)

def prune_bbox(coords, lb, ub):
   inboundslow = np.all(coords >= lb - 0.001, axis=-1)
   inboundshigh = np.all(coords <= ub + 0.001, axis=-1)
   inbounds = np.logical_and(inboundslow, inboundshigh)
   return inbounds

def helper_test_xtal_cryst1(spacegroup, headless=True, dump_pdbs=False):
   pymol = pytest.importorskip('pymol')
   xtal = wu.sym.Xtal(spacegroup)

   # wu.showme(xtal, showgenframes=True)

   cellsize = 100
   crd = cellsize * np.array([
      [0.28, 0.13, 0.13],
      [0.28, 0.16, 0.13],
      [0.28, 0.13, 0.15],
   ])

   if dump_pdbs:
      xtal.dump_pdb('test1.pdb', crd, cellsize=cellsize, cells=1)
      xtal.dump_pdb('test2.pdb', crd, cellsize=cellsize, cells=None)
      assert 0

   with tempfile.TemporaryDirectory() as tmpdir:
      pymol.cmd.delete('all')
      fname = f'{tmpdir}/test.pdb'
      xtal.dump_pdb(fname, crd, cellsize=cellsize)
      pymol.cmd.load(fname)
      pymol.cmd.symexp('pref', 'test', 'all', 9e9)
      coords1 = pymol.cmd.get_coords()
      pymol.cmd.delete('all')

   coords2 = xtal.symcoords(crd, cellsize=cellsize, cells=(-2, 1))
   assert len(coords1) == 27 * 3 * xtal.nsub
   assert len(coords2) == 64 * 3 * xtal.nsub
   coords1 = coords1.round().astype('i')
   coords2 = coords2.round().astype('i')[..., :3]

   s1 = set([tuple(x) for x in coords1])
   s2 = set([tuple(x) for x in coords2])
   expected_ratio = (4**3 - 3**3) / 3**3
   assert len(s2 - s1) == (64 - 27) * 3 * xtal.nsub
   assert len(s1 - s2) == 0, f'canonical frames mismatch {spacegroup}'
   assert len(s1.intersection(s2)) == len(coords1), f'canonical frames mismatch {spacegroup}'

def test_hxtal_viz(headless=True, spacegroup='P 2 3'):
   pymol = pytest.importorskip('pymol')
   xtal = wu.sym.Xtal(spacegroup)
   ic(xtal.unitframes.shape)
   wu.showme(
      xtal,
      headless=headless,
      showpoints=1,
      showgenframes=False,
      cells=1,
      symelemscale=0.7,
      pointscale=0.8,
      pointshift=(0.0, 0.1, 0.0),
   )

   # elem1 = wu.sym.SymElem(2, [1, 0, 0], [0, 0.25, 0.0])
   # elem2 = wu.sym.SymElem(3, [1, 1, -1], [0, 0, 0])
   # xtal = wu.sym.Xtal([elem1, elem2])
   # for a, b, c in itertools.product(*[(0, 1)] * 3):
   #    # wu.showme(xtal, cellshift=[a, b, c], showgenframes=a == b == c == 0)
   #    wu.showme(xtal, cellshift=[a, b, c], headless=headless)

def _test_hxtal_viz_gyroid(headless=True):
   # elem1 = wu.sym.SymElem(2, [1, 0, 0], [0, 0.25, 0.0])
   # elem2 = wu.sym.SymElem(3, [1, 1, 1], [0, 0, 0])
   # xtal = wu.sym.Xtal([elem1, elem2])

   # xtal = wu.sym.Xtal('I 21 3')

   # wu.showme(xtal, headless=headless, fanshift=[-0.03, 0.05], fansize=[0.15, 0.12])
   # wu.showme(xtal, headless=headless, fanshift=[-0.03, 0.05], fansize=[0.15, 0.12])
   wu.showme(xtal, headless=headless, showpoints=1)
   #for a, b, c in itertools.product(*[(0, 1)] * 3):
   #   # wu.showme(xtal, cellshift=[a, b, c], showgenframes=a == b == c == 0)
   #   wu.showme(xtal, cellshift=[a, b, c], headless=headless, fanshift=[-0.03, 0.05],
   #             fansize=[0.15, 0.12])

if __name__ == '__main__':
   main()