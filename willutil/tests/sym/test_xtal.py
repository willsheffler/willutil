import itertools, tempfile
import numpy as np
import pytest
import willutil as wu

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

def test_xtal_cryst1(headless=True):
   pymol = pytest.importorskip('pymol')
   xtal = wu.sym.Xtal('P 2 3')

   # wu.showme(xtal, showgenframes=True)

   cellsize = 100
   crd = cellsize * np.array([
      [0.28, 0.13, 0.13],
      [0.28, 0.16, 0.13],
      [0.28, 0.13, 0.15],
   ])
   # xtal.dump_pdb('test1.pdb', crd, cellsize=cellsize, cells=1)
   # xtal.dump_pdb('test2.pdb', crd, cellsize=cellsize, cells=None)
   # assert 0

   with tempfile.TemporaryDirectory() as tmpdir:
      fname = f'{tmpdir}/test.pdb'
      xtal.dump_pdb(fname, crd, cellsize=cellsize)
      pymol.cmd.load(fname)
      print(pymol.cmd.get_object_list())
      pymol.cmd.symexp('pref', 'test', 'all', 9e9)
      coords = pymol.cmd.get_coords()
      coords2 = xtal.symcoords(crd, cellsize=cellsize, cells=5)
      coords2 = coords2[..., :3].round()

      ic(np.max(coords % 1))
      ic(np.max(coords2.round() % 1))
      coords = coords.round().astype('i')
      coords2 = coords2.round().astype('i')
      ic(coords)
      ic(coords2)
      ic(np.min(coords), np.min(coords2))
      ic(np.max(coords), np.max(coords2))
      tup1 = [tuple(x) for x in coords]
      tup2 = [tuple(x) for x in coords2]
      s1 = set(tup1)
      s2 = set(tup2)
      ic(len(coords))
      ic(len(s1.intersection(s2)))
      ic(len(s1 - s2))
      ic(len(s2 - s1))

      assert set(tup1) == set(tup2)
      ncoord = 27 * 3 * xtal.nsub
      assert len(coords) == ncoord
      assert len(coords2) == ncoord

   ic('test_xtal_cryst1 DONE')

def test_hxtal_viz(headless=True):
   pymol = pytest.importorskip('pymol')
   xtal = wu.sym.Xtal('I 21 3')
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

def main():
   # test_xtal_cellframes()
   test_xtal_cryst1(False)
   # test_symelem(headless=False)
   # test_hxtal_viz(headless=False)
   # _test_hxtal_viz_gyroid(headless=False)
   ic('test_xtal.py DONE')

if __name__ == '__main__':
   main()