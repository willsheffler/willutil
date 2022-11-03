import itertools
import numpy as np
import willutil as wu

def test_symelem(headless=True):
   elem1 = wu.homog.SymElem(2, [1, 0, 0], [0, 0, 0])
   elem2 = wu.homog.SymElem(2, [1, 0, 0], [0, 10, 0])

   x = wu.hrand()
   e2 = wu.hxform(x, elem1)
   assert np.allclose(e2.coords, wu.hxform(x, elem1.coords))

   # x = wu.hrand()
   # e2 = wu.hxform(x, elem1)
   # assert np.allclose(e2.coords, wu.hxform(x, elem1.coords))

   wu.showme(elem1, headless=headless)
   wu.showme(wu.hxform(wu.hrot([0, 1, 0], 120, [0, 0, 1]), elem1), headless=headless)
   # wu.showme([elem1], fancover=0.8)

def _test_hxtal_viz(headless=True):
   elem1 = wu.homog.SymElem(2, [1, 0, 0], [0, 0.25, 0.0])
   elem2 = wu.homog.SymElem(3, [1, 1, 1], [0, 0, 0])
   xtal = wu.homog.Xtal([elem1, elem2])
   for a, b, c in itertools.product(*[(0, 1)] * 3):
      # wu.showme(xtal, cellshift=[a, b, c], showgenframes=a == b == c == 0)
      wu.showme(xtal, cellshift=[a, b, c], headless=headless)

def _test_hxtal_viz_gyroid(headless=True):
   elem1 = wu.homog.SymElem(2, [1, 0, 0], [0, 0.25, 0.0])
   elem2 = wu.homog.SymElem(3, [1, 1, 1], [0, 0, 0])
   xtal = wu.homog.Xtal([elem1, elem2])
   for a, b, c in itertools.product(*[(0, 1)] * 3):
      # wu.showme(xtal, cellshift=[a, b, c], showgenframes=a == b == c == 0)
      wu.showme(xtal, cellshift=[a, b, c], headless=headless)

def main():
   # test_symelem(headless=False)
   # _test_hxtal_viz(headless=False)
   _test_hxtal_viz_gyroid(headless=False)

if __name__ == '__main__':
   main()