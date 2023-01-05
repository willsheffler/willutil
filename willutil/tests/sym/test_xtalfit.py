import numpy as np
import willutil as wu

def main():
   test_xtalfit()

def test_xtalfit():
   x = wu.sym.Xtal('L632')

   np.random.seed(7)

   pts0 = np.array([
      [18.44458833, 0.02276394, -3.29067642, 1.],
      [-5.08245051, 7.47204762, -2.46703027, 1.],
      [-2.60027436, -6.70113306, 0.52336958, 1.],
      [17.3808256, -0.92255618, -0.84246037, 1.],
   ]).reshape(-1, 1, 4)
   pts = pts0.copy()

   cellsize, cartshift = x.fit_coords(pts)
   # ic(cellsize)
   # ic(cartshift)
   assert np.allclose(cellsize, 27.423643867)
   assert np.allclose(cartshift, np.array([-2.07587708, -4.49116458, -0.81612973, 0.]))

   cellsize, cartshift = x.fit_coords(pts, noshift=True)
   # ic(cellsize)
   # ic(cartshift)
   assert np.allclose(cellsize, 35.828085361)
   assert np.allclose(cartshift, np.array([0, 0, 0, 0]))

   assert np.allclose(pts0, pts)

if __name__ == '__main__':
   main()