import numpy as np
import willutil as wu

def main():
   test_asugrid_P432_432D2()
   test_asugrid_P432_432()
   test_asugrid_P432_422()

   test_asugrid_P213()

   test_asugrid_I213()

   test_asugrid_I4132()

   test_asugrid_L632()

def test_asugrid_I213():
   sym = 'I 21 3'
   x = wu.sym.xtal(sym)
   frames = x.primary_frames()
   allframes = x.frames(cells=3, xtalrad=0.7)
   framesavoid = allframes[len(frames):]
   cellsize = 100
   # pos = wu.hpoint([30, 20, 20])
   pos = x.asucen(cellsize=cellsize)
   newpos, newcell = wu.sym.place_asu_grid(
      pos,
      cellsize,
      frames,
      framesavoid,
      lbub=0.2,
      lbubcell=0.1,
      nsamp=20,
      nsampcell=5,
      distcontact=(0.2, 0.24),
      distavoid=0.35,
      # distspread=2,
      clusterdist=0.05,
   )
   ref = np.array([[43.06842105, 34.64736842, 52.72105263, 1.], [43.06842105, 32.54210526, 50.61578947, 1.], [47.27894737, 36.75263158, 54.82631579, 1.], [47.27894737, 34.64736842, 52.72105263, 1.], [51.48947368, 38.85789474, 56.93157895, 1.], [45.17368421, 32.54210526, 48.51052632, 1.], [49.38421053, 36.75263158, 52.72105263, 1.], [51.48947368, 36.75263158, 54.82631579, 1.], [47.27894737, 32.54210526, 50.61578947, 1.], [53.59473684, 38.85789474, 54.82631579, 1.], [49.38421053, 34.64736842, 50.61578947, 1.], [53.59473684, 36.75263158, 52.72105263, 1.], [51.48947368, 36.75263158, 50.61578947, 1.], [47.27894737, 32.54210526, 46.40526316, 1.], [49.38421053, 32.54210526, 48.51052632, 1.], [55.7, 38.85789474, 52.72105263, 1.], [51.48947368, 34.64736842, 48.51052632, 1.], [53.59473684, 36.75263158, 48.51052632, 1.], [55.7, 38.85789474, 48.51052632, 1.]])
   assert np.allclose(ref, newpos)

def test_asugrid_P213():
   sym = 'P 21 3'
   x = wu.sym.xtal(sym)
   frames = x.primary_frames()
   allframes = x.frames(cells=3, xtalrad=0.6)
   framesavoid = allframes[len(frames):]
   cellsize = 100
   # pos = wu.hpoint([30, 20, 20])
   pos = x.asucen(cellsize=cellsize)
   newpos, newcell = wu.sym.place_asu_grid(
      pos,
      cellsize,
      frames,
      framesavoid,
      lbub=0.2,
      lbubcell=0.1,
      nsamp=20,
      nsampcell=5,
      distcontact=(0.1, 0.33),
      distavoid=0.42,
      distspread=3,
      clusterdist=0.09,
   )
   ref = np.array([[37.63684211, 18.24210526, 42.63157895, 1.], [41.84736842, 20.34736842, 44.73684211, 1.], [46.05789474, 22.45263158, 44.73684211, 1.], [41.84736842, 20.34736842, 40.52631579, 1.], [37.63684211, 16.13684211, 38.42105263, 1.], [46.05789474, 22.45263158, 40.52631579, 1.], [41.84736842, 20.34736842, 36.31578947, 1.], [50.26842105, 24.55789474, 42.63157895, 1.], [33.42631579, 14.03157895, 38.42105263, 1.], [37.63684211, 16.13684211, 34.21052632, 1.], [46.05789474, 22.45263158, 36.31578947, 1.], [50.26842105, 24.55789474, 38.42105263, 1.], [41.84736842, 18.24210526, 32.10526316, 1.], [54.47894737, 28.76842105, 42.63157895, 1.], [33.42631579, 11.92631579, 34.21052632, 1.], [43.95263158, 22.45263158, 32.10526316, 1.], [54.47894737, 28.76842105, 38.42105263, 1.], [37.63684211, 14.03157895, 30., 1.], [50.26842105, 24.55789474, 34.21052632, 1.], [52.37368421, 28.76842105, 34.21052632, 1.], [48.16315789, 24.55789474, 30., 1.], [50.26842105, 28.76842105, 30., 1.], [56.58421053, 32.97894737, 34.21052632, 1.], [54.47894737, 32.97894737, 30., 1.]])
   assert np.allclose(ref, newpos)
   return

def test_asugrid_I4132():
   sym = 'I4132_322'
   x = wu.sym.xtal(sym)
   frames = x.primary_frames()
   allframes = x.frames(cells=3, xtalrad=0.7)
   framesavoid = allframes[len(frames):]
   cellsize = 100
   # pos = wu.hpoint([30, 20, 20])
   pos = x.asucen(cellsize=cellsize)
   newpos, newcell = wu.sym.place_asu_grid(
      pos,
      cellsize,
      frames,
      framesavoid,
      lbub=0.2,
      lbubcell=0.1,
      nsamp=30,
      nsampcell=1,
      distcontact=(0.0, 0.3),
      distavoid=0.31,
      distspread=8,
      clusterdist=0.01,
   )
   ref = np.array([[-7.69576183, 3.52909483, 14.10201183, 1.], [-7.69576183, 3.52909483, 12.72270148, 1.], [-6.31645148, 3.52909483, 12.72270148, 1.], [-6.31645148, 3.52909483, 11.34339114, 1.], [-6.31645148, 2.14978448, 11.34339114, 1.], [-4.93714114, 2.14978448, 12.72270148, 1.], [-4.93714114, 2.14978448, 11.34339114, 1.]])
   assert np.allclose(newpos, ref)

def test_asugrid_L632():
   sym = 'L632'
   x = wu.sym.xtal(sym)
   frames = x.primary_frames()
   allframes = x.frames(cells=3, xtalrad=0.7)
   framesavoid = allframes[len(frames):]
   cellsize = 100
   # pos = wu.hpoint([30, 20, 20])
   pos = x.asucen(cellsize=cellsize)
   newpos, newcell = wu.sym.place_asu_grid(
      pos,
      cellsize,
      frames,
      framesavoid,
      lbub=0.2,
      lbubcell=0.2,
      nsamp=20,
      nsampcell=10,
      distcontact=(0.0, 0.5),
      distavoid=0.6,
      distspread=0.005,
      clusterdist=0.12,
   )
   ref = np.array([[21.49909241, 1.05263158, 1.05263158, 1.], [21.49909241, -5.26315789, 1.05263158, 1.], [21.49909241, 1.05263158, -5.26315789, 1.], [21.49909241, 1.05263158, 7.36842105, 1.], [21.49909241, 7.36842105, 1.05263158, 1.], [21.49909241, -5.26315789, -5.26315789, 1.], [21.49909241, -5.26315789, 7.36842105, 1.], [21.49909241, 7.36842105, -5.26315789, 1.], [21.49909241, 7.36842105, 7.36842105, 1.], [21.49909241, -1.05263158, -11.57894737, 1.], [21.49909241, 5.26315789, -11.57894737, 1.], [21.49909241, -1.05263158, 13.68421053, 1.], [21.49909241, -7.36842105, -11.57894737, 1.], [21.49909241, 5.26315789, 13.68421053, 1.], [23.60435556, 15.78947368, 1.05263158, 1.], [23.60435556, -15.78947368, 1.05263158, 1.], [21.49909241, -7.36842105, 13.68421053, 1.], [23.60435556, -15.78947368, -5.26315789, 1.], [23.60435556, 15.78947368, -5.26315789, 1.], [23.60435556, -15.78947368, 7.36842105, 1.], [23.60435556, 15.78947368, 7.36842105, 1.], [21.49909241, 1.05263158, -17.89473684, 1.], [23.60435556, 15.78947368, -11.57894737, 1.], [23.60435556, -15.78947368, -11.57894737, 1.], [21.49909241, -5.26315789, -17.89473684, 1.], [21.49909241, 7.36842105, -17.89473684, 1.], [23.60435556, -15.78947368, 13.68421053, 1.], [23.60435556, 15.78947368, 13.68421053, 1.], [21.49909241, 1.05263158, 20., 1.], [21.49909241, -5.26315789, 20., 1.], [21.49909241, 7.36842105, 20., 1.], [23.60435556, -15.78947368, -17.89473684, 1.], [23.60435556, 15.78947368, -17.89473684, 1.], [23.60435556, -15.78947368, 20., 1.], [23.60435556, 15.78947368, 20., 1.]])
   assert np.allclose(newpos, ref)

def test_asugrid_P432_422():

   sym = 'P 4 3 2 422'
   x = wu.sym.xtal(sym)
   frames = x.primary_frames(contacting_only=True)
   allframes = x.frames(cells=3, xtalrad=0.9)
   framesavoid = allframes[len(x.primary_frames()):]
   cellsize = 100
   # pos = wu.hpoint([30, 20, 20])
   # pos = x.asucen(cellsize=cellsize)
   pos = np.array([0.16, 0.36, 0.0, 1])
   newpos, newcell = wu.sym.place_asu_grid(
      pos,
      cellsize,
      frames,
      framesavoid,
      lbub=0.2,
      lbubcell=0.2,
      nsamp=60,
      nsampcell=1,
      distcontact=(0.0, 0.35),
      distavoid=0.35,
      distspread=0.06,
      clusterdist=0.12,
   )
   assert np.allclose(newpos, np.array([[16.33898305, 35.66101695, 0.33898305, 1.], [18.37288136, 35.66101695, -3.72881356, 1.], [20.40677966, 37.69491525, 0.33898305, 1.], [19.05084746, 35.66101695, 4.40677966, 1.]]))

def test_asugrid_P432_432():
   sym = 'P 4 3 2 432'
   x = wu.sym.xtal(sym)
   frames = x.primary_frames(contacting_only=True)
   allframes = x.frames(cells=3, xtalrad=0.9)
   framesavoid = allframes[len(x.primary_frames()):]
   cellsize = 100
   # pos = wu.hpoint([30, 20, 20])
   # pos = x.asucen(cellsize=cellsize)
   pos = np.array([0.2, 0.36, 0.1, 1])
   newpos, newcell = wu.sym.place_asu_grid(
      pos,
      cellsize,
      frames,
      framesavoid,
      lbub=0.15,
      lbubcell=0.2,
      nsamp=20,
      nsampcell=5,
      distcontact=(0.0, 0.35),
      distavoid=0.25,
      distspread=0.17,
      clusterdist=0.12,
   )
   assert np.allclose(newpos, np.array([[17.63157895, 32.05263158, 9.21052632, 1.], [20.78947368, 36.78947368, 9.21052632, 1.], [14.47368421, 27.31578947, 9.21052632, 1.], [16.05263158, 36.78947368, 10.78947368, 1.], [14.47368421, 33.63157895, 6.05263158, 1.], [12.89473684, 32.05263158, 10.78947368, 1.], [9.73684211, 27.31578947, 7.63157895, 1.], [20.78947368, 28.89473684, 12.36842105, 1.], [11.31578947, 28.89473684, 2.89473684, 1.], [8.15789474, 32.05263158, 9.21052632, 1.], [5., 27.31578947, 7.63157895, 1.]]))

def test_asugrid_P432_432D2():
   sym = 'P 4 3 2 432D2'
   x = wu.sym.xtal(sym)
   frames = x.primary_frames(contacting_only=True)
   allframes = x.frames(cells=3, xtalrad=0.9)
   framesavoid = allframes[len(x.primary_frames()):]
   cellsize = 100
   # pos = wu.hpoint([30, 20, 20])
   # pos = x.asucen(cellsize=cellsize)
   pos = np.array([0.2, 0.36, 0.1, 1])
   newpos, newcell = wu.sym.place_asu_grid(
      pos,
      cellsize,
      frames,
      framesavoid,
      lbub=0.15,
      lbubcell=0.2,
      nsamp=40,
      nsampcell=1,
      distcontact=(0.0, 0.35),
      distavoid=0.3,
      distspread=0.15,
      clusterdist=0.07,
   )
   assert np.allclose(newpos, np.array([[19.61538462, 35.61538462, 9.61538462, 1.], [19.61538462, 37.15384615, 11.15384615, 1.], [21.15384615, 37.15384615, 9.61538462, 1.], [21.15384615, 35.61538462, 8.07692308, 1.], [18.07692308, 37.15384615, 9.61538462, 1.], [18.07692308, 35.61538462, 8.07692308, 1.], [22.69230769, 35.61538462, 9.61538462, 1.], [21.15384615, 38.69230769, 11.15384615, 1.], [16.53846154, 35.61538462, 9.61538462, 1.], [15.76923077, 34.84615385, 7.30769231, 1.], [13.46153846, 34.84615385, 8.07692308, 1.]]))
   return

   for i in range(len(newpos)):
      # result0 = wu.hxform(wu.hscaled(newcell[i], frames), newpos[i], is_points=True)
      # wu.showme(result0, sphere=25 / 2, kind='point'
      # result1 = wu.hxform(wu.hscaled(newcell[i], framesavoid), newpos[i])
      # wu.showme(result1, sphere=3)
      f = np.concatenate([frames, allframes])
      colors = [(0, 1, 1)] + [(1, 0, 0)] * (len(frames) - 1) + [(1, 1, 1)] * (len(f))
      result = wu.hxform(wu.hscaled(newcell[i], f), newpos[i])
      wu.showme(result, sphere=9, col=colors)

if __name__ == '__main__':
   main()