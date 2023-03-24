import willutil as wu

def main():
   test_asugrid()

def test_asugrid():
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
   ic(newpos.shape)
   return
   for i in range(len(newpos)):
      # result0 = wu.hxform(wu.hscaled(newcell[i], frames), newpos[i], is_points=True)
      # wu.showme(result0, sphere=25 / 2, is_points=True)
      # result1 = wu.hxform(wu.hscaled(newcell[i], framesavoid), newpos[i])
      # wu.showme(result1, sphere=3)
      result = wu.hxform(wu.hscaled(newcell[i], allframes), newpos[i])
      wu.showme(result, sphere=3)

if __name__ == '__main__':
   main()