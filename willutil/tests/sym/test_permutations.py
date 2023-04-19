import numpy as np
import willutil as wu

def main():
   test_icos_perm()
   ic('PASS test_permutations')

def test_icos_perm():
   frames = wu.sym.frames('icos')
   perms = wu.sym.permutations('icos')
   pts = wu.hxform(frames, [1, 2, 3])
   d = wu.hnorm(pts[0] - pts)
   for i, perm in enumerate(perms):
      xinv = wu.hinv(frames[i])
      for j, p in enumerate(perm):
         assert np.allclose(xinv @ frames[p], frames[j])
      assert np.allclose(d, wu.hnorm(pts[i] - pts[perm]))

if __name__ == '__main__':
   main()
