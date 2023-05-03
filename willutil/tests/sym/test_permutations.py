import numpy as np
import willutil as wu

def main():
   test_icos_perm()
   _test_i213_perm()
   ic('PASS test_permutations')

def _test_i213_perm():
   # frames = wu.sym.frames('I4132', sgonly=True, cells=5).astype(np.float32)
   # perms = wu.sym.symframe_permutations_torch(frames)
   # wu.save(perms, '/home/sheffler/WILLUTIL_SYM_PERMS_I4132_5.pickle')
   # assert 0

   perms = wu.load('/home/sheffler/WILLUTIL_SYM_PERMS_I4132_5_int32.pickle')
   # wu.save(perms, '/home/sheffler/WILLUTIL_SYM_PERMS_I4132_5_int32.pickle')
   ic(perms.shape)

   assert 0

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
