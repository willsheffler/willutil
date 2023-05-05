import numpy as np
import willutil as wu

def main():
   # test_symelems_I432(showme=False)
   # assert 0

   test_symelems_P4232(showme=True)

   test_symelems_P23()
   test_symelems_F23()
   test_symelems_I23()

   test_symelems_P213()
   test_symelems_I213()

   test_symelems_P432()

   WIP_i4132_perm()
   test_icos_perm()

   ic('PASS test_permutations')

def WIP_i4132_perm():

   # sym = 'I4132'
   # frames = wu.sym.frames('I4132', sgonly=True, cells=5)
   # perms = wu.sym.symframe_permutations_torch(frames)
   # perms = wu.load('/home/sheffler/WILLUTIL_SYM_PERMS_I4132_5_int32.pickle')
   # unitframes = wu.sym.sgframes('I4132', cellgeom='unit')
   # ic(unitframes.shape)

   # f = wu.sym.sgframes('I213', cells=2, cellgeom=[10])
   # wu.showme(f @ wu.htrans([0, 0, 0]) @ wu.halign([0, 0, 1], [1, 1, 1]),**vizopt)
   # wu.showme(f @ wu.htrans([5, -5, 0]) @ wu.halign([0, 0, 1], [1, -1, 1]),**vizopt)
   # wu.showme(f @ wu.halign([0, 0, 1], [1, -1, 1]),**vizopt)

   # wu.save(perms, '/home/sheffler/WILLUTIL_SYM_PERMS_I4132_5.pickle')
   # assert 0

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
