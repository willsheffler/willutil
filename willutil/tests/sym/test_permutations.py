import numpy as np
import willutil as wu

def main():
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

   sym = 'P432'
   se = wu.sym.compute_symelems(sym, wu.sym.sgframes(sym, cellgeom='unit'))
   ic(sym, se)
   # showsymelems(sym, se)
   assert len(se) == 3 and len(se['C2']) == 3 and len(se['C3']) == 1 and len(se['C4']) == 2

   sym = 'I213'
   se = wu.sym.compute_symelems(sym, wu.sym.sgframes(sym, cellgeom='unit'))
   ic(sym, se)
   # showsymelems(sym, se)
   assert len(se) == 3 and len(se['C2']) == 2 and len(se['C3']) == 1 and len(se['C21']) == 2

   sym = 'P213'
   se = wu.sym.compute_symelems(sym, wu.sym.sgframes(sym, cellgeom='unit'))
   ic(sym, se)
   # se['C21'] = se['C21'][:2]
   # showsymelems(sym, se)
   assert len(se) == 2 and len(se['C3']) == 2 and len(se['C21']) == 2

   assert 0

   # wu.save(perms, '/home/sheffler/WILLUTIL_SYM_PERMS_I4132_5.pickle')
   # assert 0

def showsymelems(sym, se):
   f = wu.sym.sgframes(sym, cells=2, cellgeom=[10])
   ii = 0
   col = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]
   for i, c in enumerate(se):
      for j, s in enumerate(se[c]):
         wu.showme(f @ wu.htrans(s.cen * 10 + 0.3 * wu.hvec([0.1, 0.2, 0.3])) @ wu.halign([0, 0, 1], s.axis), xyzlen=[0.6, 0.8, 1])
         # wu.showme(f @ wu.htrans(s.cen * 10) @ wu.halign([0, 0, 1], s.axis), colors=[col[ii]], xyzlen=[0.4, 0.2 if len(c) == 2 else 0.6, 1])
         ii += 1

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
