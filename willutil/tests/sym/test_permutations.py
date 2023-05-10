import numpy as np
import willutil as wu
import itertools
from willutil.sym.spacegroup_symelems import _compute_symelems

def main():
   # test_symelems_I432(showme=False)
   # assert 0

   # for k, v in wu.sym.sg_symelem_frame_ids_dict.items():
   # print(k, v.shape)
   # assert 0

   # WIP_P23_perm()
   # test_icos_perm()
   ic('PASS test_permutations')

def WIP_opcompid():
   f = wu.sym.frames('P23', cells=4)
   ic(wu.sym.symelems('P23'))

   for ielem, se in enumerate(wu.sym.symelems('P23')):
      fcompid = wu.sym.sg_symelem_frame444_compids_dict['P23'][:, ielem]
      fopid = se.frame_operator_ids(f)
      ids = fcompid.copy()
      for i in range(np.max(fopid)):
         fcids = fcompid[fopid == i]
         idx0 = fcompid == fcids[0]
         for fcid in fcids[1:]:
            idx = fcompid == fcid
            ids[idx] = min(min(ids[idx]), min(ids[idx0]))
      for i, id in enumerate(sorted(set(ids))):
         ids[ids == id] = i
      for i in range(max(ids)):
         ic(f[ids == i, :3, 3])
      assert 0

def WIP_P23_perm():

   frames = wu.sym.sgframes('P23', cells=4)
   # semap = wu.sym.symelems('P23')
   semap = _compute_symelems('P23')

   selems = list(itertools.chain(*semap.values()))

   perms = wu.sym.symframe_permutations_torch(frames)

   compid = -np.ones((len(frames), len(selems)), dtype=np.int32)
   for ielem, se in enumerate(selems):
      compid[:, ielem] = se.frame_component_ids(frames, perms)

   ielem = 4
   ecen, eaxs = selems[ielem].cen, selems[ielem].axis
   ic(selems[ielem])
   for icomp in range(np.max(compid[:, ielem])):
      # ic(np.max(frames[:, :3, 3]))
      selframes = compid[:, ielem] == icomp
      assert len(selframes)
      testf = frames[selframes] @ wu.htrans(ecen) @ wu.halign([0, 0, 1], eaxs)
      # ic(testf.shape)
      # print(testf[:, :3, 3])
      # wu.showme(testf)

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
