import numpy as np
import willutil as wu

def main():
   check_frame_opids()

def check_frame_opids():
   sym = 'I4132'
   elems = wu.sym.symelems(sym, cyclic=False, screws=False)
   for i, elem in enumerate(elems):
      if not elem.issues: continue

      elem = elems[1]
      ic(elem)
      ic(elem.kind)
      ic(elem.isoct)
      ic(elem.cen)

      frames = wu.sym.sgframes(sym, cells=4)
      frames2 = wu.sym.sgframes(sym, cells=2)
      perms = wu.sym.sgpermutations(sym, cells=4)

      scale = 20
      # wu.showme(elem, scale=scale, name='ref', symelemscale=5)
      offset = wu.htrans([.02, .025, .03])
      # wu.showme(elem.operators @ elem.origin @ offset, scale=scale)
      # wu.showme(wu.hscaled(scale, elem.cen))
      # wu.sym.showsymelems(sym, [elem], scale=scale)
      # wu.showme(elem.operators @ offset, scale=scale)
      # wu.showme(frames2 @ offset, scale=scale)

      compids = elem.frame_component_ids_bycenter(frames, sanitycheck=False)
      opids = elem.frame_operator_ids(frames)
      opcompids = elem.make_operator_component_joint_ids(frames, opids, compids)

      import pymol

      # ids = opids
      # ids = compids
      ids = opcompids
      offset = wu.htrans([.002, .0025, .003])
      seenit = np.empty((0, 4))
      # for i in range(np.max(ids)):
      for i in range(100):
         assert np.sum(ids == i) > 0
         compframes = frames[ids == i]
         wu.showme(compframes @ elem.origin @ offset, scale=scale)
         cens = einsum('fij,j->fi', compframes, elem.origin[:, 3])
         assert not np.any(np.all(np.isclose(cens[None], seenit[:, None]), axis=2))
         seenit = np.concatenate([cens, seenit])

   assert 0

if __name__ == '__main__':
   main()
