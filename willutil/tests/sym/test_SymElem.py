import numpy as np
import willutil as wu
from willutil.sym.SymElem import _make_operator_component_joint_ids, ComponentIDError
from willutil.sym.spacegroup_symelems import _compute_symelems, _find_compound_symelems
from willutil.sym.permutations import symframe_permutations_torch

def main():
   # mcdock_bug1()
   # assert 0
   check_frame_opids()

def mcdock_bug1():
   sym = 'I4132'
   elems = wu.sym.symelems(sym)
   ic(elems)

   frames4 = wu.sym.frames(sym, sgonly=True, cells=4)
   f = frames4[wu.sym.sg_symelem_frame444_opcompids_dict[sym][:, 1, 1] == 109]
   wu.showme(f, scale=10)
   ic(f)

def check_frame_opids():
   sym = 'P3'

   unitframes = wu.sym.sgframes(sym, cellgeom='unit')
   n_std_frames = 4
   n_min_frames = 2
   frames = wu.sym.sgframes(sym, cells=n_std_frames, cellgeom='nonsingular')
   frames2 = wu.sym.sgframes(sym, cells=n_std_frames - 2, cellgeom='nonsingular')

   lattice = wu.sym.lattice_vectors(sym, cellgeom='nonsingular')
   ic(lattice)
   ic(unitframes.shape)
   ic(frames.shape)

   elems = _compute_symelems(sym, aslist=True)
   # for e in elems:
   # ic(e)
   # elems = _check_alternate_elems(sym, lattice, elems, frames, frames2)
   # for e in elems:
   # ic(e)
   # assert 0
   celems = _find_compound_symelems(sym, elems, frames, aslist=True)
   for e in elems + celems:
      ic(e)
      # wu.showme(e.tolattice(lattice), scale=10)
   # perms = wu.sym.sgpermutations(sym, cells=4)

   scale = 10

   # wu.showme(frames, scale=10)
   # assert 0

   perms = symframe_permutations_torch(frames, maxcols=len(frames2))

   # elems = wu.sym.symelems(sym, cyclic=True, screws=False)

   for i, unitelem in enumerate(elems + celems):
      alternate_elem_frames = elems[0].tolattice(lattice).operators
      # alternate_elem_frames = [np.eye(4), wu.hrot([0, 0, 1], 120)]
      for j, elemframe in enumerate(alternate_elem_frames):
         # if True:
         # if not elem.issues: continue

         # elem = elems[2]
         # ic(elem)
         # ic(elem.kind)
         # ic(elem.isoct)
         # ic(elem.cen)
         ic(i)
         # ic(unitelem)
         # ic(elemframe)
         elem = unitelem.tolattice(lattice).xformed(elemframe)
         ic(elem)
         # continue

         # wu.showme(elem, scale=scale, name='ref', symelemscale=5)
         # offset = wu.htrans([.02, .025, .03])
         # wu.showme(elem.operators @ elem.origin @ offset, scale=scale)
         # wu.showme(wu.hscaled(scale, elem.cen))
         # wu.sym.showsymelems(sym, [elem], scale=scale)
         # wu.showme(elem.operators @ offset, scale=scale)
         # wu.showme(frames2 @ offset, scale=scale)

         # compids = elem.frame_component_ids_bycenter(frames, sanitycheck=False)elem
         try:
            compids = elem.frame_component_ids(frames, perms, sanitycheck=True)
         except ComponentIDError:
            print('!' * 80, flush=True)
            print('elem has bad componentids, trying an alternate position')
            print('!' * 80, flush=True)
            continue

         opids = elem.frame_operator_ids(frames, sanitycheck=True)
         opcompids = _make_operator_component_joint_ids(elem, elem, frames, opids, compids, sanitycheck=True)

         print('SUCCESS', flush=True)
         break

      if i != 8:
         continue
      # tmp = wu.hxformpts(wu.hscaled(100, frames[opcompids == 109]), wu.hscaled(100, elem.cen + elem.axis))
      # ic(tmp)
      # assert 0
      import pymol
      # ids = opids
      # ids = compids
      ids = opcompids
      offset = wu.htrans([.002, .0025, .003])
      seenit = np.empty((0, 4))
      for i in range(np.max(ids)):
         assert np.sum(ids == i) > 0
         compframes = frames[ids == i]
         wu.showme(compframes @ elem.origin @ offset, scale=scale, name=f'id{i}')
         # cens = einsum('fij,j->fi', compframes, elem.origin[:, 3])
         # assert not np.any(np.all(np.isclose(cens[None], seenit[:, None]), axis=2))
         # seenit = np.concatenate([cens, seenit])

   assert 0

if __name__ == '__main__':
   main()
