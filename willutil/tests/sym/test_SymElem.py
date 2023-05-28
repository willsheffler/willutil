import pytest
import numpy as np
import willutil as wu
from willutil.sym.SymElem import *
from willutil.sym.SymElem import _make_operator_component_joint_ids
from willutil.sym.spacegroup_symelems import _compute_symelems, _find_compound_symelems
from willutil.sym.permutations import symframe_permutations_torch

def main():
   test_screw_elem()
   test_screw_elem_frames()
   # mcdock_bug1()
   # assert 0
   check_frame_opids()

def test_screw_elem_frames():
   f320 = np.array([[1., 0., -0., 0.], [0., 1., -0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
   f321 = np.array([[-0.5, 0.8660254, 0., 0.5], [-0.8660254, -0.5, 0., 0.8660254], [0., 0., 1., 0.56666667], [0., 0., 0., 1.]])
   f322 = np.array([[-0.5, 0.8660254, 0., 1.], [-0.8660254, -0.5, 0., 0.], [0., 0., 1., 0.56666667], [0., 0., 0., 1.]])

   f310 = np.array([[-0.5, -0.8660254, 0., 0.], [0.8660254, -0.5, 0., 0.], [0., 0., 1., 1.13333333], [0., 0., 0., 1.]])
   f311 = np.array([[-0.5, -0.8660254, 0., 1.], [0.8660254, -0.5, 0., 0.], [0., 0., 1., 1.13333333], [0., 0., 0., 1.]])
   f312 = np.array([[-0.5, -0.8660254, 0., 0.5], [0.8660254, -0.5, 0., -0.8660254], [0., 0., 1., 1.13333333], [0., 0., 0., 1.]])

   a, an, c, h = wu.haxis_angle_cen_hel_of(f320)
   ic(a, an, c, h)

   assert 0

def test_screw_elem():
   ic('test_screw_elem')
   S2 = np.sqrt(2)
   S3 = np.sqrt(3)

   assert SymElem(1, [0, 0, 1], hel=1).label == 'C11'
   assert SymElem(2, [0, 0, 1], hel=0.5).label == 'C21'
   assert SymElem(2, [0, 1, 1], hel=S2 / 2).label == 'C21'
   assert SymElem(2, [1, 1, 1], hel=S3 / 2).label == 'C21'
   with pytest.raises(ScrewError):
      assert SymElem(2, [1, 1, 1], hel=1)
   with pytest.raises(ScrewError):
      SymElem(2, [0, 0, 1], hel=1)
   with pytest.raises(ScrewError):
      SymElem(1, [0, 0, 1], hel=0.5)
   with pytest.raises(ScrewError):
      SymElem(1, [1, 2, 3], hel=0.5)

   assert SymElem(3, [0, 0, 1], hel=1 / 3).label == 'C31'
   assert SymElem(3, [0, 0, 1], hel=2 / 3).label == 'C32'
   assert SymElem(3, [1, 1, 1], hel=S3 * 1 / 3).label == 'C31'
   assert SymElem(3, [1, 1, 1], hel=S3 * 2 / 3).label == 'C32'
   with pytest.raises(ScrewError):
      SymElem(3, [0, 0, 1], hel=1)

   assert SymElem(4, [0, 0, 1], hel=0.25).label == 'C41'
   assert SymElem(4, [0, 0, 1], hel=0.50).label == 'C42'
   assert SymElem(4, [0, 0, 1], hel=0.75).label == 'C43'
   with pytest.raises(ScrewError):
      SymElem(4, [0, 0, 1], hel=-0.25)
   with pytest.raises(ScrewError):
      SymElem(4, [0, 0, 1], hel=1)

   assert SymElem(6, [0, 0, 1], hel=1 / 6).label == 'C61'
   assert SymElem(6, [0, 1, 0], hel=2 / 6).label == 'C62'
   assert SymElem(6, [1, 0, 0], hel=3 / 6).label == 'C63'
   assert SymElem(6, [0, 1, 0], hel=4 / 6).label == 'C64'
   assert SymElem(6, [0, 0, 1], hel=5 / 6).label == 'C65'

   for i in range(100):
      x = wu.hrand()
      x31 = wu.hinv(x) @ wu.hrot([0, 0, 1], 120) @ wu.htrans([0, 0, 1]) @ x
      x32 = wu.hinv(x) @ wu.hrot([0, 0, 1], 240) @ wu.htrans([0, 0, 1]) @ x
      assert np.allclose(1, wu.homog.axis_angle_cen_hel_of(x31)[3])
      assert np.allclose(-1, wu.homog.axis_angle_cen_hel_of(x32)[3])

   x31 = wu.hrot([0, 0, 1], 120) @ wu.htrans([0, 0, 1 / 3])
   x32 = wu.hrot([0, 0, 1], 240) @ wu.htrans([0, 0, 1 / 3])

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
