REBUILD_SPACEGROUP_DATA = False
# REBUILD_SPACEGROUP_DATA = True

import numpy as np, itertools
import willutil as wu
from willutil.sym.spacegroup_data import *
from willutil.sym.spacegroup_util import *
from willutil.sym.spacegroup_symelems import _compute_symelems
from willutil.sym.permutations import symframe_permutations_torch
from willutil.storage import load_package_data, save_package_data, have_package_data

def _get_spacegroup_data():
   sgdata = dict()
   if have_package_data('spacegroup_data'):
      sgdata = load_package_data('spacegroup_data')
      if not REBUILD_SPACEGROUP_DATA:
         return sgdata
   ic('rebuilding spacegroup data')
   sg_frames_dict = sgdata.get('sg_frames_dict', dict())
   sg_cheshire_dict = sgdata.get('sg_cheshire_dict', dict())
   sg_symelem_dict = sgdata.get('sg_symelem_dict', dict())
   sg_permutations444_dict = sgdata.get('sg_permutations444_dict', dict())
   sg_symelem_frame444_opids_dict = sgdata.get('sg_symelem_frame444_opids_dict', dict())
   sg_symelem_frame444_compids_dict = sgdata.get('sg_symelem_frame444_compids_dict', dict())
   sg_symelem_frame444_opcompids_dict = sgdata.get('sg_symelem_frame444_opcompids_dict', dict())

   # sg_symelem_dict = dict()

   from willutil.sym import spacegroup_frames
   sg_improper = dict()

   for i, (k, v) in enumerate(sg_tag.items()):

      if v in sg_lattice: sg_lattice[k] = sg_lattice[v]
      else: sg_lattice[v] = sg_lattice[k]

      if k not in sg_frames_dict:
         tmp, sg_cheshire_dict[k] = getattr(spacegroup_frames, f'symframes_{v}')()
         frames = np.zeros((len(tmp), 4, 4))
         frames[:, 3, 3] = 1
         frames[:, 0, :3] = tmp[:, 0:3]
         frames[:, 1, :3] = tmp[:, 3:6]
         frames[:, 2, :3] = tmp[:, 6:9]
         frames[:, :3, 3] = tmp[:, 9:]
         frames[frames[:, 0, 3] > 0.999, 0, 3] -= 1
         frames[frames[:, 1, 3] > 0.999, 1, 3] -= 1
         frames[frames[:, 2, 3] > 0.999, 2, 3] -= 1
         assert np.sum(frames == 12345) == 0
         sg_frames_dict[k] = frames
         sg_imporper = not np.allclose(1, np.linalg.det(frames))
      frames = sg_frames_dict[k]
      # if not sg_imporper:

      if sg_lattice[k] == 'CUBIC' and sg_is_chiral(k):

         if k not in sg_symelem_dict:
            print(k, 'detect symelems', flush=True)
            sg_symelem_dict[k] = _compute_symelems(k, frames)
            sg_symelem_dict[k] = list(itertools.chain(*sg_symelem_dict[k].values()))  # flatten
            ic(sg_symelem_dict[k])

         frames4 = latticeframes(frames, np.eye(3), 4)
         # len(frames)*8 keeps only enough perm frames for 2x2x2 cell
         if k not in sg_permutations444_dict:
            print(k, 'compute permutations', flush=True)
            sg_permutations444_dict[k] = symframe_permutations_torch(frames4, maxcols=len(frames) * 8)
         perms = sg_permutations444_dict[k]
         nops = len(sg_symelem_dict[k])

         if k not in sg_symelem_frame444_opcompids_dict:
            print('rebuild symelem frameids', k, flush=True)
            sg_symelem_frame444_opids_dict[k] = -np.ones((len(frames4), nops), dtype=np.int32)
            sg_symelem_frame444_compids_dict[k] = -np.ones((len(frames4), nops), dtype=np.int32)
            sg_symelem_frame444_opcompids_dict[k] = -np.ones((len(frames4), nops, nops), dtype=np.int32)
            for ielem, se in enumerate(sg_symelem_dict[k]):
               ic(k, ielem)
               sg_symelem_frame444_opids_dict[k][:, ielem] = se.frame_operator_ids(frames4)
               sg_symelem_frame444_compids_dict[k][:, ielem] = se.frame_component_ids(frames4, perms)
               for jelem, se2 in enumerate(sg_symelem_dict[k]):
                  fopid = sg_symelem_frame444_opids_dict[k][:, ielem]
                  fcompid = sg_symelem_frame444_compids_dict[k][:, jelem]
                  ids = fcompid.copy()
                  for i in range(np.max(fopid)):
                     fcids = fcompid[fopid == i]
                     idx0 = fcompid == fcids[0]
                     for fcid in fcids[1:]:
                        idx = fcompid == fcid
                        ids[idx] = min(min(ids[idx]), min(ids[idx0]))
                  for i, id in enumerate(sorted(set(ids))):
                     ids[ids == id] = i
                  sg_symelem_frame444_opcompids_dict[k][:, ielem, jelem] = ids

   sgdata = dict(
      sg_frames_dict=sg_frames_dict,
      sg_cheshire_dict=sg_cheshire_dict,
      sg_symelem_dict=sg_symelem_dict,
      sg_permutations444_dict=sg_permutations444_dict,
      sg_symelem_frame444_opids_dict=sg_symelem_frame444_opids_dict,
      sg_symelem_frame444_compids_dict=sg_symelem_frame444_compids_dict,
      sg_symelem_frame444_opcompids_dict=sg_symelem_frame444_opcompids_dict,
   )
   ic('saving spacegroup data')
   save_package_data(sgdata, 'spacegroup_data')

   return sgdata

_sgdata = _get_spacegroup_data()
sg_frames_dict = _sgdata['sg_frames_dict']
sg_cheshire_dict = _sgdata['sg_cheshire_dict']
sg_symelem_dict = _sgdata['sg_symelem_dict']
sg_permutations444_dict = _sgdata['sg_permutations444_dict']
sg_symelem_frame444_opids_dict = _sgdata['sg_symelem_frame444_opids_dict']
sg_symelem_frame444_compids_dict = _sgdata['sg_symelem_frame444_compids_dict']
sg_symelem_frame444_opcompids_dict = _sgdata['sg_symelem_frame444_opcompids_dict']

#def makedata2():
#   sg_symelem_frame444_opids_dict = dict()
#   sg_symelem_frame444_compids_sym_dict = dict()
#   sg_symelem_frame444_opcompids_dict = dict()
#   for k, v in sg_symelem_dict.items():
#      if sg_lattice[k] == 'CUBIC' and sg_is_chiral(k):
#
#         frames4 = latticeframes(sg_frames_dict[k], np.eye(3), 4)
#         nops = len(sg_symelem_dict[k])
#         perms = sg_permutations444_dict[k]
#         sg_symelem_frame444_opids_dict[k] = -np.ones((len(frames4), nops), dtype=np.int32)
#         sg_symelem_frame444_compids_dict[k] = -np.ones((len(frames4), nops), dtype=np.int32)
#         sg_symelem_frame444_opcompids_dict[k] = -np.ones((len(frames4), nops, nops), dtype=np.int32)
#         for ielem, se in enumerate(sg_symelem_dict[k]):
#            ic(k, ielem)
#            sg_symelem_frame444_opids_dict[k][:, ielem] = se.frame_operator_ids(frames4)
#            sg_symelem_frame444_compids_dict[k][:, ielem] = se.frame_component_ids(frames4, perms)
#            for jelem, se2 in enumerate(sg_symelem_dict[k]):
#               fopid = sg_symelem_frame444_opids_dict[k][:, ielem]
#               fcompid = sg_symelem_frame444_compids_dict[k][:, jelem]
#
#               ids = fcompid.copy()
#               for i in range(np.max(fopid)):
#                  fcids = fcompid[fopid == i]
#                  idx0 = fcompid == fcids[0]
#                  for fcid in fcids[1:]:
#                     idx = fcompid == fcid
#                     ids[idx] = min(min(ids[idx]), min(ids[idx0]))
#               for i, id in enumerate(sorted(set(ids))):
#                  ids[ids == id] = i
#
#               sg_symelem_frame444_opcompids_dict[k][:, ielem, jelem] = ids
#   sgdata = (
#      sg_frames_dict,
#      sg_cheshire_dict,
#      sg_symelem_dict,
#      sg_permutations444_dict,
#      sg_symelem_frame444_opids_dict,
#      sg_symelem_frame444_compids_dict,
#      sg_symelem_frame444_opcompids_dict,
#   )
#   save_package_data(sgdata, 'spacegroup_data')
#   assert 0
# makedata2()
