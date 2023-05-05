REBUILD_SPACEGROUP_DATA = False
# REBUILD_SPACEGROUP_DATA = True

import numpy as np
import willutil as wu
from willutil.sym.spacegroup_data import *
from willutil.sym.spacegroup_util import *
from willutil.sym.spacegroup_symelems import compute_symelems

def _get_spacegroup_data():
   from willutil.storage import load_package_data, save_package_data, have_package_data

   if not REBUILD_SPACEGROUP_DATA and have_package_data('spacegroup_data'):
      sgdata = load_package_data('spacegroup_data')
   else:
      ABBERATION = ['B11m']

      from willutil.sym import spacegroup_frames
      sg_frames_dict = dict()
      sg_cheshire_dict = dict()
      sg_symelem = dict()
      sg_improper = dict()
      for i, (k, v) in enumerate(sg_tag.items()):
         if k in ABBERATION: continue

         if v in sg_lattice: sg_lattice[k] = sg_lattice[v]
         else: sg_lattice[v] = sg_lattice[k]

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

         if not sg_imporper:
            sg_symelem[k] = compute_symelems(k, frames)

      sgdata = (
         sg_frames_dict,
         sg_cheshire_dict,
         sg_symelem,
      )
      save_package_data(sgdata, 'spacegroup_data')

   return sgdata

(
   sg_frames_dict,
   sg_cheshire_dict,
   sg_symelem,
) = _get_spacegroup_data()
