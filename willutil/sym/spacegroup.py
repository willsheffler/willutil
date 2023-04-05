import willutil as wu
from willutil.sym.spacegroup_data import _sg_frames, _sg_cheshire
from willutil.sym.spacegroup_data import *

def sgframes(spacegroup, cellgeom=None):
   if spacegroup not in sg_lattice:
      spacegroup = sg_from_pdbname[spacegroup]
   cellgeom = check_cellgeom(spacegroup, cellgeom)

   frames = _sg_frames[spacegroup]
   return frames