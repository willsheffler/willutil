import itertools
import willutil as wu
from willutil.sym.spacegroup_data import _sg_frames, _sg_cheshire
from willutil.sym.spacegroup_data import *

def sgframes(spacegroup, cellgeom=None, cells=1):
   cells = process_num_cells(cells)
   if spacegroup not in sg_lattice:
      spacegroup = sg_from_pdbname[spacegroup]
   unitframes = _sg_frames[spacegroup]
   latticevec = lattice_vectors(spacegroup, cellgeom)
   xshift = wu.htrans(cells @ latticevec)
   frames = wu.hxform(xshift, unitframes, flat=True)
   return frames.round(10)

def process_num_cells(cells):
   if cells == None:
      return np.eye(4)[None]
   if isinstance(cells, (int, float)):
      ub = cells // 2
      lb = ub - cells + 1
      cells = [(a, b, c) for a, b, c in itertools.product(*[range(lb, ub + 1)] * 3)]

   elif len(cells) == 2:
      lb, ub = cells
      cells = [(a, b, c) for a, b, c in itertools.product(*[range(lb, ub + 1)] * 3)]
   elif len(cells) == 3:
      cells = [(a, b, c) for a, b, c in itertools.product(
         range(cells[0][0], cells[0][1] + 1),
         range(cells[1][0], cells[1][1] + 1),
         range(cells[2][0], cells[2][1] + 1),
      )]
   else:
      raise ValueError(f'bad cells {cells}')
   return np.array(cells)