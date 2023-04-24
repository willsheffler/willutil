import itertools
import willutil as wu

from willutil.sym.spacegroup_data import *
from willutil.sym.spacegroup_util import *
from willutil.sym.spacegroup_deriveddata import *

_memoized_frames = dict()

def sgframes(
   spacegroup: str,
   cellgeom=None,
   cells=1,
   sortframes='nosort',
   roundgeom=10,
   xtalrad=9e9,
   asucen=[0.5, 0.5, 0.5],
   xtalcen=None,
   **kw,
):
   spacegroup = spacegroup.upper()
   if cellgeom not in ('unit', None):
      cellgeom = tuple(round(x, roundgeom) for x in cellgeom)
   cells = process_num_cells(cells)
   key = spacegroup, cellgeom, tuple(cells.flat), sortframes
   if not key in _memoized_frames:

      if spacegroup not in sg_lattice:
         spacegroup = sg_from_pdbname[spacegroup]
      unitframes = sg_frames_dict[spacegroup]
      if cellgeom == 'unit': latticevec = np.eye(3)
      else: latticevec = lattice_vectors(spacegroup, cellgeom)
      frames = latticeframes(unitframes, latticevec, cells)

      frames = prune_frames(frames, asucen, xtalrad, xtalcen)
      frames = sort_frames(frames, method=sortframes)

      _memoized_frames[key] = frames.round(10)
      if len(_memoized_frames) > 10_000:
         wu.WARNME(f'sgframes holding >10000 _memoized_frames')

   return _memoized_frames[key]

def prune_frames(frames, asucen, xtalrad, center=None):
   center = center or asucen
   center = wu.hpoint(center)
   asucen = wu.hpoint(asucen)
   pos = wu.hxform(frames, asucen)
   dis = wu.hnorm(pos - center)
   frames = frames[dis <= xtalrad]
   return frames

def sort_frames(frames, method):
   if method == 'nosort':
      return frames
   if method == 'dist_to_asucen':
      assert 0
