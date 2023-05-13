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
   sortframes='default',
   roundgeom=10,
   xtalrad=9e9,
   asucen=[0.5, 0.5, 0.5],
   xtalcen=None,
   **kw,
):
   spacegroup = spacegroup_canonical_name(spacegroup)
   if cellgeom not in ('unit', None):
      cellgeom = tuple(round(x, roundgeom) for x in cellgeom)
   cells = process_num_cells(cells)
   key = spacegroup, cellgeom, tuple(cells.flat), sortframes
   if not key in _memoized_frames:
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

def sgpermutations(spacegroup: str, cells=4):
   assert cells == 4
   spacegroup = spacegroup_canonical_name(spacegroup)
   return sg_permutations444_dict[spacegroup]

def symelems(spacegroup: str, psym=None, asdict=False, screws=True, cyclic=True):
   if isinstance(psym, int):
      psym = f'c{psym}'
   spacegroup = spacegroup_canonical_name(spacegroup)
   se = sg_symelem_dict[spacegroup]
   if not screws:
      se = [e for e in se if e.screw == 0]
   if not cyclic:
      se = [e for e in se if not e.iscyclic]
   if psym:
      return [e for e in se if e.label == psym.upper()]

   if asdict:
      d = defaultdict(list)
      for e in se:
         d[e.label].append(e)
      se = d
   return se

def copies_per_cell(spacegroup):
   spacegroup = spacegroup_canonical_name(spacegroup)
   return len(sg_frames_dict[spacegroup])

def spacegroup_canonical_name(spacegroup):
   spacegroup = spacegroup.replace('p', 'P').replace('i', 'I').replace('f', 'F')
   if spacegroup not in sg_lattice:
      spacegroup = sg_from_pdbname[spacegroup]
   return spacegroup

def latticetype(spacegroup):
   try:
      return sg_lattice[spacegroup]
   except KeyError:
      return sg_lattice[sg_from_pdbname[spacegroup]]

def cryst1_line(spacegroup, lattice):
   cellgeom = cellgeom_from_lattice(lattice)
   return wu.sym.cryst1_pattern_full % (*cellgeom, spacegroup)

def prune_frames(frames, asucen, xtalrad, center=None):
   center = center or asucen
   center = wu.hpoint(center)
   asucen = wu.hpoint(asucen)
   pos = wu.hxform(frames, asucen)
   dis = wu.hnorm(pos - center)
   frames = frames[dis <= xtalrad]
   return frames

def cellgeom_from_lattice(lattice, radians=False):
   a = wu.hnorm(lattice[0])
   b = wu.hnorm(lattice[1])
   c = wu.hnorm(lattice[2])
   if radians:
      A = wu.hangle(lattice[1], lattice[2])
      B = wu.hangle(lattice[0], lattice[2])
      C = wu.hangle(lattice[0], lattice[1])
   else:
      A = wu.hangle_degrees(lattice[1], lattice[2])
      B = wu.hangle_degrees(lattice[0], lattice[2])
      C = wu.hangle_degrees(lattice[0], lattice[1])
   return [a, b, c, A, B, C]

def sort_frames(frames, method):
   if method == 'default':
      return frames
   if method == 'dist_to_asucen':
      assert 0
