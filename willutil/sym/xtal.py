import itertools
import numpy as np
import willutil as wu

cryst1_pattern = "CRYST1  %7.3f  %7.3f  %7.3f  90.00  90.00  90.00 %s\n"

class Xtal:
   def __init__(self, name='xtal', symelems=None, **kw):
      self.name = name
      if symelems is None:
         self.name, symelems, nsub = wu.sym.xtalinfo(name)
      self.symelems = symelems
      self.genframes = self.generate_candidate_frames(**kw)  # expensive-ish
      self.unitframes = self.generate_unit_frames(self.genframes)
      self.nsub = len(self.unitframes)
      assert self.nsub == nsub, f'nsub for "{self.name}" should be {nsub}, not {self.nsub}'
      self.coverelems = self.generate_cover_symelems(self.genframes)
      self.unitelems = self.generate_unit_symelems(self.genframes)

   def symcoords(self, asymcoords, cellsize, cells, flat=True):
      asymcoords = wu.hpoint(asymcoords)
      frames = self.cellframes(cellsize, cells)
      coords = wu.hxform(frames, asymcoords)
      if flat: coords = coords.reshape(-1, 4)
      return coords

   def dump_pdb(self, fname, asymcoords, cellsize, cells=None):
      cryst1 = cryst1_pattern % (*(cellsize, ) * 3, self.name.replace('_', ' '))
      asymcoords = np.asarray(asymcoords)

      if cells == None:
         wu.pdb.dump_pdb_from_points(fname, asymcoords, header=cryst1)
      else:
         coords = self.symcoords(asymcoords, cellsize, cells)
         wu.pdb.dump_pdb_from_points(fname, coords)

   @property
   def nunit(self):
      return len(self.unitframes)

   def cellframes(self, cellsize=1, cells=1, flat=True):
      if cells == None:
         return np.eye(4)[None]
      if isinstance(cells, int):
         ub = cells // 2
         lb = ub - cells + 1
         cells = [(a, b, c) for a, b, c in itertools.product(*[range(lb, ub + 1)] * 3)]
      else:
         lb, ub = cells
         cells = [(a, b, c) for a, b, c in itertools.product(*[range(lb, ub + 1)] * 3)]
      cells = wu.hpoint(cells)
      cells = cells.reshape(-1, 4)
      xcellshift = wu.htrans(cells)
      frames = self.unitframes.copy()
      frames = wu.hxform(xcellshift, frames)
      frames[..., :3, 3] *= cellsize
      if flat: frames = frames.reshape(-1, 4, 4)
      return frames

   def generate_candidate_frames(self, depth=30, radius=9e9, trials=1000, **kw):
      generators = np.concatenate([s.operators for s in self.symelems])
      x, _ = wu.cpp.geom.expand_xforms_rand(generators, depth=depth, radius=radius, trials=trials)
      testpoint = [0.001, 0.002, 0.003]
      cens = wu.hxform(x, testpoint)
      inboundslow = np.all(cens >= -2.5001, axis=-1)
      inboundshigh = np.all(cens <= 2.5001, axis=-1)
      inbounds = np.logical_and(inboundslow, inboundshigh)
      x = x[inbounds]
      assert len(x) < 10000
      return x

   def generate_unit_frames(self, candidates, bbox=[(0, 0, 0), (1, 1, 1)], testpoint=None):
      bbox = np.asarray(bbox)
      testpoint = testpoint if testpoint else [0.001, 0.002, 0.003]
      cens = wu.hxform(candidates, testpoint)
      inboundslow = np.all(cens >= bbox[0] - 0.0001, axis=-1)
      inboundshigh = np.all(cens <= bbox[1] + 0.0001, axis=-1)
      inbounds = np.logical_and(inboundslow, inboundshigh)
      frames = candidates[inbounds]
      return frames

   def generate_unit_symelems(self, candidates, bbox=[(0, 0, 0), (1, 1, 1)]):
      unitelems = list()
      for i, symelem in enumerate(self.symelems):
         elems = wu.hxform(self.unitframes, symelem)
         unitelems.append(list(zip(elems, self.unitframes)))
      return unitelems

   def generate_cover_symelems(self, candidates, bbox=[(0, 0, 0), (1, 1, 1)]):
      coverelems = list()
      bbox = np.asarray(bbox)
      for i, symelem in enumerate(self.symelems):
         testpoint = symelem.cen[:3]
         cens = wu.hxform(candidates, testpoint)
         inboundslow = np.all(cens >= bbox[0] - 0.0001, axis=-1)
         inboundshigh = np.all(cens <= bbox[1] + 0.0001, axis=-1)
         inbounds = np.logical_and(inboundslow, inboundshigh)
         frames = candidates[inbounds]
         elems = wu.hxform(frames, symelem)
         # ic(elems)
         assert len(elems) == len(frames), f'{len(elems)} {len(frames)}'
         coverelems.append(list(zip(elems, frames)))
      return coverelems
