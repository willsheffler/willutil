import itertools, os
import numpy as np
import willutil as wu

cryst1_pattern = "CRYST1  %7.3f  %7.3f  %7.3f  90.00  90.00  90.00 %s\n"

class Xtal:
   def __init__(self, name='xtal', symelems=None, **kw):
      self.info = None
      self.symelems = symelems
      if symelems is None:
         self.name, self.info = wu.sym.xtalinfo(name)
         if 'dimension' not in self.info:
            self.info.dimension = 3
         self.symelems = self.info.symelems
         self.dimension = self.info.dimension
         self.sub = self.info.nsub
      self.compute_frames(**kw)
      # self.symelems_to_unitcell()

   def symelems_to_unitcell(self):
      for s in self.symelems:
         if s.cen[0] < 0: s.cen[0] += 1
         if s.cen[1] < 0: s.cen[1] += 1
         if s.cen[2] < 0: s.cen[2] += 1
         if s.cen[0] > 1: s.cen[0] -= 1
         if s.cen[1] > 1: s.cen[1] -= 1
         if s.cen[2] > 1: s.cen[2] -= 1

   def compute_frames(self, **kw):
      if self.dimension == 3:
         self.genframes = self.generate_candidate_frames(bound=2, **kw)  # expensive-ish
         self.coverelems = self.generate_cover_symelems(self.genframes)
         self.unitframes = self.generate_unit_frames(self.genframes)
         self.unitelems = self.generate_unit_symelems(self.genframes)
      else:
         self.genframes = self.generate_candidate_frames(bound=3, **kw)  # expensive-ish
         self.coverelems = self.generate_cover_symelems(self.genframes, bbox=None)
         self.unitframes = self.genframes
         self.unitelems = self.coverelems
      self.nsub = len(self.unitframes)
      if self.info is not None and self.info.nsub is not None:
         assert self.nsub == self.info.nsub, f'nsub for "{self.name}" should be {self.info.nsub}, not {self.nsub}'

   def symcoords(self, asymcoords, cellsize, cells, flat=True):
      asymcoords = wu.hpoint(asymcoords)
      frames = self.cellframes(cellsize, cells)
      coords = wu.hxform(frames, asymcoords)
      if flat: coords = coords.reshape(-1, 4)
      return coords

   def dump_pdb(self, fname, asymcoords, cellsize, cells=None):
      cryst1 = cryst1_pattern % (*(cellsize, ) * 3, self.info.spacegroup)
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

   def generate_candidate_frames(self, depth=100, bound=2.5, genradius=9e9, trials=10000, **kw):
      cachefile = wu.datapath(f'xtal/lotsframes_{self.name.replace(" ","_")}.npy')
      if self.dimension == 2 or not os.path.exists(cachefile):
         generators = np.concatenate([s.operators for s in self.symelems])
         x, _ = wu.cpp.geom.expand_xforms_rand(generators, depth=depth, radius=genradius, trials=trials)
         testpoint = [0.001, 0.002, 0.003]
         cens = wu.hxform(x, testpoint)
         inboundslow = np.all(cens >= -bound - 0.001, axis=-1)
         inboundshigh = np.all(cens <= bound + 0.001, axis=-1)
         inbounds = np.logical_and(inboundslow, inboundshigh)
         x = x[inbounds]
         assert len(x) < 10000
         if self.dimension == 3:
            np.save(cachefile, x)
      else:
         x = np.load(cachefile)
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
         elems = symelem.xformed(self.unitframes)
         unitelems.append(list(zip(elems, self.unitframes)))
      return unitelems

   def generate_cover_symelems(self, candidates, bbox=[(0, 0, 0), (1, 1, 1)]):
      coverelems = list()
      bbox = np.asarray(bbox) if bbox is not None else None
      for i, symelem in enumerate(self.symelems):
         testpoint = symelem.cen[:3]
         cens = wu.hxform(candidates, testpoint)
         frames = candidates
         if bbox is not None:
            inboundslow = np.all(cens >= bbox[0] - 0.0001, axis=-1)
            inboundshigh = np.all(cens <= bbox[1] + 0.0001, axis=-1)
            inbounds = np.logical_and(inboundslow, inboundshigh)
            frames = candidates[inbounds]
         elems = symelem.xformed(frames)
         # ic(len(elems))
         assert len(elems) == len(frames), f'{len(elems)} {len(frames)}'
         coverelems.append(list(zip(elems, frames)))
      return coverelems
