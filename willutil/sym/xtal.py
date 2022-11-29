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
         self.spacegroup = self.info.spacegroup
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
         self.genframes = self.generate_candidate_frames(bound=1.5, **kw)  # expensive-ish
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

   def asucen(self, cellsize=1, use_olig_nbrs=False, olig_nbr_wt=0.75, **kw):
      elems = self.symelems
      # ic([e.cen for e in elems])
      cen0 = np.mean(np.stack([e.cen for e in elems]), axis=0)
      if use_olig_nbrs:
         opcens = [np.mean(wu.hxform(e.operators[1:], cen0), axis=0) for e in self.symelems]
         cen = np.mean(np.stack(opcens), axis=0)
         # this is arbitrary
         cen = olig_nbr_wt * cen + (1 - olig_nbr_wt) * cen0
      else:
         cen = cen0
      return wu.hscaled(cellsize, cen)

   def central_symelems(self, cells=3, target=None):
      assert 0, 'WARNING central_symelems is buggy, axis direction wrong!?!'
      # assert 0
      # _targets = {'I 41 3 2': [0.3, 0.4, 0.5]}
      # if target is None:
      # if self.spacegroup in _targets:
      # target = _targets[self.spacegroup]
      # else:
      # target = [0.3, 0.4, 0.5]
      target = wu.hpoint(target)
      cenelems = list()
      cells = interp_xtal_cell_list(cells)
      for i, elems in enumerate(self.coverelems):
         best = 9e9, None
         for cellshift in cells:
            xcellshift = wu.htrans(cellshift)
            for j, elem in enumerate(elems):
               elem = elem.xformed(xcellshift)
               # ic(i, j, elem.cen)
               d = wu.hnorm(elem.cen - target)
               if d < best[0]:
                  best = d, elem
         cenelems.append(best[1])
      return cenelems

   def frames(self, cells=3, **kw):
      return self.cellframes(cells=cells, **kw)

   def cellframes(self, cellsize=1, cells=1, flat=True, center=None, asucen=None, radius=None):
      if self.dimension != 3:
         return self.unitframes
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

      cells = wu.hpoint(cells)
      cells = cells.reshape(-1, 4)
      xcellshift = wu.htrans(cells)
      frames = self.unitframes.copy()
      frames = wu.hxform(xcellshift, frames)
      frames[..., :3, 3] *= cellsize
      if flat: frames = frames.reshape(-1, 4, 4)
      # ic(frames.shape)
      # ic(frames[:10, :3, 3])
      if center is not None:
         if asucen is None: asucen = center
         if radius is None: radius = 0.5 * cellsize
         center = wu.hpoint(center)
         asucen = wu.hpoint(asucen)
         # ic(asucen)
         pos = wu.hxform(frames, asucen)
         # ic(pos.shape)
         # ic(pos[:10, 3])
         # wu.showme(center, sphere=30)
         # wu.showme(pos, name='pos', sphere=10)
         # assert 0
         dis = wu.hnorm(pos - center)
         # ic(center)
         # ic(dis)
         ic(frames.shape)
         frames = frames[dis <= radius]
         # ic(radius)
         # ic(center)
         # ic(asucen)
         # ic(frames.shape)
         # assert 0
      return frames

   def generate_candidate_frames(
      self,
      depth=100,
      bound=2.5,
      genradius=9e9,
      trials=10000,
      cache=True,
      **kw,
   ):
      cachefile = wu.datapath(f'xtal/lots_of_frames_{self.name.replace(" ","_")}.npy')
      if self.dimension == 2 or not os.path.exists(cachefile) or not cache:
         generators = np.concatenate([s.operators for s in self.symelems])
         x, _ = wu.cpp.geom.expand_xforms_rand(generators, depth=depth, radius=genradius, trials=trials)
         testpoint = [0.001, 0.002, 0.003]
         cens = wu.hxform(x, testpoint)
         inboundslow = np.all(cens >= -bound - 0.001, axis=-1)
         inboundshigh = np.all(cens <= bound + 0.001, axis=-1)
         inbounds = np.logical_and(inboundslow, inboundshigh)
         x = x[inbounds]
         assert len(x) < 10000
         if self.dimension == 3 and cache:
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
         for e, f in zip(elems, self.unitframes):
            e.origin = f
         unitelems.append(elems)
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
         for e, f in zip(elems, frames):
            assert np.allclose(np.eye(4), e.origin)
            e.origin = f
         coverelems.append(elems)
      return coverelems

def interp_xtal_cell_list(cells):
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
   return cells
