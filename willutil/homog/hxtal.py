import numpy as np
import willutil as wu

class Xtal:
   def __init__(self, symelems, name='xtal'):
      self.name = name
      self.symelems = symelems
      self.genframes = self.generate_frames()  # expensive
      self.unitframes = self.generate_unit_frames(self.genframes)
      self.coverelems = self.generate_cover_symelems(self.genframes)
      self.unitelems = self.generate_unit_symelems(self.genframes)

   @property
   def nunit(self):
      return len(self.unitframes)

   def generate_frames(self, depth=50, radius=9e9, trials=100000):
      generators = np.concatenate([s.operators for s in self.symelems])
      x, _ = wu.cpp.geom.expand_xforms_rand(generators, depth=depth, radius=radius, trials=trials)
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
         assert len(elems) == len(frames)
         coverelems.append(list(zip(elems, frames)))
      return coverelems

class SymElem:
   def __init__(self, nfold, axis, cen=[0, 0, 0]):
      self.nfold = nfold
      self.coords = wu.hray(cen, axis)
      x = wu.hrot(self.coords, nfold=nfold)
      self.operators = np.stack([wu.hpow(x, p) for p in range(nfold)])

   @property
   def cen(self):
      return self.coords[..., 0]

   @property
   def axis(self):
      return self.coords[..., 1]

   @property
   def angle(self):
      return np.pi * 2 / self.nfold

def genxtal(symelems):
   pass
