import functools as ft
import numpy as np
import willutil as wu
from willutil.homog import *

class RBLatticeRBSampler:
   def __init__(self, cartsd, scalesd=0, *args, **kw):
      self.cartsd = cartsd
      self.scalesd = scalesd
      self.rbsampler = RBSampler(*args, cartsd=cartsd, **kw)

   def __call__(self, state, *args, **kw):
      assert isinstance(state, wu.Bunch)
      rb, _ = self.rbsampler(state.position, *args, **kw)
      scale = state.scale + np.random.randn() * self.scalesd
      return wu.Bunch(position=rb, scale=scale), state

class RBSampler:
   def __init__(
      self,
      cartsd=None,
      rotsd=None,
      lever=None,
      biasradial=1,
      biasdir=[1, 0, 0],
      center=[0, 0, 0],
      minradius=0,
      **kw,
   ):
      self.cartsd = cartsd
      self.rotsd = rotsd
      if rotsd == None:
         self.rotsd = cartsd / (lever or 20)
      else:
         assert lever == None, f'if rotsd specified, no lever must be provided'
      self.biasdir = hnormalized(biasdir)
      self.biasradial = float(biasradial)
      self.minradius = minradius
      self.center = hpoint(center)

   def __call__(self, position, scale=1):
      prevpos = position.copy()
      newpos = prevpos
      for i in range(100):
         perturb = hrand(1, self.cartsd * scale, self.rotsd * scale)
         if self.biasradial != 1:
            p = hproj(self.biasdir, hcart3(perturb))
            pp = hprojperp(self.biasdir, hcart3(perturb))
            trans = p[:3] * self.biasradial + pp[:3]
            perturb[:3, 3] = trans
            assert hvalid(perturb)
         newpos0 = position.copy()
         cen = hxform(position, self.center)
         newpos0[:3, 3] -= cen[:3]
         newpos0 = hxform(perturb, newpos0)
         newpos0[:3, 3] += cen[:3]
         assert hvalid(newpos0)
         rad = wu.hnorm(hxform(newpos0, self.center))
         if rad > self.minradius:
            newpos = newpos0
            break
      else:
         raise ValueError(f'bad configuration, cant escape minradius {self.minradius}')
      # if self.scalesd is not None:
      # return Bunch()
      return newpos, prevpos
