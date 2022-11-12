import functools as ft
import numpy as np
import willutil as wu
from willutil.homog import *

class RBSampler:
   def __init__(
      self,
      cartsd=None,
      rotsd=None,
      lever=None,
      biasradial=None,
      biasdir=None,
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
      for i in range(100):
         perturb = hrand(1, self.cartsd * scale, self.rotsd * scale)
         p = hproj(self.biasdir, hcart3(perturb))
         pp = hprojperp(self.biasdir, hcart3(perturb))
         trans = p[:3] * self.biasradial + pp[:3]
         perturb[:3, 3] = trans
         assert hvalid(perturb)

         newpos = position.copy()
         cen = hxform(position, self.center)
         newpos[:3, 3] -= cen[:3]
         newpos = hxform(perturb, newpos)
         newpos[:3, 3] += cen[:3]
         assert hvalid(newpos)
         rad = wu.hnorm(hxform(newpos, self.center))
         if rad > self.minradius:
            break
      else:
         return prevpos, prevpos
      return newpos, prevpos
