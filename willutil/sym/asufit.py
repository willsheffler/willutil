import functools as ft
import numpy as np
import willutil as wu
from willutil.homog import *

def asufit(
   sym,
   coords,
   frames=None,
   objfunc=None,
   sampler=None,
   mc=None,
   showme=False,
   cartsd=None,
   iterations=300,
   lever=None,
   temperature=1,
   thresh=0.000001,
   minradius=None,
   resetinterval=100,
   correctionfactor=2,
   showme_accepts=False,
   **kw,
):
   ic('asufit', sym)

   kw = wu.Bunch(kw)
   asym = wu.RigidBody(coords, **kw)
   bodies = [asym] + [wu.RigidBody(parent=asym, xfromparent=x, **kw) for x in frames[1:]]
   if kw.biasradial is None: kw.biasradial = wu.sym.symaxis_radbias(sym, 2, 3)
   kw.biasdir = wu.hnormalized(asym.com())
   if frames is None: frames = wu.sym.frames(sym)
   if lever is None: kw.lever = asym.rog() * 1.5
   if minradius is None: kw.minradius = wu.hnorm(asym.com()) * 0.5

   ObjFuncDefault = wu.rigid.LatticeOverlapObjective
   SamplerDefault = wu.search.LatticeRBSampler

   if objfunc is None:
      objfunc = ObjFuncDefault(
         asym,
         scoreframes=[(0, 1), (0, 2)],
         clashframes=[(1, 2), (1, 3), (2, 3)],
         bodies=bodies,
         sym=sym,
         **kw,
      )

   if sampler is None:
      sampler = SamplerDefault(
         cartsd=cartsd,
         center=asym.com(),
         **kw,
      )
   if mc is None:
      mc = wu.MonteCarlo(objfunc, temperature=temperature, **kw)

   start = asym.state
   mc.try_this(start)
   initialscore = mc.best

   if showme: wu.showme(bodies, name='start', **kw)

   wu.pdb.dump_pdb_from_points('start.pdb', asym.coords)

   for i in range(iterations):
      if i % 50 == 0 and i > 0:
         asym.state = mc.beststate
         if i % 200:
            asym.state = mc.startstate
         if mc.acceptfrac < 0.1:
            mc.temperature *= correctionfactor / resetinterval * 100
            sampler.cartsd /= correctionfactor / resetinterval * 100
         if mc.acceptfrac > 0.3:
            mc.temperature /= correctionfactor / resetinterval * 100
            sampler.cartsd *= correctionfactor / resetinterval * 100
         # ic(mc.acceptfrac, mc.best)

      pos, prev = sampler(asym.state)
      accept = mc.try_this(pos)
      if not accept:
         asym.state = prev
      else:
         if mc.best < thresh:
            # ic('end', i, objfunc(mc.beststate))
            # if showme: wu.showme(bodies, name='mid%i' % i, **kw)
            return mc

         # ic(i, mc.last)
         # if i % 10 == 0:
         if showme and showme_accepts: wu.showme(bodies, name='mid%i' % i, **kw)

   assert mc.beststate is not None
   # ic('end', mc.best)
   initscore = objfunc(mc.startstate, verbose=True)
   stopscore = objfunc(mc.beststate, verbose=True)
   ic('init', initscore)
   ic('stop', stopscore)
   # ic(mc.beststate[:3, :3])
   # ic(mc.beststate[:3, 3])
   # ic(mc.beststate)
   asym.state = mc.beststate
   wu.pdb.dump_pdb_from_points('stop.pdb', asym.coords)
   wu.pdb.dump_pdb_from_points('stopcoords.pdb', wu.hxform(mc.beststate.position, asym._coords))

   # ic('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
   # ic(bodies[0].contact_fraction(bodies[1]))
   # ic(bodies[0].contact_fraction(bodies[2]))
   if showme: wu.showme(bodies, name='end', **kw)
   # wu.showme(bodies[0], name='pairs01', showcontactswith=bodies[1], showpairsdist=16, col=(1, 1, 1))
   # wu.showme(bodies[0], name='pairs02', showcontactswith=bodies[2], showpairsdist=16, col=(1, 1, 1))
   # wu.showme(bodies[1], name='pairs10', showcontactswith=bodies[0], showpairsdist=16, col=(1, 0, 0))
   # wu.showme(bodies[2], name='pairs20', showcontactswith=bodies[0], showpairsdist=16, col=(0, 0, 1))

   # ic(coords.shape)
   # wu.showme(hpoint(coords))
   # coords = asym.coords
   # wu.showme(wu.hxform(mc.beststate, coords))

   return mc
