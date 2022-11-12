import functools as ft
import numpy as np
import willutil as wu
from willutil.homog import *

def asufit(
   sym,
   coords,
   symaxes,
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
   if minradius is None: kw.minradius = wu.hnorm(asym.com()) * 0.7

   if objfunc is None:
      objfunc = wu.rigid.RBObjective(
         asym,
         scoreframes=[(0, 1), (0, 2)],
         clashframes=[(1, 2), (1, 3), (2, 3)],
         bodies=bodies,
         sym=sym,
         symaxes=symaxes,
         **kw,
      )
   if sampler is None:
      sampler = wu.search.RBSampler(
         cartsd=cartsd,
         center=asym.com(),
         **kw,
      )
   if mc is None:
      mc = wu.MonteCarlo(objfunc, temperature=temperature, **kw)

   mc.try_this(asym.position)
   initialscore = mc.best

   if showme: wu.showme(bodies, name='start')

   wu.pdb.dump_pdb_from_points('start.pdb', asym.coords)

   for i in range(iterations):
      if i % 50 == 0 and i > 0:
         asym.position = mc.bestconfig
         if i % 200:
            asym.position = mc.startconfig
         if mc.acceptfrac < 0.1:
            mc.temperature *= correctionfactor / resetinterval * 100
            sampler.cartsd /= correctionfactor / resetinterval * 100
         if mc.acceptfrac > 0.3:
            mc.temperature /= correctionfactor / resetinterval * 100
            sampler.cartsd *= correctionfactor / resetinterval * 100
         # ic(mc.acceptfrac, mc.best)

      pos, prev = sampler(asym.position)
      accept = mc.try_this(pos)
      if not accept:
         asym.position = prev
      else:
         if mc.best < thresh:
            # ic('end', i, objfunc(mc.bestconfig))
            return mc

         # ic(i, mc.last)
         # if i % 10 == 0:
         # if showme: wu.showme(bodies, name='mid%i' % i)
   assert mc.bestconfig is not None
   # ic('end', mc.best)
   initscore = objfunc(mc.startconfig, verbose=True)
   stopscore = objfunc(mc.bestconfig, verbose=True)
   ic('init', initscore)
   ic('stop', stopscore)
   # ic(mc.bestconfig[:3, :3])
   # ic(mc.bestconfig[:3, 3])
   # ic(mc.bestconfig)
   asym.position = mc.bestconfig
   wu.pdb.dump_pdb_from_points('stop.pdb', asym.coords)
   wu.pdb.dump_pdb_from_points('stopcoords.pdb', wu.hxform(mc.bestconfig, asym._coords))

   # ic('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
   # ic(bodies[0].contact_fraction(bodies[1]))
   # ic(bodies[0].contact_fraction(bodies[2]))
   if showme: wu.showme(bodies, name='end')
   # wu.showme(bodies[0], name='pairs01', showcontactswith=bodies[1], showpairsdist=16, col=(1, 1, 1))
   # wu.showme(bodies[0], name='pairs02', showcontactswith=bodies[2], showpairsdist=16, col=(1, 1, 1))
   # wu.showme(bodies[1], name='pairs10', showcontactswith=bodies[0], showpairsdist=16, col=(1, 0, 0))
   # wu.showme(bodies[2], name='pairs20', showcontactswith=bodies[0], showpairsdist=16, col=(0, 0, 1))

   # ic(coords.shape)
   # wu.showme(hpoint(coords))
   # coords = asym.coords
   # wu.showme(wu.hxform(mc.bestconfig, coords))

   return mc
