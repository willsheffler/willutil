import functools as ft
import numpy as np
import willutil as wu
from willutil.homog import *

class RBObjective:
   def __init__(
      self,
      initial,
      bodies=None,
      contactfrac=None,
      scoreframes=None,
      clashframes=None,
      lever=20,
      biasradial=None,
      biasdir=None,
      contactdist=None,
      driftpenalty=1,
      clashpenalty=1,
      angpenalty=1,
      spreadpenalty=1,
      minradius=0,
      **kw,
   ):

      self.initial = initial.position.copy()
      self.initialcom = initial.com().copy()
      self.lever = lever or initial.rog()
      self.contactfrac = contactfrac
      assert 0 <= contactfrac <= 1
      self.scoreframes = scoreframes
      self.clashframes = clashframes
      self.bodies = bodies
      self.biasdir = hnormalized(biasdir)
      self.biasradial = biasradial
      self.contactdist = contactdist

      self.driftpenalty = driftpenalty
      self.clashpenalty = clashpenalty
      self.angpenalty = angpenalty
      self.spreadpenalty = spreadpenalty
      self.minradius = minradius

   def __call__(self, position, verbose=False):
      asym = self.bodies[0]
      asym.position = position

      tmp1 = self.initial.copy()
      p = hproj(self.biasdir, hcart3(tmp1))
      pp = hprojperp(self.biasdir, hcart3(tmp1))
      tmp1[:3, 3] = p[:3] / self.biasradial + pp[:3]
      tmp2 = self.bodies[0].position.copy()
      p = hproj(self.biasdir, hcart3(tmp2))
      pp = hprojperp(self.biasdir, hcart3(tmp2))
      tmp2[:3, 3] = p[:3] / self.biasradial + pp[:3]
      xdiff = hdiff(tmp1, tmp2, lever=self.lever)

      clashfrac, contactfrac = 0, 0
      scores = list()
      clash = 0
      fracs = list()
      # bods = self.bodies[1:] if self.scoreframes is None else [self.bodies[i] for i in self.scoreframes]
      for ib, b in enumerate(self.bodies):
         for jb, b2 in enumerate(self.bodies):
            if (ib, jb) in self.scoreframes:
               f1, f2 = b.contact_fraction(b2, contactdist=self.contactdist)
               # if verbose: ic(ib, jb, f1, f2)
               fracs.extend([f1, f2])
               diff11 = max(0, f1 - self.contactfrac) / (1 - self.contactfrac)
               diff12 = max(0, self.contactfrac - f1) / self.contactfrac
               diff21 = max(0, f2 - self.contactfrac) / (1 - self.contactfrac)
               diff22 = max(0, self.contactfrac - f2) / self.contactfrac

               # ic(f1, diff11, diff12, f2, diff21, diff22)

               scores.append((max(diff11, diff12)**2))
               scores.append((max(diff21, diff22)**2))
            elif (ib, jb) in self.clashframes:
               clash += (self.clashpenalty / 10 * (b.clashes(b2) / len(b)))**2

      # ic([int(_) for _ in scores])
      # ic([int(_ * 100) for _ in fracs])
      # ic(max(scores), (self.driftpenalty * xdiff)**2)

      # zxang0 = wu.homog.dihedral([0, 0, 1], [0, 0, 0], [1, 0, 0], self.initialcom)
      ax1 = wu.sym.axes('icos')[2]
      ax2 = wu.sym.axes('icos')[3]
      nf1rot = wu.homog.dihedral([0, 0, 1], [0, 0, 0], ax1, asym.com())
      nf2rot = wu.homog.dihedral([0, 0, -1], [0, 0, 0], ax2, asym.com())

      angokrange = np.pi / 8
      # angdiff = max(0, abs(zxang0 - zxang) - angokrange)
      angdiff1 = max(0, abs(nf1rot) - angokrange)
      angdiff2 = max(0, abs(nf2rot) - angokrange)
      axsdist1 = wu.hnorm(wu.hprojperp(ax1, asym.com()))
      axsdist2 = wu.hnorm(wu.hprojperp(ax2, asym.com()))
      angdiff1 = angdiff1 * axsdist1
      angdiff2 = angdiff2 * axsdist2
      # ic(angdiff1)
      # ic(angdiff2)
      # ic(nf1rot, nf2rot)
      # ic(abs(zxang0 - zxang))
      # ic(angdiff)
      # ic(xdiff, max(scores))

      scores[0] *= 2
      scores[1] *= 2
      if verbose:
         # ic(scores)
         ic(fracs)
         # ic((self.driftpenalty * xdiff)**2)
         # ic((self.angpenalty * 10 * angdiff * wu.hnorm(wu.hprojperp([1, 0, 0], asym.com())))**2)
      s = [
         10 * sum(scores),
         # (wu.hnorm(asym.com()) - self.minradius)**2 * 100,
         (self.spreadpenalty * (max(fracs) - min(fracs)))**2,
         (self.driftpenalty * xdiff)**2,
         (self.angpenalty * angdiff1)**2,
         (self.angpenalty * angdiff2)**2,
         0.1 * (axsdist1 + axsdist2)
      ]
      # ic(s)
      return np.sum(s)

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
      # self.bodies[0].position = position
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

def asufit(
   sym,
   coords,
   frames=None,
   objfunc=None,
   sampler=None,
   mc=None,
   showme=False,
   cartsd=None,
   nf1=2,
   nf2=3,
   iterations=300,
   lever=None,
   temperature=1,
   thresh=0.000001,
   minradius=None,
   resetinterval=100,
   correctionfactor=2,
   **kw,
):
   kw = wu.Bunch(kw)
   asym = wu.RigidBody(coords, **kw)
   bodies = [asym] + [wu.RigidBody(parent=asym, xfromparent=x, **kw) for x in frames[1:]]
   if kw.biasradial is None: kw.biasradial = wu.sym.symaxis_radbias(sym, 2, 3)
   kw.biasdir = wu.hnormalized(asym.com())
   if frames is None: frames = wu.sym.frames(sym)
   if lever is None: kw.lever = asym.rog() * 1.5
   if minradius is None: kw.minradius = wu.hnorm(asym.com()) * 0.7

   if objfunc is None:
      objfunc = RBObjective(
         asym,
         scoreframes=[(0, 1), (0, 2)],
         clashframes=[(1, 2), (1, 3)],
         bodies=bodies,
         **kw,
      )
   if sampler is None:
      sampler = RBSampler(
         cartsd=cartsd,
         center=asym.com(),
         **kw,
      )
   if mc is None:
      mc = wu.MonteCarlo(objfunc, temperature=temperature, **kw)

   mc.try_this(asym.position)
   initialscore = mc.best

   # ic(mc.bestconfig)
   # if showme:
   # wu.showme(bodies, name='start')
   # ic(asym.coords.shape, 'start')
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

   # if showme:
   # ic('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
   # ic(bodies[0].contact_fraction(bodies[1]))
   # ic(bodies[0].contact_fraction(bodies[2]))
   # wu.showme(bodies, name='end')
   # wu.showme(bodies[0], name='pairs01', showcontactswith=bodies[1], showpairsdist=16, col=(1, 1, 1))
   # wu.showme(bodies[0], name='pairs02', showcontactswith=bodies[2], showpairsdist=16, col=(1, 1, 1))
   # wu.showme(bodies[1], name='pairs10', showcontactswith=bodies[0], showpairsdist=16, col=(1, 0, 0))
   # wu.showme(bodies[2], name='pairs20', showcontactswith=bodies[0], showpairsdist=16, col=(0, 0, 1))

   # ic(coords.shape)
   # wu.showme(hpoint(coords))
   # coords = asym.coords
   # wu.showme(wu.hxform(mc.bestconfig, coords))

   return mc
