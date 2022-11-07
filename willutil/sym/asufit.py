import functools as ft
import numpy as np
import willutil as wu

class RBObjective:
   def __init__(self, initial, lever=30, radbias=1, **kw):
      self.initial = initial.position
      self.lever = lever or initial.rog()
      self.radbias = radbias

   def __call__(self, bodies):
      tmp1 = self.initial.copy()
      tmp2 = bodies[0].position.copy()
      tmp1[:3, 3] /= self.radbias * bodies[0].comdirn()[:3]
      tmp2[:3, 3] /= self.radbias * bodies[0].comdirn()[:3]
      diff = wu.hdiff(tmp1, tmp2, lever=self.lever)
      b = bodies[0]
      clashes, contacts = 0, 0
      scores = list()
      for b2 in bodies[1:]:
         scores.append(-b.contacts(b2) / (b.clashes(b2) + 10)**2)
      assert len(scores) == 2
      a, b = scores
      score = (a + b)**(2) / (abs(a - b) + 1)
      return -score  #/ (diff + 3) * 3

class RBSampler:
   def __init__(self, cartsd=0.1, rotsd=None, lever=None, radbias=1.0, **kw):
      self.cartsd = cartsd
      self.rotsd = rotsd
      if rotsd == None:
         self.rotsd = cartsd / (lever or 30)
      else:
         assert lever == None, f'if rotsd specified, no lever must be provided'
      self.radbias = radbias

   def __call__(self, body, scale=1):
      perturb = wu.hrand(1, self.cartsd * scale, self.rotsd * scale)
      perturb[:3, 3] *= self.radbias * body.comdirn()[:3]
      body.moveby_com(perturb)
      return perturb

   def undo(self, body, move):
      body.moveby(wu.hinv(move))

def asufit(frames, coords, objfunc=None, sampler=None, mc=None, **kw):
   asym = wu.RigidBody(coords)
   bodies = [asym] + [wu.RigidBody(parent=asym, xfromparent=x) for x in frames[1:]]

   if objfunc is None: objfunc = RBObjective(asym, **kw)
   if sampler is None: sampler = RBSampler(**kw)
   if mc is None: mc = wu.MonteCarlo(objfunc, temperature=0.01, **kw)

   wu.showme(bodies, name='start')
   for i in range(100):
      move = sampler(asym)
      accept = mc.try_this(bodies)
      if not accept:
         sampler.undo(asym, move)
      else:
         ic(mc.last)

   wu.showme(bodies, name='end')
   assert 0
