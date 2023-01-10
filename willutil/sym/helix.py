import numpy as np
import willutil as wu

class Helix:
   """helical symmetry"""
   def __init__(self, turns, phase, nfold=1, turnsB=1):
      self.nfold = nfold
      self.turns = turns
      self.phase = phase
      self.turnsB = turnsB
      # assert nfold == 1
      assert turnsB == 1
      if phase < 0 or phase > 1:
         raise ValueError(f'helix phase must be 0-1, if you need beyond this range, adjust nturns')

   def frames(self, radius, spacing, turns=1, maxdist=9e9, start=None, closest=0, **kw):
      '''phase is a little artifical here, as really it just changes self.turns
         "central" frame will be ontop. if closest is given, frames will be sorted on dist to cen
         otherwise central frame will be first, then others in order from bottom to top
      '''
      axis = np.array([0, 0, 1, 0])
      if isinstance(turns, (int, float)):
         turns = (-turns, turns)
      if start is None:
         start = np.eye(4)
         start[0, 3] = radius
      ang = 2 * np.pi / (self.turns + self.phase)
      lb = turns[0] * self.turns - 1
      ub = turns[1] * self.turns + 2
      # ic(turns, self.turns, lb, ub)
      frames = list()
      for icyc in range(self.nfold):
         xcyc = wu.hrot(axis, (np.pi * 2) / self.nfold * icyc, degrees=False)
         frames += [xcyc @ wu.hrot(axis, i * ang, hel=i * spacing / self.turns, degrees=False) for i in range(lb, ub)]

      frames = np.stack(frames)
      frames = wu.hxform(frames, start)
      dist = wu.hnorm(frames[:, :, 3] - start[:, 3])
      frames = frames[np.argsort(dist)]
      frames = frames[dist <= maxdist]
      if closest > 0:
         frames = frames[:closest]
      return frames
