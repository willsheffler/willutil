import numpy as np
import willutil as wu

from numba import njit

class SymScaffIcos:
   """represents framework for placing stuff on icosahedrol"""
   def __init__(self):
      self.sym = 'icos'
      self.occ = np.zeros((60, 3), dtype=bool)
      self.perm = wu.sym.permutations(self.sym)

      self.nbr2 = self.perm[:, 34]
      self.nbr3a = self.perm[:, 47]
      self.nbr3b = self.perm[:, 48]
      self.nbr3 = np.stack([self.nbr3a, self.nbr3b], axis=1)
      self.nbr5a = self.perm[:, 11]
      self.nbr5b = self.perm[:, 53]
      self.nbr5c = self.perm[:, 56]
      self.nbr5d = self.perm[:, 14]
      self.nbr5 = np.stack([self.nbr5a, self.nbr5b, self.nbr5c, self.nbr5d], axis=1)
      self.nbrs = (self.nbr2, self.nbr3, self.nbr5)

      self.placed = list()

   def reset(self):
      self.occ[:] = False
      self.placed.clear()

#      if r < 0.333:
#         pos = [(isub, 1), (isub, 2), (nbr5[isub, 0], 2)]
#         # pos = [(isub, 2), (nbr5[isub, 0], 2), (nbr5[isub, 3], 2)]  # 6 missing, can be 3 together
#      elif r < 0.666:
#         pos = [(isub, 1), (isub, 2), (nbr5[isub, 3], 1)]
#         # pos = [(isub, 2), (nbr3[isub, 1], 2), (nbr5[isub, 0], 2)]  # 6 missing, can be 3 together
#      else:
#         pos = [(isub, 1), (nbr3[isub, 0], 1)]
#
#      # elif i == 2:
#      # pos = [(isub, 1), (isub, 2), (nbr3[isub, 1], 1)]
#
#      # pos = ((isub, 2), (nbr3[isub, 0], 2), (nbr5[isub, 0], 2))  # full
#
#      # pos = [(isub, 2), (nbr3[isub, 1], 2), (nbr5[isub, 0], 2)]  # 6 missing, can be 3 together
#      # pos = [(isub, 2), (nbr2[isub], 2), (nbr5[isub, 3], 2)]  # 6 missing, can be 3 together # same

@njit
def pos_try3(i, isub, nbrs):
   # can assemble with single #1 and one missing 5fold
   # can also assemble less complete structs wtih with more #1s
   nbr2, nbr3, nbr5 = nbrs
   r = np.random.rand()
   if i == 0:
      pos = [
          (isub, 0),
          (isub, 2),
          (nbr2[isub], 0),
      ]
   elif r < 0.5:
      pos = [
          (isub, 0),
          (isub, 2),
          (nbr2[isub], 0),
          (nbr3[isub, 0], 2),
      ]  # ***
      # pos = [(isub, 0), (isub, 2), (nbr2[isub], 0)]
   else:
      pos = [
          (isub, 0),
          (isub, 2),
          (nbr5[isub, 3], 2),
          (nbr2[isub], 0),
      ]  # ***
   return pos

@njit
def shape_553line(isub, nbrs):
   nbr2, nbr3, nbr5 = nbrs
   return [
       (isub, 2),
       (nbr5[isub, 0], 2),
       (nbr2[isub], 1),
   ]

@njit
def shape_553longtri(isub, nbrs):
   nbr2, nbr3, nbr5 = nbrs
   return [
       (isub, 2),
       (nbr5[isub, 0], 2),
       (nbr3[isub, 1], 1),
   ]

@njit
def shape_3_5_5a5_triangle(isub, nbrs):
   nbr2, nbr3, nbr5 = nbrs
   return [
       (isub, 1),
       (isub, 2),
       (nbr5[isub, 0], 2),
   ]

@njit
def pos_try4(i, isub, nbrs):
   # can assemble with single #1 and one missing 5fold
   # can also assemble less complete structs wtih with more #1s
   nbr2, nbr3, nbr5 = nbrs
   r = np.random.rand()
   if r < 10.5:
      pos = shape_3_5_5a5_triangle(isub, nbrs)
   else:
      pos = [
          (isub, 1),
          (nbr3[isub, 0], 1),
          (nbr5[isub, 3], 2),
      ]
   return pos

   # if placed:
   # iocc = np.random.choice(np.where(occ[:, 2])[0])
   # isub = np.random.choice(nbr5[iocc])
   # if occ[isub, 2]: continue
   # else:

@njit
def mark555(occ, nbrs, ntries=500):
   placed = list()
   for i in range(ntries):
      isub = int(np.random.rand() * 60)

      # pos = pos_try3(i, isub, nbrs)
      # pos = [(isub, 2), (nbr2[isub], 2), (nbr5[isub, 3], 2)]  # 6 missing, can be 3 together # same
      pos = pos_try4(i, isub, nbrs)

      clash = False
      for i, a in pos:
         clash |= occ[i, a]
      if not clash:
         for i, a in pos:
            occ[i, a] = True
         placed.append(pos)

   return placed
