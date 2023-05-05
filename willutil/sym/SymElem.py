import copy
import numpy as np
import willutil as wu

class ScrewError(Exception):
   pass

class SymElem:
   def __init__(self, nfold, axis, cen=[0, 0, 0], axis2=None, label=None, vizcol=None, scale=1, parent=None, children=None, hel=0):
      self.nfold = nfold
      self.origcen = cen
      self.angle = np.pi * 2 / self.nfold
      self.axis = wu.homog.hnormalized(axis)
      self.axis2 = axis2
      self.hel = hel
      self.check_screw()
      self.scale = scale
      self.iscyclic = axis2 is None
      self.isdihedral = axis2 is not None
      self.place_center(cen)
      self.vizcol = vizcol
      if self.axis2 is not None:
         self.axis2 = wu.homog.hnormalized(self.axis2)
      self.origin = np.eye(4)
      self.label = label
      if self.label is None:
         if axis2 is None:
            self.label = f'C{self.nfold}'
            if self.screw != 0: self.label += f'{self.screw}'
         else: self.label = f'D{self.nfold}'
      self.mobile = False
      if wu.homog.hgeom.h_point_line_dist([0, 0, 0], cen, axis) > 0.0001: self.mobile = True
      if axis2 is not None and wu.hpointlinedis([0, 0, 0], cen, axis2) > 0.0001: self.mobile = True
      self.operators = self.make_operators()
      self.numops = len(self.operators)
      self.parent = parent
      self.children = children or list()

   def __eq__(self, other):
      if self.nfold != other.nfold: return False
      if not np.allclose(self.axis, other.axis): return False
      if self.axis2 is not None and not np.allclose(self.axis2, other.axis2): return False
      if not np.allclose(self.cen, other.cen): return False
      if not np.allclose(self.hel, other.hel): return False
      if not np.allclose(self.screw, other.screw): return False
      assert np.allclose(self.operators, other.operators)
      return True

   def check_screw(self):
      if self.hel == 0.0:
         self.screw = 0
         return
      assert not self.axis2

      self.screw = self.axis / self.hel
      self.screw = 1 / self.screw[np.argmax(np.abs(self.screw))]
      self.screw = self.nfold * self.screw

      # ic(self.nfold, self.axis, self.hel, self.screw)
      if not all([
            np.allclose(round(self.screw), self.screw),
            self.screw < self.nfold,
            self.screw > -self.nfold,
      ]):
         raise ScrewError()

      assert np.isclose(self.screw, round(self.screw))
      self.screw = int(round(self.screw))

      if self.screw < 0: self.screw += self.nfold

#      if self.screw == 3 and self.nfold == 4:
#         self.screw = self.nfold - self.screw
#         self.axis = -self.axis
#         self.hel = -self.hel
#
#      if self.nfold == 3:
#         if np.min(np.abs(self.axis)) < 0.1:
#            self.hel = self.hel % 1
#         else:
#            self.hel = self.hel % np.sqrt(3)
#      elif self.nfold == 4:
#         self.hel = self.hel % 1.0
#      elif self.nfold == 6:
#         self.hel = self.hel % 1.0
#      # assert self.screw <= self.nfold / 2

   def place_center(self, cen):
      self.cen = wu.homog.hgeom.hpoint(cen)
      if self.isdihedral: return
      dist = wu.homog.line_line_distance_pa(cen, self.axis, _cube_edge_cen * self.scale, _cube_edge_axis)
      w = np.argmin(dist)
      newcen, _ = wu.homog.line_line_closest_points_pa(cen, self.axis, _cube_edge_cen[w] * self.scale, _cube_edge_axis[w])
      # ic(cen, newcen)
      if not np.any(np.isnan(newcen)):
         self.cen = newcen

   def make_operators(self):
      # ic(self)
      x = wu.homog.hgeom.hrot(self.axis, nfold=self.nfold, center=self.cen)
      ops = [wu.homog.hgeom.hpow(x, p) for p in range(self.nfold)]
      if self.axis2 is not None:
         xd2f = wu.homog.hgeom.hrot(self.axis2, nfold=2, center=self.cen)
         ops = ops + [xd2f @ x for x in ops]
      if self.hel != 0.0:
         for i, x in enumerate(ops):
            x[:, 3] += self.axis * self.hel * i
      ops = np.stack(ops)
      assert wu.homog.hgeom.hvalid(ops)
      return ops

   @property
   def coords(self):
      axis2 = [0, 0, 0, 0] if self.axis2 is None else self.axis2
      return np.stack([self.axis, axis2, self.cen])

   def xformed(self, xform):
      assert xform.shape[-2:] == (4, 4)
      single = False
      if xform.ndim == 2:
         xform = xform.reshape(1, 4, 4)
         single = True
      result = list()
      for x in xform:
         # other = copy.copy(self)
         # other.axis = wu.hxform(x, self.axis)
         # if self.axis2 is not None: other.axis2 = wu.hxform(x, self.axis2)
         # other.cen = wu.hxform(x, self.cen)
         # other.make_operators()
         axis = wu.hxform(x, self.axis)
         axis2 = None if self.axis2 is None else wu.hxform(x, self.axis2)
         cen = wu.hxform(x, self.cen)
         other = SymElem(self.nfold, axis, cen, axis2, self.label, self.vizcol, 1.0)  #self.scale)
         result.append(other)
      if single:
         result = result[0]
      return result

   def __repr__(self):
      # ax = self.axis / min(self.axis[self.axis != 0])
      ax = (self.axis / np.max(np.abs(self.axis))).round(6)
      if self.axis2 is None:
         if self.screw == 0:
            return f'SymElem({self.nfold}, axis={list(ax[:3])}, cen={list(self.cen[:3])})'
         else:
            return f'SymElem({self.nfold}, axis={list(ax[:3])}, cen={list(self.cen[:3])}, hel={self.hel})'
      else:
         ax2 = (self.axis2 / np.max(np.abs(self.axis2))).round(6)
         return f'SymElem({self.nfold}, axis={list(ax[:3])}, axis2={list(ax2[:3])}, cen={list(self.cen[:3])})'

_cubeedges = [
   [[0, 0, 0], [1, 0, 0]],
   [[0, 0, 0], [0, 0, 1]],
   [[0, 0, 0], [0, 1, 0]],
   [[0, 0, 1], [1, 0, 0]],
   [[0, 1, 0], [1, 0, 0]],
   [[0, 1, 0], [0, 0, 1]],
   [[1, 0, 0], [0, 0, 1]],
   [[0, 0, 1], [0, 1, 0]],
   [[1, 0, 0], [0, 1, 0]],
   [[0, 1, 1], [1, 0, 0]],
   [[1, 1, 0], [0, 0, 1]],
   [[1, 0, 1], [0, 1, 0]],
]
_cube_edge_cen, _cube_edge_axis = np.array(_cubeedges).swapaxes(0, 1)
