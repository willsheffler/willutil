import copy
import numpy as np
import willutil as wu

class ScrewError(Exception):
   pass

class SymElem:
   def __init__(
      self,
      nfold,
      axis,
      cen=[0, 0, 0],
      axis2=None,
      label=None,
      vizcol=None,
      scale=1,
      parent=None,
      children=None,
      hel=0,
   ):
      self._set_nfold(nfold)
      self.origcen = cen
      self.angle = np.pi * 2 / self.nfold
      self.axis = wu.homog.hnormalized(axis)
      self.axis2 = axis2
      self.hel = hel
      self.check_screw()
      self.scale = scale
      self._make_label(label)
      self._set_kind()
      self.place_center(cen)
      self.origin = np.eye(4)
      self.vizcol = vizcol
      self.axis[self.axis == -0] = 0
      self.cen[self.cen == -0] = 0
      self.id = None
      if self.axis2 is not None:
         self.axis2 = wu.homog.hnormalized(self.axis2)
         self.axis2[self.axis2 == -0] = 0

      self.mobile = False
      if wu.homog.hgeom.h_point_line_dist([0, 0, 0], cen, axis) > 0.0001: self.mobile = True
      if axis2 is not None and wu.hpointlinedis([0, 0, 0], cen, axis2) > 0.0001: self.mobile = True
      self.operators = self.make_operators()
      self.numops = len(self.operators)
      self.parent = parent
      self.children = children or list()

   def _set_nfold(self, nfold):
      if isinstance(nfold, str):
         self.label = nfold[:-2]  # strip componend nfolds
         if nfold[0] in 'CD':
            nfold = int(nfold[1:-2])
         else:
            self._opinfo = int(nfold[-2]), int(nfold[-1])
            assert nfold[0] in 'TO'
            nfold = dict(T=12, O=24)[nfold[0]]
      self.nfold = nfold

   def frame_operator_ids(self, frames):
      ids = np.arange(len(frames), dtype=np.int32)
      opsframes = einsum('oij,fjk->ofik', self.operators[1:], frames)
      for iop, opframes in enumerate(opsframes):
         a, b = np.where(np.all(np.isclose(frames[None], opframes[:, None]), axis=(-1, -2)))
         ids[a] = np.minimum(ids[a], ids[b])
      for i, id in enumerate(sorted(set(ids))):
         ids[ids == id] = i
      # for i in range(np.max(ids)):
      # wu.showme(frames[ids == i])
      # assert 0
      return ids

   def frame_component_ids(self, frames, permutations, sym=None):
      assert len(permutations) == len(frames)
      opframes = np.eye(4).reshape(1, 4, 4)
      iframematch0 = self.matching_frames(frames)
      fid = 0
      compid = -np.ones(len(frames))
      for iframe, perm in enumerate(permutations):
         iframematch = perm[iframematch0]
         iframematch = iframematch[iframematch >= 0]
         centest = einsum('fij,j->fi', frames[iframematch], self.cen)
         axstest = einsum('fij,j->fi', frames[iframematch], self.axis)
         assert np.allclose(centest, centest[0])
         assert np.allclose(axstest, axstest[0])
         if np.allclose(compid[iframematch], -1):
            compid[iframematch] = fid
            fid += 1
         else:
            assert min(compid[iframematch]) == max(compid[iframematch])
      return compid

   def matching_frames(self, frames):
      symaxs = einsum('fij,j->fi', frames, self.axis)
      symcen = einsum('fij,j->fi', frames, self.cen)
      match = np.logical_and(
         np.all(np.isclose(self.axis, symaxs), axis=1),
         np.all(np.isclose(self.cen, symcen), axis=1),
      )
      w = np.where(match)[0]
      return w

   def _make_label(self, label):
      if hasattr(self, 'label'): return
      self.label = label
      if self.label is None:
         if self.axis2 is None:
            self.label = f'C{self.nfold}'
            if self.screw != 0: self.label += f'{self.screw}'
         else:
            self.label = f'D{self.nfold}'

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

      self.screw = np.abs(self.axis) / self.hel
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

      if self.screw == 3 and self.nfold == 4:
         self.screw = self.nfold - self.screw
         self.axis = -self.axis
         self.hel = -self.hel

      if self.nfold == 3:
         if np.min(np.abs(self.axis)) < 0.1:
            self.hel = self.hel % 1
         else:
            self.hel = self.hel % np.sqrt(3)
      elif self.nfold == 4:
         self.hel = self.hel % 1.0
      elif self.nfold == 6:
         self.hel = self.hel % 1.0
      # assert self.screw <= self.nfold / 2

   def place_center(self, cen):
      self.cen = wu.homog.hgeom.hpoint(cen)
      if not self.iscyclic: return
      dist = wu.homog.line_line_distance_pa(cen, self.axis, _cube_edge_cen * self.scale, _cube_edge_axis)
      w = np.argmin(dist)
      newcen, _ = wu.homog.line_line_closest_points_pa(cen, self.axis, _cube_edge_cen[w] * self.scale, _cube_edge_axis[w])
      # ic(cen, newcen)
      if not np.any(np.isnan(newcen)):
         self.cen = newcen

   def _set_kind(self):
      self.iscyclic, self.isdihedral, self.istet, self.isoct, self.isscrew, self.iscompound = [False] * 6
      if self.label == 'T':
         self.kind, self.istet, self.iscompound = 'tet', True, True
      elif self.label == 'O':
         self.kind, self.isoct, self.iscompound = 'oct', True, True
      elif not np.isclose(self.hel, 0):
         assert self.axis2 is None
         self.kind, self.isscrew = 'screw', True
      elif self.axis2 is not None:
         self.kind, self.isdihedral, self.iscompound = 'dihedral', True, True
      else:
         self.kind, self.iscyclic = 'cyclic', True

   def make_operators(self):
      # ic(self)
      if self.label == 'T':
         ops = wu.sym.frames(self.label)
         assert self._opinfo == (3, 2)
         ops = wu.halign2([1, 1, 1], [0, 0, 1], self.axis, self.axis2) @ ops
         # ic(self.axis, self.axis2)
      elif self.label == 'O':
         assert self._opinfo in [(4, 3), (4, 2), (3, 2)]
         ops = wu.sym.frames(self.label)
      else:
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
      ax = (self.axis / np.max(np.abs(self.axis))).round(6)
      if np.allclose(ax.round(), ax):
         ax = ax.astype('i')
      if self.istet:
         ax2 = (self.axis2 / np.max(np.abs(self.axis2))).round(6)
         s = f'SymElem(\'T32\', axis={list(ax[:3])}, axis2={list(ax2[:3])}, cen={list(self.cen[:3])}, label=\'{self.label}\')'
      elif self.isoct:
         ax2 = (self.axis2 / np.max(np.abs(self.axis2))).round(6)
         s = f'SymElem(\'O43\', axis={list(ax[:3])}, axis2={list(ax2[:3])}, cen={list(self.cen[:3])}, label=\'{self.label}\')'
      elif self.axis2 is None:
         if self.screw == 0:
            s = f'SymElem({self.nfold}, axis={list(ax[:3])}, cen={list(self.cen[:3])}, label=\'{self.label}\')'
         else:
            s = f'SymElem({self.nfold}, axis={list(ax[:3])}, cen={list(self.cen[:3])}, hel={self.hel}, label=\'{self.label}\')'
      else:
         ax2 = (self.axis2 / np.max(np.abs(self.axis2))).round(6)
         s = f'SymElem({self.nfold}, axis={list(ax[:3])}, axis2={list(ax2[:3])}, cen={list(self.cen[:3])}, label=\'{self.label}\')'
      # s = s.replace('0.0,', '0,').replace('0.0],', '0]')
      return s

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
