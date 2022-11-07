import numpy as np
import willutil as wu

class RigidBody:
   def __init__(self, coords=None, extra=None, position=np.eye(4), parent=None, xfromparent=None, contactdis=8,
                clashdis=3, **kw):
      self.extra = extra
      self.parent = parent
      self.xfromparent = xfromparent
      self._position = position
      self._coords = None
      self.bvh = None
      if coords is not None:
         self._coords = wu.hpoint(coords)
         self.bvh = wu.cpp.bvh.BVH(coords[..., :3])
      elif parent is not None:
         self.bvh = parent.bvh
         self._coords = parent._coords
      self.clashdis = clashdis
      self.contactdis = contactdis

   def __len__(self):
      return len(self._coords)

   def moveby(self, x):
      x = np.asarray(x)
      if x.ndim == 1:
         x = wu.htrans(x)
      self.position = wu.hxform(x, self.position)

   def moveby_com(self, x):
      x = np.asarray(x)
      if x.ndim == 1:
         x = wu.htrans(x)
      com = self.com()
      self.moveby(-com)
      self.position = wu.hxform(x, self.position)
      self.moveby(com)

   @property
   def position(self):
      if self.parent != None:
         return self.xfromparent @ self.parent.position
      return self._position

   @position.setter
   def position(self, newposition):
      if self.parent != None:
         raise ValueError(f'RigidBody with parent cant have position set')
      self._position = newposition

   @property
   def coords(self):
      return wu.hxform(self.position, self._coords)

   def com(self):
      return self.position @ self.bvh.com()

   def comdirn(self):
      return wu.hnormalized(self.bvh.com())

   def rog(self):
      d = self.coords - self.com()
      return np.sqrt(np.sum(d**2) / len(d))

   def contact_count(self, other, maxdis):
      assert isinstance(other, RigidBody)
      return wu.cpp.bvh.bvh_count_pairs(self.bvh, other.bvh, self.position, other.position, maxdis)

   def contacts(self, other):
      return self.contact_count(other, self.contactdis)

   def clashes(self, other):
      return self.contact_count(other, self.clashdis)

   def conntact_pairs(self, other):
      assert isinstance(other, RigidBody)

      if pos1 is None: pos1 = self.pos
      if pos2 is None: pos2 = other.pos
      if not buf:
         buf = np.empty((10000, 2), dtype="i4")
      pairs, overflow = rp.bvh.bvh_collect_pairs(
         self.bvh_bb_atomno if atomno else (self.bvh_bb if use_bb else self.bvh_cen), other.bvh_bb_atomno if atomno else
         (other.bvh_bb if use_bb else other.bvh_cen), pos1, pos2, maxdis, buf)
      assert not overflow
      return pairs
