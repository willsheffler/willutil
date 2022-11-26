import numpy as np
import willutil as wu

class RigidBody:
   def __init__(
         self,
         coords=None,
         extra=None,
         position=np.eye(4),
         parent=None,
         xfromparent=None,
         contactdis=8,
         clashdis=3,
         usebvh=True,
         scale=1,
         **kw,
   ):
      self.extra = extra
      self.parent = parent
      self.xfromparent = xfromparent
      if self.xfromparent is not None:
         assert wu.hvalid(self.xfromparent)
      self._position = position
      self._coords = None
      self.bvh = None
      if coords is not None:
         self._coords = wu.hpoint(coords)
         self._com = wu.hcom(self._coords)
         if usebvh:
            self.bvh = wu.cpp.bvh.BVH(coords[..., :3])
      elif parent is not None:
         self.bvh = parent.bvh
         self._coords = parent._coords
         self._com = wu.hcom(self._coords)

      self.clashdis = clashdis
      self.contactdis = contactdis
      self.usebvh = usebvh
      self._scale = scale

   def __len__(self):
      return len(self._coords)

   @property
   def state(self):
      assert self.parent is None
      state = wu.Bunch(position=self.position, scale=self.scale())
      assert isinstance(state.scale, (int, float))
      return state

   @state.setter
   def state(self, state):
      assert self.parent is None
      self.position = state.position
      self.set_scale(state.scale)

   def moveby(self, x):
      x = np.asarray(x)
      if x.ndim == 1:
         x = wu.htrans(x)
      self.position = wu.hxform(x, self.position)
      assert wu.hvalid(self.position)

   def move_about_com(self, x):
      x = np.asarray(x)
      if x.ndim == 1:
         x = wu.htrans(x)
      com = self.com()
      self.moveby(-com)
      self.position = wu.hxform(x, self.position)
      self.moveby(com)

   def set_scale(self, scale):
      assert self.parent is None
      assert isinstance(scale, (int, float))
      self._scale = scale

   def scale(self):
      if self.parent is None:
         return self._scale
      return self.parent.scale()

   @property
   def position(self):
      if self.parent is None:
         return self._position
      x = self.xfromparent.copy()
      x[:3, 3] *= self.scale()
      return x @ self.parent.position

   @position.setter
   def position(self, newposition):
      if self.parent != None:
         raise ValueError(f'RigidBody with parent cant have position set')
      if newposition.shape[-2:] != (4, 4):
         raise ValueError(f'RigidBody position is 4,4 matrix (not point)')
      self._position = newposition.reshape(4, 4)

   @property
   def coords(self):
      return wu.hxform(self.position, self._coords)

   def com(self):
      return self.position @ self._com

   def comdirn(self):
      return wu.hnormalized(self.com())

   def rog(self):
      d = self.coords - self.com()
      return np.sqrt(np.sum(d**2) / len(d))

   def contact_count(self, other, contactdist, usebvh=None):
      assert isinstance(other, RigidBody)
      if usebvh or (usebvh is None and self.usebvh):
         count = wu.cpp.bvh.bvh_count_pairs(self.bvh, other.bvh, self.position, other.position, contactdist)
      else:
         # import scipy.spatial
         # d = scipy.spatial.distance_matrix(self.coords, other.coords)
         d = wu.hnorm(self.coords[None] - other.coords[:, None])
         count = np.sum(d < contactdist)
      return count

   def contacts(self, other):
      return self.contact_count(other, self.contactdis)

   def clashes(self, other):
      return self.contact_count(other, self.clashdis)

   def point_contact_count(self, other, contactdist=8):
      p = self.interactions(other, contactdist=contactdist)
      a = set(p[:, 0])
      b = set(p[:, 1])
      # ic(a)
      # ic(b)
      return len(a), len(b)

   def contact_fraction(self, other, contactdist=8):

      # wu.pdb.dump_pdb_from_points('bodyA.pdb', self.coords)
      # wu.pdb.dump_pdb_from_points('bodyB.pdb', other.coords)

      p = self.interactions(other, contactdist=contactdist)
      a = set(p[:, 0])
      b = set(p[:, 1])
      # ic(len(a), len(self.coords))
      # ic(len(b), len(other.coords))

      return len(a) / len(self.coords), len(b) / len(self.coords)

   def interactions(self, other, contactdist=8, buf=None, usebvh=None):
      assert isinstance(other, RigidBody)
      if usebvh or (usebvh is None and self.usebvh):
         if not buf: buf = np.empty((100000, 2), dtype="i4")
         pairs, overflow = wu.cpp.bvh.bvh_collect_pairs(self.bvh, other.bvh, self.position, other.position, contactdist,
                                                        buf)
         assert not overflow
      else:
         d = wu.hnorm(self.coords[None] - other.coords[:, None])
         pairs = np.stack(np.where(d <= contactdist), axis=1)
      return pairs
