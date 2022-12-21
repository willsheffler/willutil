import numpy as np
import willutil as wu

class RigidBodyFollowers:
   def __init__(self, bodies=None, coords=None, frames=None, cellsize=1, **kw):
      if bodies is not None:
         self.asym = bodies[0]
         self.symbodies = bodies[1:]
         self.bodies = bodies
      elif frames is not None:
         self.asym = RigidBody(coords, **kw)
         self.symbodies = [RigidBody(parent=self.asym, xfromparent=x, **kw) for x in frames[1:]]
         self.bodies = [self.asym] + self.symbodies
      self.cellsize = cellsize

   def clashes(self, nbrs=None):
      if isinstance(nbrs, int): nbrs = [nbrs]
      if nbrs is None:
         clsh = [self.asym.clashes(b) for b in self.symbodies]
         return any(clsh)
      return any(self.asym.clashes(self.bodies[i]) for i in nbrs)

   def contact_fraction(self, nbrs=None):
      if isinstance(nbrs, int): nbrs = [nbrs]
      if nbrs is None:
         return [self.asym.contact_fraction(b) for b in self.symbodies]
      else:
         return [self.asym.contact_fraction(self.bodies[i]) for i in nbrs]

   def scale_frames(self, scalefactor, safe=True):
      self.cellsize *= scalefactor
      changed = any([b.scale_frame(scalefactor) for b in self.bodies])
      if not changed:
         if safe:
            raise ValueError(f'no frames could be scaled, scale_frames only valid for unbounded symmetry')
      return changed

   def get_neighbors_by_axismatch(self, axis, perp=False):
      nbrs = list()
      for i in range(1, len(self.bodies)):
         tonbaxis = wu.haxisof(self.bodies[i].xfromparent)
         ang = wu.hangline(tonbaxis, axis)
         # ic(perp, ang, axis, tonbaxis)
         if (not perp and ang > 0.001) or (perp and abs(ang - np.pi / 2) < 0.001):
            nbrs.append(i)
      return nbrs

   def frames(self):
      return np.stack([b.xfromparent for b in self.bodies])

   def __len__(self):
      return len(self.bodies)

class RigidBody:
   def __init__(
      self,
      coords=None,
      contact_coords=None,
      extra=None,
      position=np.eye(4),
      parent=None,
      xfromparent=np.eye(4),
      contactdis=8,
      clashdis=3,
      usebvh=True,
      scale=1,
      interacting_points=None,
      recenter=False,
      **kw,
   ):
      self.extra = extra
      self.parent = parent
      self.xfromparent = xfromparent.copy()
      assert wu.hvalid(self.xfromparent)

      self._position = position
      self._coords = None
      self.bvh = None
      self.contactbvh = None
      self.clashdis = clashdis
      self.contactdis = contactdis
      self.usebvh = usebvh
      self._scale = scale
      self.tolocal = np.eye(4)
      self.toglobal = np.eye(4)
      assert (parent is None) != (coords is None)
      if coords is not None:
         coords = coords.copy()
         if contact_coords is None: contact_coords = coords
         contact_coords = contact_coords.copy()
         if recenter:
            # oldcom =
            self.tolocal = wu.htrans(-wu.hcom(coords))
            self.toglobal = wu.hinv(self.tolocal)
            coords = wu.hxform(self.tolocal, coords)
            contact_coords = wu.hxform(self.tolocal, contact_coords)
            # position must be set to move coords back to gloabal frame
            self.position = self.toglobal.copy()
         self._coords = wu.hpoint(coords)
         self._contact_coords = wu.hpoint(contact_coords)
         self._com = wu.hcom(self._coords)
         if usebvh:
            self.bvh = wu.cpp.bvh.BVH(coords[..., :3])
            self.contactbvh = wu.cpp.bvh.BVH(contact_coords[..., :3])
      elif parent is not None:
         self.bvh = parent.bvh
         self.contactbvh = parent.contactbvh
         self._coords = parent._coords
         self._com = wu.hcom(self._coords)
         parent.children.append(self)
         self.clashdis = parent.clashdis
         self.contactdis = parent.contactdis
         self.usebvh = parent.usebvh
         self._scale = parent.scale

      self.children = list()

   def __len__(self):
      return len(self._coords)

   def scale_frame(self, scalefactor):
      if self.xfromparent is not None:
         if wu.hnorm(self.xfromparent[:, 3]) > 0.0001:
            self.xfromparent = wu.hscaled(scalefactor, self.xfromparent)
            return True
      return False

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
   def globalposition(self):
      assert self.parent is None
      # self.positions has been set to move local coords into intial global frame
      # tolocal moves position so identity doesn't move global frame coords
      # yeah, confusing... self.position 'moves' opposite of intuition
      return wu.hxform(self.tolocal, self.position)

   @property
   def coords(self):
      return wu.hxform(self.position, self._coords)

   @property
   def globalcoords(self):
      return wu.hxform(self.globalposition, self._coords)

   @property
   def allcoords(self):
      crd = [self.coords]
      crd = crd + [c.coords for c in self.children]
      return np.stack(crd)

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
         count = wu.cpp.bvh.bvh_count_pairs(self.contactbvh, other.contactbvh, self.position, other.position,
                                            contactdist)
      else:
         assert 0
         # import scipy.spatial
         # d = scipy.spatial.distance_matrix(self.coords, other.coords)
         d = wu.hnorm(self.coords[None] - other.coords[:, None])
         count = np.sum(d < contactdist)
      return count

   def contacts(self, other):
      return self.contact_count(other, self.contactdis)

   def clashes(self, other, clashdis=None):
      # ic(self.clashdis)
      clashdis = clashdis or self.clashdis
      return self.contact_count(other, self.clashdis)

   def point_contact_count(self, other, contactdist=8):
      p = self.interactions(other, contactdist=contactdist)
      a = set(p[:, 0])
      b = set(p[:, 1])
      # ic(a)
      # ic(b)
      return len(a), len(b)

   def contact_fraction(self, other, contactdist=None):
      contactdist = contactdist or self.contactdis

      # wu.pdb.dump_pdb_from_points('bodyA.pdb', self.coords)
      # wu.pdb.dump_pdb_from_points('bodyB.pdb', other.coords)

      p = self.interactions(other, contactdist=contactdist)
      a = set(p[:, 0])
      b = set(p[:, 1])
      # ic(len(a), len(self.coords))
      # ic(len(b), len(other.coords))

      return len(a) / len(self.coords), len(b) / len(self.coords)

   def clash_distances(self, other, maxdis=8):
      crd1 = self.coords
      crd2 = other.coords
      interactions = self.clash_interactions(other, maxdis)
      crd1 = crd1[interactions[:, 0]]
      crd2 = crd2[interactions[:, 1]]
      return wu.hnorm(crd1 - crd2)

   def interactions(self, other, contactdist=8, buf=None, usebvh=None):
      assert isinstance(other, RigidBody)
      if usebvh or (usebvh is None and self.usebvh):
         if not buf: buf = np.empty((100000, 2), dtype="i4")
         pairs, overflow = wu.cpp.bvh.bvh_collect_pairs(self.contactbvh, other.contactbvh, self.position,
                                                        other.position, contactdist, buf)
         assert not overflow
      else:
         d = wu.hnorm(self.contact_coords[None] - other.contact_coords[:, None])
         pairs = np.stack(np.where(d <= contactdist), axis=1)
      return pairs

   def clash_interactions(self, other, contactdist=8, buf=None, usebvh=None):
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

   def dumppdb(self, fname, dumpchildren=False, spacegroup=None, **kw):
      if dumpchildren:
         crd = self.allcoords
         wu.pdb.dumppdb(fname, crd, nchain=len(self.children) + 1, **kw)
      elif spacegroup is not None:
         wu.pdb.dumppdb(fname, self.coords, spacegroup=spacegroup, cellsize=self.scale(), **kw)
      else:
         wu.pdb.dumppdb(fname, self.coords, **kw)
