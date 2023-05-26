import copy
import numpy as np
import willutil as wu
from willutil.sym.symframes import tetrahedral_frames, octahedral_frames
from willutil.homog.hgeom import halign2, halign, htrans, hinv, hnorm, hxform, hnormalized, hpoint, hvec, angle
from willutil.sym.spacegroup_util import tounitcellpts, applylatticepts, lattice_vectors

class ScrewError(Exception):
   pass

class ComponentIDError(Exception):
   pass

class SymElem:
   def __init__(
         self,
         nfold,
         axis,
         cen=[0, 0, 0],
         axis2=None,
         *,
         label=None,
         vizcol=None,
         scale=1,
         parent=None,
         children=None,
         hel=0,
         lattice=np.eye(3),
         screw=None,
         adjust_cyclic_center=True,
   ):
      self._init_args = wu.Bunch(vars()).without('self')
      self.vizcol = vizcol
      self.scale = scale

      self._set_geometry(nfold, axis, cen, axis2, hel, lattice, adjust_cyclic_center)
      self._check_screw(screw)
      self._make_label(label)
      self._set_kind()

      self.mobile = False
      if wu.homog.hgeom.h_point_line_dist([0, 0, 0], cen, axis) > 0.0001: self.mobile = True
      if axis2 is not None and wu.homog.hgeom.h_point_line_dist([0, 0, 0], cen, axis2) > 0.0001: self.mobile = True
      self.operators = self.make_operators()
      self.numops = len(self.operators)
      self.parent = parent
      self.children = children or list()

      self.numeric_cleanup()
      self.issues = []

   def numeric_cleanup(self):
      self.axis = self.axis.round(9)
      self.cen = self.cen.round(9)
      if self.axis2 is not None: self.axis2 = self.axis2.round(9)
      self.hel = self.hel.round(9)
      self.axis[self.axis == -0] = 0
      self.cen[self.cen == -0] = 0
      # self.index = None
      if self.axis2 is not None:
         self.axis2 = wu.homog.hnormalized(self.axis2)
         self.axis2[self.axis2 == -0] = 0
      if not self.isscrew:
         if angle(self.axis, [1, 1.1, 1.2]) > np.pi / 2:
            self.axis = -self.axis
         if self.axis2 is not None and angle(self.axis2, [1, 1.1, 1.2]) > np.pi / 2:
            self.axis2 = -self.axis2

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

   def frame_operator_ids(self, frames, sanitycheck=True):
      # ic(self)
      # from willutil.viz.pymol_viz import showme
      # showme(10 * einsum('fij,j->fi', frames, [0, 0, 0, 1]), scale=10)
      # showme(10 * einsum('fij,j->fi', frames, [0.5, 0.5, 0.5, 1]), scale=10)
      # showme(10 * einsum('fij,j->fi', frames, [1, 1, 1, 1]), scale=10)
      # assert 0
      opids = np.arange(len(frames), dtype=np.int32)
      opsframes = einsum('oij,fjk->ofik', self.operators[1:], frames)
      for iop, opframes in enumerate(opsframes):
         a, b = np.where(np.all(np.isclose(frames[None], opframes[:, None]), axis=(-1, -2)))
         opids[a] = np.minimum(opids[a], opids[b])
      for i, id in enumerate(sorted(set(opids))):
         opids[opids == id] = i
      if sanitycheck and self.iscompound:
         for i in range(np.max(opids)):
            ids = (opids == i)
            # showme(frames[ids])
            if np.sum(ids) == len(self.operators):
               assert np.allclose(self.cen, frames[ids, :, 3].mean(axis=0))
      return opids

   def frame_component_ids(self, frames, permutations, sym=None, sanitycheck=True):
      if self.iscompound:
         # compound elements (D2, T, etc) will never overlap their centers
         return self.frame_component_ids_bycenter(frames, sanitycheck)
      assert len(permutations) == len(frames)
      opframes = np.eye(4).reshape(1, 4, 4)
      iframematch0 = self.matching_frames(frames)
      # ic(iframematch0, len(frames), permutations.shape)
      fid = 0
      compid = -np.ones(len(frames), dtype=np.int32)
      for iframe, perm in enumerate(permutations):
         try:
            iframematch = perm[iframematch0]
         except IndexError:
            # from willutil.viz.pymol_viz import showme
            # import willutil.viz.viz_xtal
            # showme(self, scale=10)
            # showme(frames, scale=10)
            # assert 0
            raise ComponentIDError
         iframematch = iframematch[iframematch >= 0]
         centest = einsum('fij,j->fi', frames[iframematch], self.cen)
         axstest = einsum('fij,j->fi', frames[iframematch], self.axis)
         if sanitycheck and self.iscompound:
            assert np.allclose(centest, centest[0])
            if not (self.istet or self.isoct):
               assert np.all(np.logical_or(
                  np.all(np.isclose(axstest, axstest[0]), axis=1),
                  np.all(np.isclose(axstest, -axstest[0]), axis=1),
               ))
         if np.allclose(compid[iframematch], -1):
            compid[iframematch] = fid
            fid += 1
         else:
            assert min(compid[iframematch]) == max(compid[iframematch])

      if sanitycheck and not self.iscyclic:
         _sanitycheck_compid_cens(self, frames, compid)

      return compid

   def frame_component_ids_bycenter(self, frames, sanitycheck=True):
      assert self.iscompound
      cen = einsum('fij,j->fi', frames, self.cen)
      d = hnorm(cen[:, None] - cen[None])
      compid = np.ones(len(frames), dtype=np.int32) * -12345
      count = 0
      for i in range(len(d)):
         if compid[i] >= 0: continue
         w = np.where(np.isclose(d[i], 0))
         assert len(w) <= len(self.operators)
         compid[w] = count
         count += 1
      if sanitycheck:
         _sanitycheck_compid_cens(self, frames, compid)
      # w = np.isclose(d, 0)[12]
      # wu.showme(cen[w], scale=10)
      # wu.showme(frames[w], scale=10)
      # assert 0
      # ic(count)
      # s = set(compid)
      # for i in range(np.max(compid)):
      # if not i in s:
      # ic(i)
      # assert len(set(compid)) == np.max(compid) + 1
      # assert 0
      return compid

   def tolattice(self, latticevec):
      newcen = applylatticepts(latticevec, self.cen)
      newhel = applylatticepts(latticevec, self.cen + self.axis * self.hel)
      newhel = hnorm(newhel - newcen)
      newelem = SymElem(self._init_args.nfold, self.axis, newcen, self.axis2, hel=newhel, screw=self.screw)
      assert self.operators.shape == newelem.operators.shape
      return newelem

   def tounit(self, latticevec):
      return self.tolattice(np.linalg.inv(latticevec))

   def matching_frames(self, frames):
      'find frames related by self.operators that are closest to cen'
      match = np.isclose(frames[None], self.operators[:, None])
      match = np.any(np.all(match, axis=(2, 3)), axis=0)
      match = np.where(match)[0]
      if len(match) != len(self.operators):
         raise ComponentIDError
         ic(frames.shape)
         ic(match)
         from willutil.viz.pymol_viz import showme
         import willutil.viz.viz_xtal
         showme(frames, scale=10)
         showme(self, scale=10)
         assert len(match) == len(self.operators)
      return match

      # symaxs = einsum('fij,j->fi', frames, self.axis)
      # symcen = einsum('fij,j->fi', frames, self.cen)
      # match = np.logical_and(
      # np.all(np.isclose(self.axis, symaxs), axis=1),
      # np.all(np.isclose(self.cen, symcen), axis=1),
      # )
      # w = np.where(match)[0]
      # return w

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

   def _check_screw(self, screw):
      if screw is not None:
         self.screw = screw
         return
      if self.hel == 0.0:
         self.screw = 0
         return
      assert not self.axis2

      self.screw = np.abs(self.axis) / self.hel
      self.screw = 1 / self.screw[np.argmax(np.abs(self.screw))]
      self.screw = self.nfold * self.screw

      # ic(self.axis.round(3), self.hel)
      if not all([
            np.allclose(round(self.screw), self.screw),
            self.screw <= self.nfold + 0.00001,
            self.screw >= -self.nfold - 0.00001,
            self.nfold == 1 or self.screw < self.nfold - 0.00001,
      ]):
         # ic('ScrewError')
         # ic(self.nfold, self.axis, self.hel, self.screw)
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

   def _set_geometry(self, nfold, axis, cen, axis2, hel, lattice, adjust_cyclic_center):
      axis = hpoint(axis)
      cen = hpoint(cen)
      self._set_nfold(nfold)
      self.angle = np.pi * 2 / self.nfold

      invlattice = np.linalg.inv(lattice)
      self.axis = hnormalized(_mul3(invlattice, axis))
      self.cen = hpoint(_mul3(invlattice, hpoint(cen)))
      self.axis2 = None if axis2 is None else hnormalized(_mul3(invlattice, hvec(axis2)))
      heltrans = _mul3(invlattice, cen + hnormalized(axis) * hel) - _mul3(invlattice, cen)
      self.hel = hnorm(heltrans)

      # if not np.isclose(hel, 0) and nfold > 1:
      # if np.allclose(self.axis, [0, 1, 0, 0]):
      # ic(lattice)
      # ic(self._init_args.hel)
      # ic(hel, self.hel)
      # assert 0

      if adjust_cyclic_center and axis2 is not None and np.isclose(hel, 0):  # cyclic
         dist = wu.homog.line_line_distance_pa(self.cen, self.axis, _cube_edge_cen * self.scale, _cube_edge_axis)
         w = np.argmin(dist)
         newcen, _ = wu.homog.line_line_closest_points_pa(self.cen, self.axis, _cube_edge_cen[w] * self.scale, _cube_edge_axis[w])
         # ic(cen, newcen)
         if not np.any(np.isnan(newcen)):
            self.cen = newcen
            # ic(newcen)

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
         ops = tetrahedral_frames
         assert self._opinfo == (3, 2)
         aln = halign2([1, 1, 1], [0, 0, 1], self.axis, self.axis2)
         ops = aln @ ops @ hinv(aln)
         ops = htrans(self.cen) @ ops @ htrans(-self.cen)
         # ic(self.axis, self.axis2)
      elif self.label == 'O':
         assert self._opinfo in [(4, 3), (4, 2), (3, 2)]
         ops = octahedral_frames
         ops = htrans(self.cen) @ ops @ htrans(-self.cen)
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

      if not self.isdihedral:
         self.origin = htrans(self.cen) @ halign([0, 0, 1], self.axis)
      else:
         # ic(self.axis)
         # ic(self.axis2)
         self.origin = htrans(self.cen) @ halign2([0, 0, 1], [1, 0, 0], self.axis, self.axis2)

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
         axis = hxform(x, self.axis)
         axis2 = None if self.axis2 is None else hxform(x, self.axis2)
         cen = hxform(x, self.cen)
         other = SymElem(self._init_args.nfold, axis, cen, axis2, label=self.label, vizcol=self.vizcol, scale=1, screw=self.screw)
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
         s = f'SymElem({self.label}, axis={list(ax[:3])}, axis2={list(ax2[:3])}, cen={list(self.cen[:3])}, label=\'{self.label}\')'
      elif self.isoct:
         ax2 = (self.axis2 / np.max(np.abs(self.axis2))).round(6)
         s = f'SymElem({self.label}, axis={list(ax[:3])}, axis2={list(ax2[:3])}, cen={list(self.cen[:3])}, label=\'{self.label}\')'
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

def showsymelems(
   sym,
   symelems,
   allframes=True,
   colorbyelem=False,
   cells=3,
   bounds=[-0.1, 1.1],
   scale=12,
   offset=0,
   weight=2.0,
   scan=0,
   lattice=None,
   # onlyz=False,
   # showframes=False,
):
   if isinstance(symelems, list):
      tmp = defaultdict(list)
      for e in symelems:
         tmp[e.label].append(e)
      symelems = tmp

   import pymol
   f = np.eye(4).reshape(1, 4, 4)
   if lattice is None:
      lattice = lattice_vectors(sym, cellgeom='nonsingular')
   if allframes:
      cellgeom = wu.sym.cellgeom_from_lattice(lattice)
      f = wu.sym.sgframes(sym, cells=cells, cellgeom=cellgeom)
   f = wu.hscaled(scale, f)

   ii = 0
   labelcount = defaultdict(lambda: 0)
   for i, c in enumerate(symelems):
      for j, sunit in enumerate(symelems[c]):
         s = sunit.tolattice(lattice)
         # if colorbyelem: args.colors = [[(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)][ii]]
         f2 = f
         if scan and not s.iscompound:
            f2 = f[:, None] @ wu.htrans(s.axis[None] * np.linspace(-scale * np.sqrt(3), scale * np.sqrt(3), scan)[:, None])[None]
            ic(f2.shape)
            f2 = f2.reshape(-1, 4, 4)
            ic(f2.shape)

         # shift = wu.htrans(s.cen * scale + offset * wu.hvec([0.1, 0.2, 0.3]))
         shift = wu.htrans(s.cen * scale)
         # shift = np.eye(4)

         if s.istet:
            configs = [
               ((s.axis, [0, 1, 0]), (None, None), [0.0, 0.8, 0.0]),
               ((-s.axis, [0, 1, 0]), (None, None), [0.0, 0.8, 0.0]),
               ((s.axis2, [1, 0, 0]), (None, None), [0.8, 0.0, 0.0]),
            ]
         elif s.isoct:
            configs = [
               (([0, 1, 1], [1, 0, 0]), (None, None), [0.7, 0.0, 0.0]),
               (([1, 1, 1], [0, 1, 0]), (None, None), [0.0, 0.7, 0.0]),
               (([0, 0, 1], [0, 0, 1]), (None, None), [0.0, 0.0, 0.7]),
            ]
         elif s.label == 'D2':
            configs = [
               ((s.axis, [1, 0, 0]), (s.axis2, [0, 1, 0]), [0.7, 0, 0]),
               ((s.axis, [0, 1, 0]), (s.axis2, [0, 0, 1]), [0.7, 0, 0]),
               ((s.axis, [0, 0, 1]), (s.axis2, [1, 0, 0]), [0.7, 0, 0]),
            ]
         elif s.label == 'D4':
            configs = [
               ((s.axis2, [1, 0, 0]), (s.axis, [0, 1, 0]), [0.7, 0, 0]),
               ((wu.hrot(s.axis, 45, s.cen) @ s.axis2, [1, 0, 0]), (s.axis, [0, 1, 0]), [0.7, 0, 0]),
               ((s.axis, [0, 0, 1]), (s.axis2, [1, 0, 0]), [0.0, 0, 0.9]),
            ]
         elif s.nfold == 2:
            configs = [((s.axis, [1, 0, 0]), (s.axis2, [0, 0, 1]), [1.0, 0.3, 0.6])]
         elif s.nfold == 3:
            configs = [((s.axis, [0, 1, 0]), (s.axis2, [1, 0, 0]), [0.6, 1, 0.3])]
         elif s.nfold == 4:
            configs = [((s.axis, [0, 0, 1]), (s.axis2, [1, 0, 0]), [0.6, 0.3, 1])]
         elif s.nfold == 6:
            configs = [((s.axis, [1, 1, 1]), (s.axis2, [-1, 1, 0]), [1, 1, 1])]
         else:
            assert 0
         name = s.label + '_' + ('ABCDEFGH')[labelcount[s.label]]

         cgo = list()
         for (tax, ax), (tax2, ax2), xyzlen in configs:
            onlyz = scan and not s.iscompound
            xyzlen = np.array(xyzlen)
            if onlyz: xyzlen[xyzlen < .999] = 0
            if s.isdihedral:
               origin = wu.halign2(ax, ax2, tax, tax2)
               xyzlen[xyzlen == 0.6] = 1
            else:
               origin = wu.halign(ax, tax)
            wu.showme(
               f2 @ shift @ origin,
               name=name,
               bounds=[b * scale for b in bounds],
               xyzlen=xyzlen,
               addtocgo=cgo,
               make_cgo_only=True,
               weight=weight,
               colorset=labelcount[s.label],
            )
         pymol.cmd.load_cgo(cgo, name)
         labelcount[s.label] += 1
         ii += 1
   from willutil.viz.pymol_viz import showcell, showcube
   showcell(scale * lattice)
   # showcube()

def _sanitycheck_compid_cens(elem, frames, compid):
   seenit = list()
   for i in range(np.max(compid)):
      assert np.sum(compid == i) > 0
      compframes = frames[compid == i]
      # wu.showme(compframes @ elem.origin @ offset, scale=scale)
      cen = einsum('ij,j->i', compframes[0], elem.origin[:, 3])
      assert np.allclose(cen, einsum('fij,j->fi', compframes, elem.origin[:, 3]))
      assert not any([np.allclose(cen, s) for s in seenit])
      seenit.append(cen)

def _make_operator_component_joint_ids(elem1, elem2, frames, fopid, fcompid, sanitycheck=True):
   from willutil.viz.pymol_viz import showme
   opcompid = fcompid.copy()
   for i in range(np.max(fopid)):
      fcids = fcompid[fopid == i]
      idx0 = fcompid == fcids[0]
      for fcid in fcids[1:]:
         idx = fcompid == fcid
         opcompid[idx] = min(min(opcompid[idx]), min(opcompid[idx0]))
   for i, id in enumerate(sorted(set(opcompid))):
      opcompid[opcompid == id] = i

   if sanitycheck and elem2.iscompound:
      seenit = np.empty((0, 4))
      for i in range(np.max(opcompid)):
         compframes = frames[opcompid == i]
         cens = einsum('fij,j->fi', compframes, elem2.origin[:, 3])
         if np.any(np.all(np.isclose(cens[None], seenit[:, None]), axis=2)):
            ic(elem1)
            ic(elem2)
            # for i in range(np.max(fopid)):
            showme(elem1.cen, scale=10, name='ref')
            showme(elem1.operators, scale=10, name='ref1')
            for i in range(100):
               showme(frames[opcompid == i] @ elem2.origin @ htrans([0.01, 0.02, 0.03]), scale=10)
            assert 0

            showme(cens, scale=10)
            showme(compframes, scale=10)
            showme(seenit, scale=10)

            assert 0
         assert not np.any(np.all(np.isclose(cens[None], seenit[:, None]), axis=2))
         seenit = np.concatenate([cens, seenit])

   return opcompid

def _mul3(a, b):
   return (a @ b[:3, None])[:, 0]
