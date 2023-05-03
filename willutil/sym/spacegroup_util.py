import itertools, collections
from opt_einsum import contract as einsum
import numpy as np
import willutil as wu
from willutil.sym.spacegroup_data import *
from willutil.sym.SymElem import *

def applylattice(lattice, unitframes):
   origshape = unitframes.shape
   if unitframes.ndim == 3: unitframes = unitframes.reshape(1, -1, 4, 4)
   if lattice.ndim == 2: lattice = lattice.reshape(1, 3, 3)
   latticeframes = unitframes.copy()
   latticeframes[:, :, :3, 3] = einsum('nij,nkj->nki', lattice, latticeframes[:, :, :3, 3])
   # latticeframes[:, :, :3, :3] = einsum('nij,nfjk->nfik', lattice, latticeframes[:, :, :3, :3])

   return latticeframes.reshape(origshape)

def latticeframes(unitframes, lattice, cells=1):
   latticeframes = applylattice(lattice, unitframes)
   cells = process_num_cells(cells)
   xshift = wu.homog.htrans(cells @ lattice)
   frames = wu.homog.hxformx(xshift, latticeframes, flat=True, improper_ok=True)
   return frames.round(10)

def tounitframes(frames, lattice, spacegroup=None):
   if not hasattr(lattice, 'shape') or lattice.shape != (3, 3):
      lattice = lattice_vectors(spacegroup, lattice)
   uframes = frames.copy()
   # cells = process_num_cells(cells)
   # xshift = wu.hinv(wu.htrans(cells @ lattice))
   uframes[:, :3, 3] = einsum('ij,jk->ik', uframes[:, :3, 3], np.linalg.inv(lattice))
   # uframes = wu.hxform(xshift, uframes, flat=True)

   return uframes.round(10)

def process_num_cells(cells):
   if cells is None:
      return np.eye(4)[None]
   if isinstance(cells, np.ndarray) and cells.ndim == 2 and cells.shape[1] == 3:
      return cells
   if isinstance(cells, (int, float)):
      ub = cells // 2
      lb = ub - cells + 1
      cells = [(a, b, c) for a, b, c in itertools.product(*[range(lb, ub + 1)] * 3)]
   elif len(cells) == 2:
      lb, ub = cells
      cells = [(a, b, c) for a, b, c in itertools.product(*[range(lb, ub + 1)] * 3)]
   elif len(cells) == 3:
      if isinstance(cells[0], int):
         cells = [(0, cells[0] - 1), (0, cells[1] - 1), (0, cells[2] - 1)]
      cells = [(a, b, c) for a, b, c in itertools.product(
         range(cells[0][0], cells[0][1] + 1),
         range(cells[1][0], cells[1][1] + 1),
         range(cells[2][0], cells[2][1] + 1),
      )]
   else:
      raise ValueError(f'bad cells {cells}')
   cells = np.array(cells)

   # order in stages, cell 0 first, cell 0 to 1, cells -1 to 1, cells -1 to 2, etc
   blocked = list()
   mn, mx = np.min(cells, axis=1), np.max(cells, axis=1)
   lb, ub = 0, 0
   prevok = np.zeros(len(cells), dtype=bool)
   for i in [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8]:
      lb, ub = min(i, lb), max(i, ub)
      ok = np.logical_and(lb <= mn, mx <= ub)
      c = cells[np.logical_and(ok, ~prevok)]
      blocked.append(c)
      prevok |= ok
      if np.all(prevok): break
   cells = np.concatenate(blocked)

   return cells

def compute_symelems(spacegroup, unitframes):
   lattice = np.eye(3)

   # if len(unitframes) < 4: ncell = 2
   # if len(unitframes) < 8: ncell = 2
   unitframes = latticeframes(unitframes, lattice, cells=2)
   frames = latticeframes(unitframes, lattice, cells=5)
   # for f in frames:
   # if np.allclose(f[:3, :3], np.eye(3)) and f[0, 3] == 0 and f[2, 3] == 0:
   # print(f)

   # relframes = einsum('aij,bjk->abik', unitframes, wu.)
   axs, ang, cen, hel = wu.homog.axis_angle_cen_hel_of(unitframes)
   axs, cen, hel = axs[:, :3], cen[:, :3], hel[:, None]

   flip = np.sum(axs * [3, 2, 1], axis=1) > 0
   axs = np.where(np.stack([flip, flip, flip], axis=1), axs, -axs)
   tag0 = np.concatenate([axs, cen, hel], axis=1).round(10)
   symelems = collections.defaultdict(list)
   for nfold in [2, 3, 4, 6, -2, -3, -4, -6]:
      screw, nfold = nfold < 0, abs(nfold)
      nfang = 2 * np.pi / nfold

      # idx = np.isclose(ang, nfang, atol=1e-6)
      if screw:
         idx = np.logical_and(np.isclose(ang, nfang, atol=1e-6), ~np.isclose(0, hel[:, 0]))
      else:
         idx = np.logical_and(np.isclose(ang, nfang, atol=1e-6), np.isclose(0, hel[:, 0]))
      if np.sum(idx) == 0: continue
      nftag = tag0[idx]
      nftag = nftag[np.lexsort(-nftag.T, axis=0)]
      nftag = np.unique(nftag, axis=0)

      nftag = nftag[np.argsort(-nftag[:, 5], kind='stable')]
      nftag = nftag[np.argsort(-nftag[:, 4], kind='stable')]
      nftag = nftag[np.argsort(-nftag[:, 3], kind='stable')]
      d = np.sum(nftag[:, 3:6]**2, axis=1).round(6)
      nftag = nftag[np.argsort(d, kind='stable')]

      # remove symmetric dups
      keep = nftag[:1]
      for tag in nftag[1:]:
         tax, tcen, thel = tag[:3], tag[3:6], tag[6]
         if np.any(np.isclose(thel, [0.5, np.sqrt(2) / 2])):
            # if is 21, 42, 63 screw, allow reverse axis with same helical shift
            symtags = np.concatenate([
               np.concatenate([(frames @ wu.hvec(+tax))[:, :3], (frames @ wu.hpoint(tcen))[:, :3], np.tile(+thel, [len(frames), 1])], axis=1),
               np.concatenate([(frames @ wu.hvec(-tax))[:, :3], (frames @ wu.hpoint(tcen))[:, :3], np.tile(-thel, [len(frames), 1])], axis=1),
               np.concatenate([(frames @ wu.hvec(-tax))[:, :3], (frames @ wu.hpoint(tcen))[:, :3], np.tile(+thel, [len(frames), 1])], axis=1),
            ])
         else:
            symtags = np.concatenate([
               np.concatenate([(frames @ wu.hvec(+tax))[:, :3], (frames @ wu.hpoint(tcen))[:, :3], np.tile(+thel, [len(frames), 1])], axis=1),
               np.concatenate([(frames @ wu.hvec(-tax))[:, :3], (frames @ wu.hpoint(tcen))[:, :3], np.tile(-thel, [len(frames), 1])], axis=1),
            ])
         seenit = np.all(np.isclose(keep[None], symtags[:, None], atol=0.001), axis=2)
         if not np.any(seenit):
            picktag = _pick_symelemtags(symtags, symelems)
            # picktag = None
            if picktag is None:
               keep = np.concatenate([keep, tag[None]])
            else:
               keep = np.concatenate([keep, picktag[None]])

      # ic((keep * 1000).astype('i'))
      for tag in keep:
         try:
            se = SymElem(nfold, tag[:3], tag[3:6], hel=tag[6])
            symelems[se.label].append(se)
         except wu.sym.ScrewError:
            continue
      # ic(screw, nfold, symelems)

      symelems = _symelem_remove_redundant_syms(symelems)
   return symelems

def _pick_symelemtags(symtags, symelems):

   # assert 0, 'this is incorrect somehow'

   # for i in [0, 1, 2]:
   #    symtags = symtags[np.argsort(-symtags[:, i], kind='stable')]
   # for i in [6, 5, 4, 3]:
   #    symtags = symtags[symtags[:, i] > -0.0001]
   #    symtags = symtags[symtags[:, i] < +0.9999]
   # for i in [5, 4, 3]:
   #    symtags = symtags[np.argsort(symtags[:, i], kind='stable')]
   # if len(symtags) == 0: return None
   # # ic(symtags)

   cen = [se.cen[:3] for psym in symelems for se in symelems[psym]]
   if cen and len(symtags):
      w = np.where(np.all(np.isclose(symtags[:, None, 3:6], np.stack(cen)[None]), axis=2))[0]
      if len(w) > 0:
         return symtags[w[0]]
   return None

   # d = np.sum(symtags[:, 3:6]**2, axis=1).round(6)
   # symtags = symtags[np.argsort(d, kind='stable')]
   # # ic(symtags[0])
   # return symtags[0]

def _symelem_remove_redundant_syms(symelems):
   symelems = symelems.copy()
   for sym1, sym2 in [('C2', 'C4'), ('C3', 'C6')]:
      if sym2 in symelems:
         newc2 = list()
         for s2 in symelems[sym1]:
            for s in symelems[sym2]:
               if np.allclose(s.axis, s2.axis) and np.allclose(s.cen, s2.cen): break
            else:
               newc2.append(s2)
         symelems[sym1] = newc2
   return symelems

def full_cellgeom(lattice, cellgeom, strict=True):
   if isinstance(cellgeom, (int, float)):
      cellgeom = [cellgeom]
   if isinstance(lattice, str) and lattice in sg_lattice:
      lattice = sg_lattice[lattice]

   # assert lattice in 'TETRAGONAL CUBIC'.split()
   assert isinstance(cellgeom, (np.ndarray, list, tuple))
   p = np.array(cellgeom)
   if lattice == 'TRICLINIC':
      p = [p[0], p[1], p[2], p[3], p[4], p[5]]
   elif lattice == 'MONOCLINIC':
      if strict:
         assert np.allclose(p[3], 90.0), f'invalid cell geometry {p}'
         assert len(p) < 6 or np.allclose(p[5], 90.0), f'invalid cell geometry {p}'
      p = [p[0], p[1], p[2], 90.0, p[4], 90.0]
   elif lattice == 'CUBIC':
      if strict:
         assert len(p) < 4 or np.allclose(p[3], 90.0), f'invalid cell geometry {p}'
         assert len(p) < 5 or np.allclose(p[4], 90.0), f'invalid cell geometry {p}'
         assert len(p) < 6 or np.allclose(p[5], 90.0), f'invalid cell geometry {p}'
         assert np.allclose(p[0], p[:3])
      p = [p[0], p[0], p[0], 90.0, 90.0, 90.0]
   elif lattice == 'ORTHORHOMBIC':
      if strict:
         assert len(p) < 4 or np.allclose(p[3], 90.0), f'invalid cell geometry {p}'
         assert len(p) < 5 or np.allclose(p[4], 90.0), f'invalid cell geometry {p}'
         assert len(p) < 6 or np.allclose(p[5], 90.0), f'invalid cell geometry {p}'
      p = [p[0], p[1], p[2], 90.0, 90.0, 90.0]
   elif lattice == 'TETRAGONAL':
      if strict:
         assert np.allclose(p[0], p[1]), f'invalid cell geometry {p}'
         assert len(p) < 4 or np.allclose(p[3], 90.0), f'invalid cell geometry {p}'
         assert len(p) < 5 or np.allclose(p[4], 90.0), f'invalid cell geometry {p}'
         assert len(p) < 6 or np.allclose(p[5], 90.0), f'invalid cell geometry {p}'
         assert np.allclose(p[0], p[1])
      p = [p[0], p[0], p[2], 90.0, 90.0, 90.0]
   elif lattice == 'HEXAGONAL':
      if strict:
         assert np.allclose(p[0], p[1]), f'invalid cell geometry {p}'
         assert len(p) < 4 or np.allclose(p[3], 90.0), f'invalid cell geometry {p}'
         assert len(p) < 5 or np.allclose(p[4], 90.0), f'invalid cell geometry {p}'
         assert len(p) < 6 or np.allclose(p[5], 120.0), f'invalid cell geometry {p}'
         assert np.allclose(p[0], p[1])
      p = [p[0], p[0], p[2], 90.0, 90.0, 120.0]
   return p

def lattice_vectors(lattice, cellgeom=None):
   if lattice in sg_lattice:
      lattice = sg_lattice[lattice]
   if cellgeom is None:
      cellgeom = [1.0, 1.0, 1.0, 90.0, 90.0, 90.0]
      if lattice == 'HEXAGONAL':
         cellgeom = [1.0, 1.0, 1.0, 90.0, 90.0, 120.0]

   a, b, c, A, B, C = full_cellgeom(lattice, cellgeom)
   cosA, cosB, cosC = [np.cos(np.radians(_)) for _ in (A, B, C)]
   sinB, sinC = [np.sin(np.radians(_)) for _ in (B, C)]

   # ic(cosB * cosC - cosA)
   # ic(sinB, sinC)
   # ic(1.0 - ((cosB * cosC - cosA) / (sinB * sinC))**2)

   lattice_vectors = np.array([[
      a,
      b * cosC,
      c * cosB,
   ], [
      0.0,
      b * sinC,
      c * (cosA - cosB * cosC) / sinC,
   ], [
      0.0,
      0.0,
      c * sinB * np.sqrt(1.0 - ((cosB * cosC - cosA) / (sinB * sinC))**2),
   ]]).T
   return lattice_vectors

def cellvol(spacegroup, cellgeom):
   a, b, c, A, B, C = cellgeom(spacegroup, cellgeom, radians=True)
   cosA, cosB, cosC = [np.cos(_) for _ in (A, B, C)]
   sinB, sinC = [np.sin(_) for _ in (B, C)]
   return a * b * c * np.sqrt(1 - cosA**2 - cosB**2 - cosC**2 + 2 * cosA * cosB * cosC)
