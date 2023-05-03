import itertools
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

   ncell = 1
   # if len(unitframes) < 4: ncell = 2
   # if len(unitframes) < 8: ncell = 2
   frames = latticeframes(unitframes, lattice, cells=ncell)

   # relframes = einsum('aij,bjk->abik', frames, wu.)
   axs, ang, cen, hel = wu.homog.axis_angle_cen_hel_of(frames)

   axs, cen = axs[:, :3], cen[:, :3]

   flip = np.sum(axs * [3, 2, 1], axis=1) > 0
   axs = np.where(np.stack([flip, flip, flip], axis=1), axs, -axs)
   tag = np.concatenate([axs, cen], axis=1).round(10)
   symelems = dict()
   for nfold in [2, 3, 4, 6]:
      nfang = 2 * np.pi / nfold

      idx = np.where(np.logical_and(
         np.isclose(ang, nfang, atol=1e-6),
         np.isclose(0, hel),
      ))[0]
      if np.sum(idx) == 0: continue
      nftag = tag[idx]
      nftag = nftag[np.lexsort(-nftag.T, axis=0)]
      nftag = np.unique(nftag, axis=0)

      # nftag = nftag[np.argsort(-nftag[:, 2], kind='stable')]
      # nftag = nftag[np.argsort(-nftag[:, 1], kind='stable')]
      # nftag = nftag[np.argsort(-nftag[:, 0], kind='stable')]
      d = np.sum(nftag[:, 3:]**2, axis=1).round(6)
      nftag = nftag[np.argsort(d, kind='stable')]
      # if spacegroup == 'I213' and nfold == 3:
      # ic(nftag)

      symelems[f'c{nfold}'] = [SymElem(nfold, t[:3], t[3:6]) for t in nftag]

   #from willutil.viz import showme
   #frames[:, :3, 3] *= 10
   #showme(frames)
   ## showme(frames @ wu.homog.htrans([1, 2, 3]))
   #showme(symelems[2], scale=10)
   #showme(symelems[3], scale=10)
   #ic(spacegroup, symelems)

   return symelems

# def compute_asucover(spacegroup, lattice, frames, nsamp=11):
#    lattice = lattice_vectors(lattice)
#    lframes = latticeframes(frames, lattice, cells=3)
#    # ic(frames.shape, lframes.shape)
#
#    assert lframes
#
#    cover = np.array([[0, 0, 0, 1], [1, 1, 1, 1]])
#    com = cover.mean(0)
#
#    samp = np.linspace(0, 1, nsamp)
#    xyz = np.meshgrid(samp, samp, samp)
#    xyz = np.stack(xyz, axis=3).reshape(-1, 3)
#    xyz = wu.homog.hpoint(xyz)
#
#    fgrid = einsum('fij,pj->pfi', lframes, xyz)
#
#    wu.homog.hnorm(fgrid[])
#
#    ic(spacegroup, fgrid.shape)
#    from willutil.viz import showme
#
#
#
#    # fgrid[..., :3] *= 100
#    # showme(fgrid.reshape(-1, 4))
#
#    assert 0
#    return com, cover

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
