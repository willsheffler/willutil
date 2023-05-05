import itertools, collections
from opt_einsum import contract as einsum
import numpy as np
import willutil as wu
from willutil.sym.spacegroup_data import *
from willutil.sym.spacegroup_util import *
from willutil.sym.SymElem import *
from willutil.homog.hgeom import h_point_line_dist

def compute_symelems(spacegroup, unitframes, torch_device=None):
   if torch_device:
      try:
         import torch
      except ImportError:
         torch_device = False

   lattice = np.eye(3)

   # if len(unitframes) < 4: ncell = 2
   # if len(unitframes) < 8: ncell = 2
   # unitframes = unitframes.astype(np.float32)
   f2cel = latticeframes(unitframes, lattice, cells=2)
   f5cel = latticeframes(unitframes, lattice, cells=5)
   # for f in f5cel:
   # if np.allclose(f[:3, :3], np.eye(3)) and f[0, 3] == 0 and f[2, 3] == 0:
   # print(f)

   # relframes = einsum('aij,bjk->abik', f2cel, wu.)
   f2geom = wu.homog.axis_angle_cen_hel_of(f2cel)
   axs, ang, cen, hel = f2geom
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

      nftag = nftag[np.argsort(-nftag[:, 0], kind='stable')]
      nftag = nftag[np.argsort(-nftag[:, 1], kind='stable')]
      nftag = nftag[np.argsort(-nftag[:, 2], kind='stable')]
      nftag = nftag[np.argsort(-nftag[:, 5], kind='stable')]
      nftag = nftag[np.argsort(-nftag[:, 4], kind='stable')]
      nftag = nftag[np.argsort(-nftag[:, 3], kind='stable')]

      d = np.sum(nftag[:, 3:6]**2, axis=1).round(6)
      nftag = nftag[np.argsort(d, kind='stable')]

      # if nfold == 3:
      # print(nftag[:, :6])
      # assert 0

      if torch_device:
         f5cell_torch = torch.tensor(f5cel, device=torch_device, dtype=torch.float32)

      # remove symmetric dups
      keep = nftag[:1]
      for itag, tag in enumerate(nftag[1:]):

         if torch_device:
            symtags = _make_symtags_torch(tag, f5cell_torch, torch_device, t)
            seenit = torch.all(torch.isclose(torch.tensor(keep[None], dtype=torch.float32, device=torch_device), symtags[:, None], atol=0.001), axis=2)
            if torch.any(seenit): continue
            picktag = _pick_symelemtags(symtags.to('cpu').numpy(), symelems)
         else:
            symtags = _make_symtags(tag, f5cel)
            seenit = np.all(np.isclose(keep[None], symtags[:, None], atol=0.001), axis=2)
            if np.any(seenit): continue
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
            seenit = symelems[se.label].copy()
            if screw and se.label[:2] in symelems:
               seenit += symelems[se.label[:2]]
            if not any([_symelem_is_same(se, se2, f5cel) for se2 in seenit]):
               symelems[se.label].append(se)
         except wu.sym.ScrewError:
            continue

      # ic(symelems)
   symelems = _symelem_remove_ambiguous_syms(symelems)
   symelems = _find_compond_symelems(symelems, f5cel, f2geom)

   return symelems

def _symelem_is_same(elem, elem2, frames):
   assert elem.iscyclic and elem.label[:2] == elem2.label[:2]
   axis = einsum('fij,j->fi', frames, elem2.axis)
   axsame = np.all(np.isclose(axis, elem.axis), axis=1)
   axsameneg = np.all(np.isclose(-axis, elem.axis), axis=1)
   axok = np.logical_or(axsame, axsameneg)
   if not np.any(axok): return False
   frames = frames[axok]
   axis = axis[axok]
   cen = einsum('fij,j->fi', frames, elem2.cen)
   censame = np.all(np.isclose(cen, elem.cen), axis=1)
   # ic(censame.shape)
   if any(censame): return True
   # if any(censame):
   # ic(elem.axis, axis[censame])
   # ic(elem.cen, cen[censame])
   # ic(elem)
   # ic(elem2)
   # assert not any(censame)  # should have been filtered out already

   d = h_point_line_dist(elem.cen, cen, axis)
   return np.min(d) < 0.001

'''
ic| sym: 'I23'
    symelems: defaultdict(<class 'list'>,
                          {'C2': [SymElem(2, axis=[0.0, 0.0, 1.0], cen=[-0.0, -0.0, 0.0]),
                                  SymElem(2, axis=[0.0, 0.0, 1.0], cen=[0.5, -0.0, 0.0]),
                                  SymElem(2, axis=[0.0, 1.0, 0.0], cen=[0.5, 0.0, -0.0]),
                                  SymElem(2, axis=[0.0, 0.0, 1.0], cen=[0.5, 0.5, 0.0])],
                           'C21': [SymElem(2, axis=[0.0, 0.0, 1.0], cen=[0.25, 0.25, 0.0], hel=0.5)],
                           'C3': [SymElem(3, axis=[1.0, 1.0, 1.0], cen=[0.0, 0.0, 0.0])]})
ic| sym: 'P213'
    symelems: defaultdict(<class 'list'>,
                          {'C21': [SymElem(2, axis=[0.0, 0.0, 1.0], cen=[0.25, -0.0, 0.0], hel=0.5),
                                   SymElem(2, axis=[0.0, 1.0, 0.0], cen=[0.5, 0.0, 0.25], hel=0.5)],
                           'C3': [SymElem(3, axis=[1.0, 1.0, 1.0], cen=[0.0, 0.0, 0.0]),
                                  SymElem(3, axis=[1.0, -1.0, -1.0], cen=[0.0, 0.0, 0.5])]})
ic| sym: 'I213'
    symelems: defaultdict(<class 'list'>,
                          {'C2': [SymElem(2, axis=[0.0, 1.0, 0.0], cen=[0.25, 0.0, -0.0]),
                                  SymElem(2, axis=[0.0, 1.0, 0.0], cen=[0.75, 0.0, -0.0])],
                           'C21': [SymElem(2, axis=[0.0, 0.0, 1.0], cen=[0.25, -0.0, 0.0], hel=0.5),
                                   SymElem(2, axis=[0.0, 0.0, -1.0], cen=[0.75, 0.0, 0.0], hel=0.5)],
                           'C3': [SymElem(3, axis=[1.0, 1.0, 1.0], cen=[0.0, 0.0, 0.0])]})
ic| sym: 'P432'
    symelems: defaultdict(<class 'list'>,
                          {'C2': [SymElem(2, axis=[0.0, 1.0, 1.0], cen=[0.0, 0.0, 0.0]),
                                  SymElem(2, axis=[0.0, 0.0, 1.0], cen=[0.5, -0.0, 0.0]),
                                  SymElem(2, axis=[0.0, 1.0, 1.0], cen=[0.5, 0.0, -0.0])],
                           'C3': [SymElem(3, axis=[1.0, 1.0, 1.0], cen=[0.0, 0.0, 0.0])],
                           'C4': [SymElem(4, axis=[-0.0, -0.0, 1.0], cen=[0.0, 0.0, 0.0]),
                                  SymElem(4, axis=[0.0, 0.0, 1.0], cen=[0.5, 0.5, 0.0])]})
ic| 'PASS test_spacegroup_symelems'

'''

def _make_symtags(tag, frames):
   import torch
   concat = np.concatenate
   tax, tcen, thel = wu.hvec(tag[:3]), wu.hpoint(tag[3:6]), np.tile(tag[6], [len(frames), 1])

   # if is 21, 42, 63 screw, allow reverse axis with same helical shift
   c1 = (frames @ tax)[:, :3]
   c2 = (frames @ tcen)[:, :3]
   if np.any(np.isclose(thel, [0.5, np.sqrt(2) / 2])):
      # if is 21, 42, 63 screw, allow reverse axis with same helical shift
      symtags = concat([
         concat([c1, c2, +thel], axis=1),
         concat([-c1, c2, -thel], axis=1),
         concat([-c1, c2, +thel], axis=1),
      ])
   else:
      symtags = concat([
         concat([c1, c2, +thel], axis=1),
         concat([-c1, c2, -thel], axis=1),
      ])

   return symtags

def _make_symtags_torch(tag, frames, torch_device, t):
   import torch
   tag = torch.tensor(tag, device=torch_device).to(torch.float32)
   # frames = torch.tensor(frames, device=torch_device).to(torch.float32)
   concat = torch.cat
   tax, tcen, thel = wu.th_vec(tag[:3]), wu.th_point(tag[3:6]), torch.tile(tag[6], [len(frames), 1])

   # concat = np.concatenate
   # tax, tcen, thel = wu.hvec(tag[:3]), wu.hpoint(tag[3:6]), np.tile(tag[6], [len(frames), 1])

   const1 = torch.tensor(0.5, device=torch_device).to(torch.float32)
   const2 = torch.tensor(np.sqrt(2.) / 2., device=torch_device).to(torch.float32)
   c1 = (frames @ tax)[:, :3]
   c2 = (frames @ tcen)[:, :3]
   if torch.any(torch.isclose(thel, const1)) or torch.any(torch.isclose(thel, const2)):
      # if is 21, 42, 63 screw, allow reverse axis with same helical shift
      symtags = concat([
         concat([c1, c2, +thel], axis=1),
         concat([-c1, c2, -thel], axis=1),
         concat([-c1, c2, +thel], axis=1),
      ])
   else:
      symtags = concat([
         concat([c1, c2, +thel], axis=1),
         concat([-c1, c2, -thel], axis=1),
      ])
   return symtags

def _find_compond_symelems(symelems, t5cell, f2geom):
   # assert 0
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

def _symelem_remove_ambiguous_syms(symelems):
   symelems = symelems.copy()
   for sym1, sym2 in [('C2', 'C4'), ('C3', 'C6')]:
      if sym2 in symelems:
         newc2 = list()
         for s2 in symelems[sym1]:
            for s in symelems[sym2]:
               if np.allclose(s.axis, s2.axis):
                  if h_point_line_dist(s.cen, s2.cen, s.axis) < 0.001:
                     break
                  # if np.allclose(s.cen, s2.cen): break

            else:
               newc2.append(s2)
         symelems[sym1] = newc2
   return symelems
