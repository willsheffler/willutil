import willutil as wu
import numpy as np
import torch

from collections import namedtuple

MotifPlacement = namedtuple('MotifPlacement', 'offset score alloffset allscore')

def chek_offsets_overlap_containment(offsets, sizes, nres, nasym, minsep=0, minbeg=0, minend=0):
   ok = torch.ones(len(offsets), dtype=bool)
   for i in range(len(sizes)):
      lbi, ubi = offsets[:, i], offsets[:, i] + sizes[i]
      for j in range(i):
         lbj, ubj = offsets[:, j], offsets[:, j] + sizes[j]
         mn = torch.minimum(lbi, lbj)
         mx = torch.maximum(ubi, ubj)
         ok &= mx - mn >= sizes[i] + sizes[j] + minsep
      for j in range(nres // nasym):
         lbs, ubs = j * nasym, (j + 1) * nasym
         ok &= torch.logical_or(
            torch.logical_and(lbs < lbi, ubi <= ubs),
            torch.logical_or(ubs <= ubi, ubi < lbs),
         )
   return ok

def make_floating_offsets(sizes, nres, nasym, minsep=0, minbeg=0, minend=0):
   assert len(sizes) < 5, 'this may work badly for many chains'
   nasym = nasym or nres
   offset1d = [torch.arange(minbeg, nasym - sizes[0] + 1 - minend)]
   offset1d += [torch.arange(minbeg, nres - l + 1 - minend) for l in sizes[1:]]
   # for o in offset1d:
   # ic(o.shape)
   offsets = torch.cartesian_prod(*offset1d)
   ok = chek_offsets_overlap_containment(offsets, sizes, nres, nasym, minsep, minbeg, minend)
   offsets = offsets[ok]
   if offsets.ndim == 1: offsets = offsets.unsqueeze(1)
   return offsets

def place_motif_rms_brute(xyz, motif, topk=10, minsep=5):
   import torch
   ca = wu.th_point(xyz[:, 1])
   mcrd = wu.th_point(torch.cat(motif)[:, 1])
   offsets = make_floating_offsets([len(m) for m in motif], len(ca), minsep=minsep)
   rms = torch.zeros(len(offsets))
   for i, offset in enumerate(offsets):
      scrd = torch.cat([ca[o:o + len(m)] for o, m in zip(offset, motif)])
      rms[i], _, _ = wu.th_rmsfit(mcrd, scrd)
   val, idx = torch.topk(rms, topk, largest=False)
   return MotifPlacement(offsets[idx], rms[idx], offsets, rms)

def place_motif_dme_brute(xyz, motif, topk=10, minsep=0, nasym=None):

   dmotif = list()
   for ichain, icrd in enumerate(motif):
      dmotif.append(list())
      for jchain, jcrd in enumerate(motif):
         dmotif[ichain].append(torch.cdist(icrd[:, 1], jcrd[:, 1]))
   d = torch.cdist(xyz[:, 1], xyz[:, 1])

   nres = len(xyz)
   nasym = nasym or nres
   ca = wu.th_point(xyz[:, 1])
   mcrd = wu.th_point(torch.cat(motif)[:, 1])
   mdist = torch.cdist(mcrd, mcrd)
   offsets = make_floating_offsets([len(m) for m in motif], nres, nasym, minsep=minsep)
   assert torch.all(offsets[:, 0] <= nasym)
   dme = torch.zeros(len(offsets), device=xyz.device)
   # dme_test = torch.zeros(len(offsets))
   for ioffset, offset in enumerate(offsets):
      scrd = torch.cat([ca[o:o + len(m)] for o, m in zip(offset, motif)])
      sdist = torch.cdist(scrd, scrd)
      # dme[ioffset] = torch.sqrt(((mdist - sdist)**2).mean())
      dme[ioffset] = torch.abs(mdist - sdist).sum()

      # dme[ioffset] = 0
      # for i in range(len(dmotif)):
      # for j in range(len(dmotif)):
      # m, n = dmotif[i][j].shape
      # dij = d[offset[i]:offset[i] + m, offset[j]:offset[j] + n]
      # ic(dij.shape, dmotif[i][j].shape)
      # dme[ioffset] += torch.abs(dmotif[i][j] - dij).sum()

   # assert torch.allclose(dme, dme_test)

   val, idx = torch.topk(dme, topk, largest=False)
   return MotifPlacement(offsets[idx], dme[idx], offsets, dme)

def place_motif_dme_fast(xyz, motif, nasym=None, nrmsalign=100, nolapcheck=10000):
   dmotif = list()
   for ichain, icrd in enumerate(motif):
      dmotif.append(list())
      for jchain, jcrd in enumerate(motif):
         dmotif[ichain].append(torch.cdist(icrd[:, 1], jcrd[:, 1]))
   d = torch.cdist(xyz[:, 1], xyz[:, 1])

   nres = len(d)
   nasym = nasym or nres
   # assert nasym == nres
   alldme = compute_offset_dme_fast(xyz, motif, dmotif, d, nres, nasym)

   _, idx = torch.topk(alldme.flatten(), min(alldme.nelement(), nolapcheck), largest=False)
   offset = torch.as_tensor(np.stack(np.unravel_index(idx.numpy(), alldme.shape), axis=1))
   ic(offset.shape)

   return alldme

def compute_offset_dme_fast(xyz, motif, dmotif, d, nres, nasym):
   offsetshape = [len(d) - len(c) + 1 for c in motif]
   offsetshape[0] = nasym - len(motif[0]) + 1
   alldme = torch.zeros(offsetshape, device=xyz.device)
   for i in range(len(dmotif)):
      dtgt = dmotif[i][i].to(dtype=d.dtype)
      distmat = d[:nasym if i == 0 else nres]
      dunf = distmat.unfold(0, len(dtgt), 1).unfold(1, len(dtgt), 1).diagonal()
      dme1b = torch.abs(dunf - dtgt.unsqueeze(2)).sum(axis=(0, 1))
      newshape = [1] * alldme.ndim
      newshape[i] = len(dme1b)
      alldme += dme1b.reshape(newshape)
      for j in range(i + 1, len(dmotif)):
         dtgt = dmotif[i][j].to(dtype=distmat.dtype)
         dunf = distmat.unfold(0, dtgt.shape[0], 1).unfold(1, dtgt.shape[1], 1)
         dme2b = torch.abs(dunf - dtgt).sum(axis=(2, 3))
         newshape = [1] * alldme.ndim
         newshape[i], newshape[j] = dme2b.shape
         alldme += 2 * dme2b.reshape(newshape)
   return alldme

def make_test_motif(xyz, sizes, minsep=5, minbeg=0, minend=0, ntries=3, rnoise=0, lever=10, nasym=None):
   nres = len(xyz)
   nasym = nasym or nres

   pos = None
   for itry in range(ntries):
      N = 1000 * 10**itry
      offsets = torch.stack([torch.randint((nasym if il == 0 else nres) - l + 1, (N, )) for il, l in enumerate(sizes)], axis=1)
      ok = chek_offsets_overlap_containment(offsets, sizes, nres, nasym, minsep, minbeg, minend)
      if torch.any(ok):
         offsets = offsets[ok]
         pos = sorted(tuple([(int(f), int(f) + sizes[i]) for i, f in enumerate(offsets[0])]))
         break
   else:
      raise ValueError(f'no valid motif partitions found in {N:,} samples')

   motif = list()
   for lb, ub in pos:
      crd = xyz[lb:ub]
      com = wu.hcom(crd.cpu(), flat=True)
      x = wu.hrandsmall(cart_sd=rnoise * 0.707, rot_sd=0.707 * rnoise / lever, centers=com)
      motif.append(wu.th_xform(x, crd.cpu()).to(xyz.device))

   return motif, pos
