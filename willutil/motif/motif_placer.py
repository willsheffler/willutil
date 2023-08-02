import willutil as wu
import numpy as np
import torch

from collections import namedtuple

MotifPlacement = namedtuple('MotifPlacement', 'offset score alloffset allscore')

def make_floating_offsets(motif, nasym, nsub=1, minsep=0, minbeg=0, minend=0):
   assert len(motif) < 4, 'this may work badly for many chains'
   nres = nsub * nasym
   lens = torch.tensor([len(c) for c in motif])
   offset1d = [torch.arange(minbeg, nasym - lens[0] + 1 - minend)]
   offset1d += [torch.arange(minbeg, nasym - l + 1 - minend) for l in lens[1:]]
   # for o in offset1d:
   # ic(o.shape)
   offsets = torch.cartesian_prod(*offset1d)
   ok = torch.ones(len(offsets), dtype=bool)
   for i in range(len(lens)):
      for j in range(i):
         ic(i, j, len(lens))
         mn = torch.minimum(offsets[:, i], offsets[:, j])
         mx = torch.maximum(offsets[:, i] + lens[i], offsets[:, j] + lens[j])
         # ic(i, j, max(mx - mn))
         ok &= mx - mn >= lens[i] + lens[j] + minsep
   offsets = offsets[ok]
   return offsets

def place_motif_rms_brute(xyz, motif, topk=10, minsep=5):
   import torch
   ca = wu.th_point(xyz[:, 1])
   mcrd = wu.th_point(torch.cat(motif)[:, 1])
   offsets = make_floating_offsets(motif, len(ca), minsep=minsep)
   rms = torch.zeros(len(offsets))
   for i, offset in enumerate(offsets):
      scrd = torch.cat([ca[o:o + len(m)] for o, m in zip(offset, motif)])
      rms[i], _, _ = wu.th_rmsfit(mcrd, scrd)
   val, idx = torch.topk(rms, topk, largest=False)
   return MotifPlacement(offsets[idx], rms[idx], offsets, rms)

def place_motif_dme_brute(xyz, motif, topk=10, minsep=0):

   dmotif = list()
   for ichain, icrd in enumerate(motif):
      dmotif.append(list())
      for jchain, jcrd in enumerate(motif):
         dmotif[ichain].append(torch.cdist(icrd[:, 1], jcrd[:, 1]))
   d = torch.cdist(xyz[:, 1], xyz[:, 1])

   ca = wu.th_point(xyz[:, 1])
   mcrd = wu.th_point(torch.cat(motif)[:, 1])
   mdist = torch.cdist(mcrd, mcrd)
   offsets = make_floating_offsets(motif, len(ca), minsep=minsep)
   dme = torch.zeros(len(offsets))
   # dme_test = torch.zeros(len(offsets))
   for ioffset, offset in enumerate(offsets):
      scrd = torch.cat([ca[o:o + len(m)] for o, m in zip(offset, motif)])
      sdist = torch.cdist(scrd, scrd)
      # dme[ioffset] = torch.sqrt(((mdist - sdist)**2).mean())
      dme[ioffset] = torch.abs(mdist - sdist).sum()

      dme[ioffset] = 0
      for i in range(len(dmotif)):
         for j in range(len(dmotif)):
            m, n = dmotif[i][j].shape
            dij = d[offset[i]:offset[i] + m, offset[j]:offset[j] + n]
            # ic(dij.shape, dmotif[i][j].shape)
            dme[ioffset] += torch.abs(dmotif[i][j] - dij).sum()

   # assert torch.allclose(dme, dme_test)

   val, idx = torch.topk(dme, topk, largest=False)
   return MotifPlacement(offsets[idx], dme[idx], offsets, dme)

def place_motif_dme_fast(xyz, motif, topk=10):
   dmotif = list()
   for ichain, icrd in enumerate(motif):
      dmotif.append(list())
      for jchain, jcrd in enumerate(motif):
         dmotif[ichain].append(torch.cdist(icrd[:, 1], jcrd[:, 1]))
   d = torch.cdist(xyz[:, 1], xyz[:, 1])

   ic(d.shape)
   nres = len(d)

   alldme = torch.zeros([len(d) - len(c) + 1 for c in motif], device=xyz.device)
   ic(alldme.shape)

   for i in range(len(dmotif)):
      dtgt = dmotif[i][i].to(dtype=d.dtype)
      dunf = d.unfold(0, len(dtgt), 1).unfold(1, len(dtgt), 1).diagonal()
      dme1b = torch.abs(dunf - dtgt.unsqueeze(2)).sum(axis=(0, 1))
      newshape = [1] * alldme.ndim
      newshape[i] = len(dme1b)
      alldme += dme1b.reshape(newshape)
      for j in range(i):
         dtgt = dmotif[i][j].to(dtype=d.dtype)
         m, n = dtgt.shape
         dunf = d.unfold(0, m, 1).unfold(1, n, 1)
         # assert torch.allclose(dunf[6, 13], d[7:7 + n, 13:13 + m])
         dme2b = torch.abs(dunf - dtgt).sum(axis=(2, 3))
         newshape = [1] * alldme.ndim
         newshape[i], newshape[j] = dme2b.shape
         alldme += 2 * dme2b.T.reshape(newshape)

   return alldme

def make_test_motif(xyz, regions, minsep=5, minbeg=0, minend=0, ntries=10000, rnoise=0, lever=10):
   nres = len(xyz)
   for i in range(ntries):
      pos = list()
      success = True
      for l in regions:
         lb = np.random.randint(nres - l + 1)
         ub = lb + l
         for lb0, ub0 in pos:
            boundlen = max(ub, ub0) - min(lb, lb0)
            totlen = l + ub0 - lb0
            success &= boundlen >= totlen + minsep
         pos.append([lb, ub])
      if success: break
   else:
      raise ValueError(f'cant make valid test motif in {ntries} trials')

   motif = list()
   for lb, ub in pos:
      crd = xyz[lb:ub]
      com = wu.hcom(crd.cpu(), flat=True)
      x = wu.hrandsmall(cart_sd=rnoise * 0.707, rot_sd=0.707 * rnoise / lever, centers=com)
      motif.append(wu.th_xform(x, crd.cpu()).to(xyz.device))

   return motif
