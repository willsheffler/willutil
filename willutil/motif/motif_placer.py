import itertools
import willutil as wu
import numpy as np
import torch
from willutil.cpp.rms import qcp_rms_regions_f4i4

from collections import namedtuple

MotifPlacement = namedtuple('MotifPlacement', 'offset score alloffset allscore')

def get_symm_cbreaks(nres, nasym=None, cbreaks=None):
   nasym = nasym or nres
   assert nres % nasym == 0
   cbreaks = [0] if cbreaks is None else cbreaks
   cbreaks = {((int(c) % nasym) + s * nasym) for c, s in itertools.product([0] + list(cbreaks), range(nres // nasym))}
   cbreaks = torch.tensor(sorted(cbreaks.union({nres})))
   return cbreaks

#def remove_symmdup_offsets(offsets, nasym):
#   assert 0
#   offsets = torch.as_tensor(offsets)
#   if offsets.ndim == 1: offsets = offsets.unsqueeze(1)
#   asymofst = offsets % nasym
#   mul = torch.cumprod(torch.cat([torch.tensor([1]), torch.max(asymofst[:, :-1], axis=0).values + 1]), dim=0)
#   uid = torch.sum(mul * asymofst, axis=1)
#   uniq_idx = uid.unique(return_inverse=True)[1]
#   isdup = torch.zeros(len(offsets), dtype=bool)
#   isdup[uniq_idx.unique()] = True
#   return isdup

def check_offsets_overlap_containment(offsets, sizes, nres, nasym=None, cbreaks=None, minsep=0, minbeg=0, minend=0):
   offsets = torch.as_tensor(offsets)
   if offsets.ndim == 1: offsets = offsets.unsqueeze(1)
   cbreaks = get_symm_cbreaks(nres, nasym, cbreaks)
   ok = torch.ones(len(offsets), dtype=bool)
   for i in range(len(sizes)):
      lbi, ubi = offsets[:, i], offsets[:, i] + sizes[i]
      for j in range(i):
         lbj, ubj = offsets[:, j], offsets[:, j] + sizes[j]
         mn = torch.minimum(lbi, lbj)
         mx = torch.maximum(ubi, ubj)
         ok &= mx - mn >= sizes[i] + sizes[j] + minsep
      for lbs, ubs in cbreaks.unfold(0, 2, 1):
         # for j in range(nres // nasym):
         # lbs, ubs = j * nasym, (j + 1) * nasym
         ok &= torch.logical_or(
            torch.logical_and(lbs <= lbi, ubi <= ubs),
            torch.logical_or(ubs <= lbi, ubi <= lbs),
         )
   # if nasym != nres:
   # ok[ok.clone()] &= remove_symmdup_offsets(offsets[ok], nasym)

   return ok

def make_floating_offsets(sizes, nres, nasym, cbreaks, minsep=0, minbeg=0, minend=0):
   # assert len(sizes) < 5, 'this may work badly for many chains'
   nasym = nasym or nres
   offset1d = [torch.arange(minbeg, nasym - sizes[0] + 1 - minend)]
   offset1d += [torch.arange(minbeg, nres - l + 1 - minend) for l in sizes[1:]]
   # for o in offset1d:
   # ic(o.shape)
   offsets = torch.cartesian_prod(*offset1d)
   ok = check_offsets_overlap_containment(offsets, sizes, nres, nasym, cbreaks, minsep, minbeg, minend)
   offsets = offsets[ok]
   if offsets.ndim == 1: offsets = offsets.unsqueeze(1)
   return offsets

def place_motif_rms_brute(xyz, motif, topk=10, minsep=0, nasym=None, cbreaks=[0]):
   import torch
   nasym = nasym or nres
   ca = wu.th_point(xyz[:, 1])
   mcrd = wu.th_point(torch.cat(motif)[:, 1])
   offsets = make_floating_offsets([len(m) for m in motif], nres, nasym, cbreaks, minsep=minsep)
   rms = torch.zeros(len(offsets))
   for i, offset in enumerate(offsets):
      scrd = torch.cat([ca[o:o + len(m)] for o, m in zip(offset, motif)])
      rms[i], _, _ = wu.th_rmsfit(mcrd, scrd)
   val, idx = torch.topk(rms, topk, largest=False)
   return MotifPlacement(offsets[idx], rms[idx], offsets, rms)

def place_motif_dme_brute(xyz, motif, topk=10, minsep=0, nasym=None, cbreaks=[0]):

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
   offsets = make_floating_offsets([len(m) for m in motif], nres, nasym, cbreaks, minsep=minsep)
   assert torch.all(offsets[:, 0] <= nasym)
   dme = torch.zeros(len(offsets), device=xyz.device)
   # dme_test = torch.zeros(len(offsets))
   for ioffset, offset in enumerate(offsets):
      scrd = torch.cat([ca[o:o + len(m)] for o, m in zip(offset, motif)])
      sdist = torch.cdist(scrd, scrd)
      dme[ioffset] = torch.sqrt(torch.square(mdist - sdist).mean())
      # dme[ioffset] = torch.square(mdist - sdist).mean()
      # dme[ioffset] = torch.abs(mdist - sdist).mean()

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

FastDMEMotifPlacement = namedtuple('FastDMEMotifPlacement', 'offset rms drms alldme')

def place_motif_dme_fast(xyz, motif, nasym=None, cbreaks=[], nrmsalign=100_000, nolapcheck=1_000_000, minsep=0, minbeg=0, minend=0, junct=0, return_alldme=False):
   assert xyz.shape[-2:] == (3, 3)
   sizes = [len(m) for m in motif]
   dmotif = list()
   for ichain, icrd in enumerate(motif):
      dmotif.append(list())
      for jchain, jcrd in enumerate(motif):
         dmotif[ichain].append(torch.cdist(icrd[:, 1], jcrd[:, 1]))
   d = torch.cdist(xyz[:, 1], xyz[:, 1])

   nres = len(d)
   nasym = nasym or nres
   # assert nasym == nres
   alldme = compute_offset_dme_fast(xyz, motif, dmotif, d, nres, nasym, junct)

   _, idx = torch.topk(alldme.flatten(), min(alldme.nelement(), nolapcheck), largest=False)
   offsets = torch.as_tensor(np.stack(np.unravel_index(idx.cpu().numpy(), alldme.shape), axis=1))

   ok = check_offsets_overlap_containment(offsets, sizes, nres, nasym, cbreaks, minsep, minbeg, minend)
   offsets = offsets[ok][:nrmsalign]

   xyz_motif = torch.cat([m[:, 1] for m in motif])
   rms = qcp_rms_regions_f4i4(xyz[:, 1].cpu(), xyz_motif.cpu(), sizes, offsets, junct=junct)
   order = np.argsort(rms)
   rms = rms[order]
   offsets = offsets[order]
   dme = alldme[tuple(offsets.T)].cpu().numpy()

   if not return_alldme:
      del alldme
      alldme = None
   return FastDMEMotifPlacement(offsets, rms, dme, alldme)

def compute_dme_corners(dmat1, dmat2, junct=0, method='mse'):
   assert dmat1.shape[-2:] == dmat2.shape
   m, n = dmat1.shape[-2:]

   if not junct:
      diff = dmat1 - dmat2
   else:
      assert junct > 0
      if dmat1.ndim == 3 and m < 2 * junct:
         diff = dmat1 - dmat2
      elif dmat1.ndim == 3:
         c = junct
         d = m - c
         # assert 0
         diff = torch.stack([
            (dmat1[..., :c, :c] - dmat2[..., :c, :c]).flatten(1),
            (dmat1[..., :c, d:] - dmat2[..., :c, d:]).flatten(1),
            (dmat1[..., d:, :c] - dmat2[..., d:, :c]).flatten(1),
            (dmat1[..., d:, d:] - dmat2[..., d:, d:]).flatten(1),
         ], axis=-1)
      elif dmat1.ndim == 4:
         cm, cn, dm, dn = junct, junct, m - junct, n - junct
         if cm > dm: cm, dm = m // 2, m // 2
         if cn > dn: cn, dn = n // 2, n // 2
         # ic(m, cm, dm)
         # ic(n, cn, dn)
         diff = torch.cat([
            (dmat1[..., :cm, :cn] - dmat2[..., :cm, :cn]).flatten(2),
            (dmat1[..., :cm, dn:] - dmat2[..., :cm, dn:]).flatten(2),
            (dmat1[..., dm:, :cn] - dmat2[..., dm:, :cn]).flatten(2),
            (dmat1[..., dm:, dn:] - dmat2[..., dm:, dn:]).flatten(2),
         ], axis=-1).unsqueeze(-1)
      else:
         raise ValueError(f'unknown distmat shape {dmat1.shape}')

   if method == 'mse':
      dme = torch.square(diff)
   elif method == 'abs':
      dme = torch.abs(diff)
   else:
      raise ValueError(f'unknown compute_dme_corners method {method}')

   return dme.sum(axis=(-1, -2))

def compute_offset_dme_fast(xyz, motif, dmotif, d, nres, nasym, junct=0, method='mse'):
   offsetshape = [len(d) - len(c) + 1 for c in motif]
   offsetshape[0] = nasym - len(motif[0]) + 1
   alldme = torch.zeros(offsetshape, device=xyz.device)  #, dtype=torch.float16)
   for i in range(len(dmotif)):
      dtgt = dmotif[i][i].to(dtype=d.dtype)
      distmat = d[:nasym if i == 0 else nres]
      dunf = distmat.unfold(0, len(dtgt), 1).unfold(1, len(dtgt), 1).diagonal().permute(2, 0, 1)
      dme1b = compute_dme_corners(dunf, dtgt, junct, method)
      newshape = [1] * alldme.ndim
      newshape[i] = len(dme1b)
      alldme += dme1b.reshape(newshape)
      for j in range(i + 1, len(dmotif)):
         dtgt = dmotif[i][j].to(dtype=distmat.dtype)
         dunf = distmat.unfold(0, dtgt.shape[0], 1).unfold(1, dtgt.shape[1], 1)
         dme2b = compute_dme_corners(dunf, dtgt, junct, method)
         newshape = [1] * alldme.ndim
         newshape[i], newshape[j] = dme2b.shape
         alldme += 2 * dme2b.reshape(newshape)
   alldme /= sum([len(m) for m in motif])**2
   alldme = torch.sqrt(alldme)
   return alldme

def make_test_motif(xyz, sizes, minsep=0, minbeg=0, minend=0, ntries=3, rnoise=0, lever=10, nasym=None, cbreaks=[0]):
   nres = len(xyz)
   nasym = nasym or nres

   pos = None
   for itry in range(ntries):
      N = 1000 * 10**itry
      offsets = torch.stack([torch.randint((nasym if il == 0 else nres) - l + 1, (N, )) for il, l in enumerate(sizes)], axis=1)
      ok = check_offsets_overlap_containment(offsets, sizes, nres, nasym, cbreaks, minsep, minbeg, minend)
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
