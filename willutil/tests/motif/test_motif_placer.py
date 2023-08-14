import random
import willutil as wu
from willutil.motif.motif_placer import *
import numpy as np
import torch

def main():
   # a = torch.rand(10000) < 0.5
   # b = torch.rand(10000) < 0.5
   # c = a * b
   # a[a.clone()] = b[a]
   # assert torch.all(a == c)
   # assert 0, 'PASS'

   test_motif_occlusion()

   test_check_offsets_overlap_containment()
   test_motif_placer()
   test_motif_placer_minbeg_minend()
   perftest_motif_placer()

   print('test_motif_placer PASS', flush=True)

def randslice(n):
   r = random.randrange(n), random.randrange(n)
   return slice(min(r), max(r))

def test_motif_occlusion():
   xyz = torch.as_tensor(wu.tests.load_test_data('xyz/1pgx_ncac.npy'), device='cpu')
   xyz = xyz[:12]

   nres, nasym = len(xyz), None
   cbreaks = get_symm_cbreaks(nres, nasym, cbreaks=[])
   sizes = [3, 3]
   # motif, motifpos = make_test_motif(xyz, sizes, rnoise=0.1, nasym=nasym, cbreaks=cbreaks)
   # print(motifpos)
   xyz[0:3, :, 0] += 100000
   xyz[6:9, :, 0] += 100000

   # placement = place_motif_dme_fast(xyz, motif, nasym=nasym, cbreaks=cbreaks)

   contacts = (1000.0 > torch.cdist(xyz[:, 1], xyz[:, 1])).to(torch.float32)
   ic(contacts.to(int))
   # compute_offset_occlusion_brute(contacts, sizes, cbreaks=cbreaks)

   occ = compute_offset_occlusion_tensor(contacts, sizes)
   ic(occ.shape)
   ic(occ.to(int))

   # showme_motif_placements(xyz, motif, placement.offset[:10])

   # assert 0

def perftest_motif_placer():
   t = wu.Timer()
   N = 100
   xyz = torch.tensor(wu.hrandpoint(3 * N)[:, :3].reshape(N, 3, 3), dtype=torch.float32, device='cuda')

   nres, nasym = len(xyz), None
   cbreaks = get_symm_cbreaks(nres, nasym, cbreaks=[])
   junct = 5
   sizes = [15, 15, 15, 15]
   # ic(sum(sizes))
   # ic(cbreaks)

   motif, motifpos = make_test_motif(xyz, sizes, rnoise=0.1, nasym=nasym, cbreaks=cbreaks)
   # ic(motifpos)
   t.checkpoint('make_test_motif')

   _ = place_motif_dme_fast(xyz[:40], motif, nasym=nasym, cbreaks=cbreaks, junct=junct, nolapcheck=10, nrmsalign=10)
   t.checkpoint('fastdme_init')
   result = place_motif_dme_fast(
      xyz,
      motif,
      nasym=nasym,
      cbreaks=cbreaks,
      junct=junct,
      return_alldme=True,
      nolapcheck=1000,
      nrmsalign=1,
      motif_occlusion_weight=1,
      motif_occlusion_dist=13,
   )
   t.checkpoint('fastdme')
   # t.report()
   assert np.allclose([m[0] for m in motifpos], result.offset[0])

   runtime = sum(t.checkpoints['fastdme'])
   ncalc = result.alldme.nelement()
   print(f'fastdme ncalc: {ncalc:,}, rate: {ncalc/runtime/1e9:7.5}G', flush=True)

   # ic(result.rms[:4])
   # ic(result.drms[:4])
   # ic(motifpos, result.offset[0])

   # wu.viz.scatter(result.rms, result.drms)

def test_check_offsets_overlap_containment():
   AC, T, F, COOC = np.allclose, True, False, check_offsets_overlap_containment

   assert AC([T, F, F], COOC([0, 1, 2], sizes=[10], nres=30, nasym=10))
   assert AC([T, T, F], COOC([0, 10, 11], sizes=[10], nres=40, nasym=20))
   assert AC([F, T, T], COOC([0, 10, 11], sizes=[10], nres=40, minbeg=3))
   assert AC([T, T, F], COOC([0, 10, 28], sizes=[10], nres=40, minend=3))
   assert AC([T, T, F, T], COOC([0, 10, 11, 20], sizes=[10], nres=40, nasym=20))

def test_motif_placer(showme=False):
   xyz = torch.as_tensor(wu.tests.load_test_data('xyz/1pgx_ncac.npy'), device='cpu')
   nres, nasym = len(xyz), None
   cbreaks = get_symm_cbreaks(nres, nasym, cbreaks=[15, 38])
   junct = 10
   sizes = [13, 14]
   # ic(sum(sizes))
   # ic(cbreaks)

   motif, motifpos = make_test_motif(xyz, sizes, rnoise=0.1, nasym=nasym, cbreaks=cbreaks)

   # fastdme = place_motif_dme_fast(xyz, motif, nasym=nasym, cbreaks=cbreaks, junct=junct)
   fastdme = place_motif_dme_fast(xyz, motif, nasym=nasym, cbreaks=cbreaks, junct=junct, return_alldme=True)

   doffset, dme, alldo, alldme = place_motif_dme_brute(xyz, motif, nasym=nasym, cbreaks=cbreaks)

   x = fastdme.alldme[tuple(alldo.T)]
   if all([junct * 2 >= s for s in sizes]):
      assert torch.allclose(x, alldme)
   # else:
   # wu.viz.scatter(alldme, x)
   # roffset, rms, allro, allrms = place_motif_rms_brute(xyz, motif, nasym=nasym)
   # print(rms)
   # ok = torch.logical_and(allrms < 10, alldme < 10)
   # wu.viz.scatter(allrms[ok], alldme[ok])

   assert np.allclose([m[0] for m in motifpos], fastdme.offset[0])

   if showme:
      showme_motif_placements(xyz, motif, doffset)
      assert 0
   # t.report()
   # wu.showme(motif)
   # wu.showme(xyz)

def showme_motif_placements(xyz, motif, offsets):
   wu.showme(xyz, name='ref')
   for i, ofst in enumerate(offsets):
      scrd = torch.cat([xyz[o:o + len(m), 1] for o, m in zip(ofst, motif)])
      mcrd = wu.th_point(torch.cat(motif)[:, 1])
      rms, _, x = wu.th_rmsfit(mcrd, wu.th_point(scrd))
      wu.showme(wu.th_xform(x, torch.cat(motif)), name=f'rms{rms}')

def test_motif_placer_minbeg_minend(showme=False):
   for minbeg in [0, 3]:
      for minend in [0, 4]:
         pdb = wu.pdb.readpdb(wu.test_data_path('pdb/1pgx.pdb1.gz'))
         xyz = torch.tensor(pdb.ncac(), device='cpu')
         nres, nasym = len(xyz), None
         cbreaks = get_symm_cbreaks(nres, nasym, cbreaks=[20, 40, 60])
         junct = 10
         sizes = [12, 13]
         motif, motifpos = make_test_motif(xyz, sizes, rnoise=0.1, nasym=nasym, cbreaks=cbreaks, minbeg=minbeg, minend=minend)
         fastdme = place_motif_dme_fast(xyz, motif, nasym=nasym, cbreaks=cbreaks, junct=junct, return_alldme=True, minbeg=minbeg, minend=minend)
         doffset, dme, alldo, alldme = place_motif_dme_brute(xyz, motif, nasym=nasym, cbreaks=cbreaks, minbeg=minbeg, minend=minend)
         x = fastdme.alldme[tuple(alldo.T - minbeg)]
         if all([junct * 2 >= s for s in sizes]):
            assert torch.allclose(x, alldme)
         assert np.allclose([m[0] for m in motifpos], fastdme.offset[0])

if __name__ == '__main__':
   main()
