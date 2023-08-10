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

   test_check_offsets_overlap_containment()
   test_motif_placer()
   perftest_motif_placer()
   print('test_motif_placer PASS', flush=True)

def perftest_motif_placer():
   t = wu.Timer()
   N = 130
   xyz = torch.tensor(wu.hrandpoint(3 * N)[:, :3].reshape(N, 3, 3), dtype=torch.float32, device='cuda')

   nres, nasym = len(xyz), None
   cbreaks = get_symm_cbreaks(nres, nasym, cbreaks=[])
   junct = 5
   sizes = [15, 15, 15, 15]
   # ic(sum(sizes))
   # ic(cbreaks)

   motif, motifpos = make_test_motif(xyz, sizes, rnoise=0.0, nasym=nasym, cbreaks=cbreaks)
   # ic(motifpos)
   t.checkpoint('make_test_motif')

   _ = place_motif_dme_fast(xyz[:40], motif, nasym=nasym, cbreaks=cbreaks, junct=junct, nolapcheck=10, nrmsalign=10)
   t.checkpoint('fastdme_init')
   result = place_motif_dme_fast(xyz, motif, nasym=nasym, cbreaks=cbreaks, junct=junct, return_alldme=True, nolapcheck=500_000, nrmsalign=50_000)
   t.checkpoint('fastdme')
   # t.report()
   assert np.allclose([m[0] for m in motifpos], result.offset[0])

   runtime = sum(t.checkpoints['fastdme'])
   ncalc = result.alldme.nelement()
   print(f'fastdme ncalc: {ncalc:,}, rate: {ncalc/runtime:7.3}', flush=True)

   # ic(result.rms[:4])
   # ic(result.drms[:4])
   # ic(motifpos, result.offset[0])

   # wu.viz.scatter(result.rms, result.drms)

def test_check_offsets_overlap_containment():
   AC, T, F, COOC = np.allclose, True, False, check_offsets_overlap_containment
   assert AC([T, F, F], COOC([0, 1, 2], sizes=[10], nres=30, nasym=10))
   assert AC([T, T, F], COOC([0, 10, 11], sizes=[10], nres=40, nasym=20))
   assert AC([T, T, F, T], COOC([0, 10, 11, 20], sizes=[10], nres=40, nasym=20))

def test_motif_placer(showme=False):
   t = wu.Timer()

   pdb = wu.pdb.readpdb(wu.test_data_path('pdb/1pgx.pdb1.gz'))
   # pdb = wu.pdb.readpdb(wu.test_data_path('pdb/1pgx.pdb1.gz'))
   xyz = torch.tensor(pdb.ncac(), device='cpu')
   # xyz = torch.tensor(pdb.ncac(), device='cuda')
   nres, nasym = len(xyz), None
   cbreaks = get_symm_cbreaks(nres, nasym, cbreaks=[35])
   junct = 10
   sizes = [12, 13, 14]
   # ic(sum(sizes))
   # ic(cbreaks)

   motif, motifpos = make_test_motif(xyz, sizes, rnoise=0.1, nasym=nasym, cbreaks=cbreaks)
   t.checkpoint('make_test_motif')

   # fastdme = place_motif_dme_fast(xyz, motif, nasym=nasym, cbreaks=cbreaks, junct=junct)
   # t.checkpoint('fastdme_init')
   fastdme = place_motif_dme_fast(xyz, motif, nasym=nasym, cbreaks=cbreaks, junct=junct, return_alldme=True)
   t.checkpoint('fastdme')
   # t.report()
   # return

   doffset, dme, alldo, alldme = place_motif_dme_brute(xyz, motif, nasym=nasym, cbreaks=cbreaks)
   t.checkpoint('dme_brute')

   x = fastdme.alldme[tuple(alldo.T)]
   if all([junct * 2 >= s for s in sizes]):
      assert torch.allclose(x, alldme)
   # else:
   # wu.viz.scatter(alldme, x)
   # roffset, rms, allro, allrms = place_motif_rms_brute(xyz, motif, nasym=nasym)
   # t.checkpoint('rms_brute')
   # print(rms)
   # ok = torch.logical_and(allrms < 10, alldme < 10)
   # wu.viz.scatter(allrms[ok], alldme[ok])

   assert np.allclose([m[0] for m in motifpos], fastdme.offset[0])

   if showme:
      wu.showme(xyz, name='ref')
      for i, ofst in enumerate(roffset):
         scrd = torch.cat([xyz[o:o + len(m), 1] for o, m in zip(ofst, motif)])
         mcrd = wu.th_point(torch.cat(motif)[:, 1])
         rms, _, x = wu.th_rmsfit(mcrd, wu.th_point(scrd))
         wu.showme(wu.th_xform(x, torch.cat(motif)), name=f'rms{rms}')

   # t.report()
   # wu.showme(motif)
   # wu.showme(xyz)

if __name__ == '__main__':
   main()
