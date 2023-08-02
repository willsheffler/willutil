import willutil as wu
from willutil.motif.motif_placer import *
import numpy as np
import torch

def main():
   test_remove_symmdup_offsets()
   # test_check_offsets_overlap_containment()
   test_motif_placer()
   print('test_motif_placer PASS', flush=True)

def test_remove_symmdup_offsets():
   RSO = remove_symmdup_offsets
   AC = np.allclose
   assert AC([(0, 0)], RSO([(0, 0), (0, 10)], nasym=10))

def test_check_offsets_overlap_containment():
   AC = np.allclose
   F = check_offsets_overlap_containment
   assert AC([1, 0, 0], F([0, 1, 2], sizes=[10], nres=30, nasym=10))
   assert AC([1, 1, 0], F([0, 10, 11], sizes=[10], nres=40, nasym=20))
   assert AC([1, 1, 0, 0], F([0, 10, 11, 20], sizes=[10], nres=40, nasym=20))

def test_motif_placer(showme=False):
   t = wu.Timer()

   pdb = wu.pdb.readpdb(wu.test_data_path('pdb/1pgx.pdb1.gz'))
   # pdb = wu.pdb.readpdb(wu.test_data_path('pdb/1pgx.pdb1.gz'))
   xyz = torch.tensor(pdb.ncac(), device='cpu')
   # xyz = torch.tensor(pdb.ncac(), device='cuda')
   nres, nasym = len(xyz), 35
   cbreaks = get_symm_cbreaks(nres, nasym, cbreaks=[0])
   ic(cbreaks)

   motif, motifpos = make_test_motif(xyz, sizes=[14, 15, 16, 17], rnoise=1, nasym=nasym, cbreaks=cbreaks)
   ic(motifpos)
   t.checkpoint('make_test_motif')

   doffset, dme, alldo, alldme = place_motif_dme_brute(xyz, motif, nasym=nasym, cbreaks=cbreaks)
   t.checkpoint('dme_brute')

   fastdme = place_motif_dme_fast(xyz, motif, nasym=nasym, cbreaks=cbreaks)
   t.checkpoint('fastdme')

   x = fastdme[tuple(alldo.T)]
   assert torch.allclose(x, alldme)

   # roffset, rms, allro, allrms = place_motif_rms_brute(xyz, motif, nasym=nasym)
   # t.checkpoint('rms_brute')
   # print(rms)
   # ok = torch.logical_and(allrms < 10, alldme < 10)
   # wu.viz.scatter(allrms[ok], alldme[ok])

   if showme:
      wu.showme(xyz, name='ref')
      for i, ofst in enumerate(roffset):
         scrd = torch.cat([xyz[o:o + len(m), 1] for o, m in zip(ofst, motif)])
         mcrd = wu.th_point(torch.cat(motif)[:, 1])
         rms, _, x = wu.th_rmsfit(mcrd, wu.th_point(scrd))
         wu.showme(wu.th_xform(x, torch.cat(motif)), name=f'rms{rms}')

   t.report()
   # wu.showme(motif)
   # wu.showme(xyz)

if __name__ == '__main__':
   main()
