import willutil as wu
from willutil.motif.motif_placer import *
import numpy as np
import torch

def main():
   test_motif_placer()

def test_motif_placer():
   pdb = wu.pdb.readpdb(wu.test_data_path('pdb/1pgx.pdb1.gz'))
   # pdb = wu.pdb.readpdb(wu.test_data_path('pdb/1pgx.pdb1.gz'))
   xyz = torch.tensor(pdb.ncac(), device='cpu')
   # xyz = torch.tensor(pdb.ncac(), device='cuda')
   nasym = 35

   t = wu.Timer()

   motif, motifpos = make_test_motif(xyz, sizes=[7, 11], rnoise=1, nasym=nasym)
   print(motifpos)
   t.checkpoint('make_test_motif')

   doffset, dme, alldo, alldme = place_motif_dme_brute(xyz, motif, nasym=nasym)
   t.checkpoint('dme_brute')

   fastdme = place_motif_dme_fast(xyz, motif, nasym=nasym)
   t.checkpoint('fastdme')

   x = fastdme[tuple(alldo.T)]
   assert torch.allclose(x, alldme)

   # roffset, rms, allro, allrms = place_motif_rms_brute(xyz, motif, nasym=nasym)
   # t.checkpoint('rms_brute')
   t.report()

   assert 0, 'PASS'

   ok = torch.logical_and(allrms < 10, alldme < 10)
   wu.viz.scatter(allrms[ok], alldme[ok])

   print(dme)
   print(rms)
   # assert torch.all(roffset[0] == doffset[0])
   assert 0

   wu.showme(xyz, name='ref')
   for i, ofst in enumerate(roffset):
      scrd = torch.cat([xyz[o:o + len(m), 1] for o, m in zip(ofst, motif)])
      mcrd = wu.th_point(torch.cat(motif)[:, 1])
      rms, _, x = wu.th_rmsfit(mcrd, wu.th_point(scrd))
      wu.showme(wu.th_xform(x, torch.cat(motif)), name=f'rms{rms}')

   assert 0

   # wu.showme(motif)
   # wu.showme(xyz)

if __name__ == '__main__':
   main()
