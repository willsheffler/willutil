import os
import willutil as wu
import numpy as np

def test_pdb_metadata():
   # wu.pdb.pdbmeta.update_source_files()

   meta = wu.pdb.pdbmeta

   assert meta.nres['1PGX'] == 83

   assert len(meta.resl) > 180_000
   assert meta.resl['1PGX'] == 1.66
   assert meta.resl.dtype == np.float64

   assert np.all(meta.resl > 0)
   assert np.sum(meta.resl > 12345) > 10_000

   # dont know if this is worth worrying about
   # assert len(meta.resl) == len(meta.nres)
   # assert len(meta.xtal) == len(meta.nres)
   # assert len(meta.seq) == len(meta.nres)
   # assert len(meta.chainseq) == len(meta.nres)

   assert len(meta.xtal) > 100_000
   _ = 'CRYST1    26.040    34.500    35.950    90.000   100.840    90.000  P 1 21 1         2'
   assert meta.xtal['1PGX'] == _

   _ = 'MDPGDASELTPAVTTYKLVINGKTLKGETTTKAVDAETAEKAFKQYANDNGVDGVWTYDDATKTFTVTEMVTEVPVASKRKED'
   assert meta.chainseq['1PGX'] == {'A': _}
   assert all(len(x) > 0 for x in meta.chainseq)

   _ = 'MDPGDASELTPAVTTYKLVINGKTLKGETTTKAVDAETAEKAFKQYANDNGVDGVWTYDDATKTFTVTEMVTEVPVASKRKED'
   assert meta.seq['1PGX'] == _

def test_meta_search():
   from willutil.pdb import pdbmeta as meta
   pdbset = meta.make_pdb_set(
      maxresl=1.5,
      max_seq_ident=30,
      minres=50,
      maxres=200,
      pisces_chains=False,
   )
   assert len(pdbset) == 1104
   assert pdbset.issuperset(
      {'4AVR', '6GN5', '3S9X', '4RRI', '4I6X', '6O40', '5FUI', '6E7E', '3R72', '3FIL', '4MQ3'})
   # print(len(pdbset), pdbset)

def test_meta_search_pisces_chains():
   from willutil.pdb import pdbmeta as meta
   pdbset = meta.make_pdb_set(
      maxresl=1.5,
      max_seq_ident=30,
      minres=50,
      maxres=200,
      pisces_chains=True,
   )
   assert len(pdbset) == 1116
   assert pdbset.issuperset(
      {'4RGDA', '1OK0A', '3X0IA', '3IMKA', '3OBQA', '1JF8A', '1Y9LA', '4BK7A', '1X8QA', '5L87A'})

def main():
   # test_pdb_metadata()
   # test_meta_search()
   test_meta_search_pisces_chains()
   pass

if __name__ == '__main__':
   main()