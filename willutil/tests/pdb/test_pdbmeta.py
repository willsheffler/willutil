import os
import numpy as np
import willutil as wu
from willutil.pdb import pdbmeta as meta

def test_pdb_metadata():
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
   pdbset = meta.make_pdb_set(
      maxresl=1.5,
      max_seq_ident=30,
      minres=50,
      maxres=200,
      pisces_chains=False,
      entrytype='all',
   )
   assert len(pdbset) == 1104
   assert pdbset.issuperset(
      {'4AVR', '6GN5', '3S9X', '4RRI', '4I6X', '6O40', '5FUI', '6E7E', '3R72', '3FIL', '4MQ3'})

   pdbset = meta.make_pdb_set(
      maxresl=1.5,
      max_seq_ident=30,
      minres=50,
      maxres=200,
      pisces_chains=False,
      entrytype='prot',
   )
   assert len(pdbset) == 1094
   assert pdbset.issuperset(
      {'4AVR', '6GN5', '3S9X', '4RRI', '4I6X', '6O40', '5FUI', '6E7E', '3R72', '3FIL', '4MQ3'})

def test_meta_search_pisces_chains():
   pdbset = meta.make_pdb_set(maxresl=1.5, max_seq_ident=30, minres=50, maxres=200,
                              pisces_chains=True, entrytype='prot')
   assert len(pdbset) == 1106
   assert pdbset.issuperset(
      {'4RGDA', '1OK0A', '3X0IA', '3IMKA', '3OBQA', '1JF8A', '1Y9LA', '4BK7A', '1X8QA', '5L87A'})

def test_meta_pdb_compound():
   _ = 'THE 1.66 ANGSTROMS X-RAY STRUCTURE OF THE B2 IMMUNOGLOBULIN-BINDING DOMAIN OF STREPTOCOCCAL PROTEIN G AND COMPARISON TO THE NMR STRUCTURE OF THE B1 DOMAIN'
   assert meta.compound['1PGX'] == _

def test_meta_clust():
   from willutil.pdb.pdbmeta import clust30, clust40, clust50, clust70, clust90, clust95, clust100

   ntot = sum(len(_) for _ in clust30)
   prev = len(clust30)
   for c in clust40, clust50, clust70, clust90, clust95, clust100:
      l = [len(_) for _ in c]
      assert ntot == sum(l)  # same total num structures
      assert len(c) > prev  # num clusters increase with sequence identity cut
      prev = len(c)
      assert l == list(reversed(sorted(l)))  # largest clusters first

def test_meta_entrytype():
   # meta.clear_pickle_cache('entrytype byentrytype'.split())
   entrytype = meta.entrytype
   byentrytype = meta.byentrytype
   assert entrytype['1PGX'] == 'prot'
   etypes = 'prot nuc prot-nuc other'.split()
   for k, v in entrytype.items():
      assert v in etypes
      assert k in byentrytype[v]

def test_meta_source():
   meta.clear_pickle_cache('source')
   assert meta.source['1PGX'] == 'STREPTOCOCCUS'

def test_meta_biotype():
   meta.clear_pickle_cache('biotype')
   bts = set(meta.biotype.values())
   assert len(bts) == 5405
   assert meta.biotype['1PGX'] == 'IMMUNOGLOBULIN BINDING PROTEIN'

def main():
   # meta.update_source_files(replace=False)
   # test_pdb_metadata()
   # test_meta_search()
   # test_meta_search_pisces_chains()
   # test_meta_pdb_compound()
   # test_meta_clust()
   # test_meta_entrytype()
   # test_meta_source()
   test_meta_biotype()

   pass

if __name__ == '__main__':
   main()