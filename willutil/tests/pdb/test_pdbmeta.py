import numpy as np
import pytest
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
   assert pdbset.issuperset({'4AVR', '6GN5', '3S9X', '4RRI', '4I6X', '6O40', '5FUI', '6E7E', '3R72', '3FIL', '4MQ3'})

   pdbset = meta.make_pdb_set(
      maxresl=1.5,
      max_seq_ident=30,
      minres=50,
      maxres=200,
      pisces_chains=False,
      entrytype='prot',
   )
   assert len(pdbset) == 1094
   assert pdbset.issuperset({'4AVR', '6GN5', '3S9X', '4RRI', '4I6X', '6O40', '5FUI', '6E7E', '3R72', '3FIL', '4MQ3'})

def test_meta_search_pisces_chains():
   pdbset = meta.make_pdb_set(maxresl=1.5, max_seq_ident=30, minres=50, maxres=200, pisces_chains=True, entrytype='prot')
   assert len(pdbset) == 1106
   assert pdbset.issuperset({'4RGDA', '1OK0A', '3X0IA', '3IMKA', '3OBQA', '1JF8A', '1Y9LA', '4BK7A', '1X8QA', '5L87A'})

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
      assert len(c) >= prev  # num clusters increase with sequence identity cut
      prev = len(c)
      assert l == list(reversed(sorted(l)))  # largest clusters first

def test_meta_entrytype():

   # # for parallel testing, only do on the main thread
   # import threading
   #     import threading
   # if threading.current_thread() is threading.main_thread():
   #     meta.clear_pickle_cachr_pickle_cache('entrytype'.split())

   entrytype = meta.entrytype
   byentrytype = meta.byentrytype
   assert entrytype['1PGX'] == 'prot'
   etypes = 'prot nuc prot-nuc other'.split()
   for k, v in entrytype.items():
      assert v in etypes
      assert k in byentrytype[v]

def test_meta_source():

   # for parallel testing, only do on the main thread
   import threading
   if threading.current_thread() is threading.main_thread():
      meta.clear_pickle_cache('source')
   assert meta.source['1PGX'] == 'STREPTOCOCCUS'

def test_meta_biotype():

   # for parallel testing, only do on the main thread
   import threading
   if threading.current_thread() is threading.main_thread():
      meta.clear_pickle_cache('biotype')
   bts = set(meta.biotype.values())
   assert len(bts) == 5405
   assert meta.biotype['1PGX'] == 'IMMUNOGLOBULIN BINDING PROTEIN'

def test_pdb_meta_strict():
   with pytest.raises(AttributeError):
      meta.this_is_not_a_real_thing

def test_meta_ligcount():

   # for parallel testing, only do on the main thread
   import threading
   if threading.current_thread() is threading.main_thread():
      meta.clear_pickle_cache('ligcount')

   df = meta.ligcount
   assert df.index[0] == 'HOH'
   # print(df['count']['CHL'])
   assert df['count']['ALA'] == 5987628
   assert df['count']['ZN'] == 42907
   assert df['count']['CHL'] == 100

def test_meta_ligpdbs():

   # for parallel testing, only do on the main thread
   import threading
   if threading.current_thread() is threading.main_thread():
      meta.clear_pickle_cache('ligpdbs')
   # print(meta.ligpdbs)
   d = set()
   for k, v in meta.ligpdbs.items():
      # print(k, v)
      # assert 0
      d.update(v)
   assert len(d) == 146116
   # print(meta.ligpdbs['CHL'])
   assert len(meta.ligpdbs['ATP']) == 2241
   assert len(meta.ligpdbs['HEM']) == 6873
   assert len(meta.ligpdbs['DOD']) == 37
   assert meta.ligpdbs['CHL'] == ['3PL9', '2X20', '6GIX', '7A4P', '6KAC', '6RHZ', '5XNL', '5XNN', '5XNO', '5XNM', '6YP7', '6YEZ', '6IGZ', '5ZJI', '7OUI', '6SL5', '1RWT', '5MDX', '4LCZ', '2BHW', '6ZZY', '6ZZX', '6ZOO', '3JCU', '6ZXS', '6YXR', '6S2Y', '6S2Z', '6QPH', '7BGI', '6L35', '6YAC', '7D0J', '4XK8', '4XK8', '7E0H', '7E0K', '7E0J', '7E0I', '7DZ7', '7DZ8', '4Y28', '1VCR', '5L8R', '7DKZ', '6JO6', '6JO5']

def test_meta_hetres():

   # for parallel testing, only do on the main thread
   import threading
   if threading.current_thread() is threading.main_thread():
      meta.clear_pickle_cache(['rescount'])
   # print(len(meta.hetres))
   assert len(meta.rescount) == 176277

   assert meta.rescount['1PGX'] == {'GLU': 6, 'LEU': 3, 'THR': 14, 'PRO': 2, 'ALA': 8, 'VAL': 9, 'TYR': 3, 'LYS': 7, 'ILE': 1, 'ASN': 3, 'GLY': 4, 'ASP': 5, 'PHE': 2, 'GLN': 1, 'TRP': 1, 'MET': 1, 'HOH': 61}

   allligs = set()
   pdbnolig = 0
   for k, v in meta.rescount.items():
      allligs.update(v.keys())
      if len(v) == 0:
         pdbnolig += 1
   print(pdbnolig)
   assert len(allligs) == 32005

   # tot = 0
   # b = dict()
   # for k, v in meta.rescount.items():
   # b[k.encode()] = [(a.encode(), b) for a, b in v.items()]
   # tot += len(v)
   # print(k)
   # print(v)
   # return
   # print(list(b.values())[0])
   # wu.save(b, 'tmp.pickle')
   # print(tot / 1000000)

#
# x.sort_values('count', ascending=False, inplace=True)
# for code, (count, natom) in x.head(100).iterrows():
#    print(code, count, natom)

def main():
   # meta.update_source_files(replace=False)
   # test_pdb_meta_strict()
   # test_pdb_metadata()
   # test_meta_search()
   # test_meta_search_pisces_chains()
   # test_meta_pdb_compound()
   # test_meta_clust()
   # test_meta_entrytype()
   # test_meta_source()
   # test_meta_biotype()
   # test_meta_ligcount()
   # test_meta_ligpdbs()
   test_meta_hetres()
   pass

if __name__ == '__main__':
   main()
