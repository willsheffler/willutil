import sys, os, urllib.request, logging, gzip, lzma, collections, tqdm
from Bio import pairwise2

import willutil as wu

log = logging.getLogger(__name__)

class PDBSearchResult:
   def __init__(self, pdbs):
      self.pdbs = pdbs

class PDBMetadata:
   __all__ = list(set(vars().keys()) - {'__module__', '__qualname__'})

   @property
   def resl(self):
      return self._load_cached('resl', self.load_pdb_resl_data)

   @property
   def xtal(self):
      return self._load_cached('xtal', self.load_pdb_xtal_data)

   @property
   def chainseq(self):
      return self._load_cached('chainseq', self.load_pdb_seqres_data)

   @property
   def seq(self):
      return self._load_cached('seq', self.get_full_seq)

   @property
   def nres(self):
      return self._load_cached('nres', self.get_nres)

   def make_pdb_set(
      self,
      maxresl=2.0,
      minres=50,
      maxres=500,
      max_seq_ident=0.5,
      pisces_chains=True,
   ):
      # print(minres, maxres, maxres, max_seq_ident)
      piscesdf = wu.pdb.get_pisces_set(maxresl, max_seq_ident)
      pisces = set(_.decode() for _ in piscesdf.code)
      if max_seq_ident <= 1.0:
         max_seq_ident *= 100

      maxresok = set(self.nres.index[(self.nres <= maxres)])
      minresok = set(self.nres.index[(self.nres >= minres)])
      reslok = set(self.resl.index[self.resl <= maxresl])
      allok = minresok.intersection(maxresok.intersection(reslok))
      hits = allok.intersection(pisces)

      print('==== make_pdb_set stats ====')
      print('maxresok', len(maxresok))
      print('minresok', len(minresok))
      print('reslok', len(reslok))
      print('allok', len(allok))
      print('pisces', len(pisces))
      print('hits', len(hits))
      print('============================')

      if pisces_chains:
         # return all pisces chains rather than pdb codes
         hits = {h.encode() for h in hits}
         chains = piscesdf.chain.unique()
         pdbchains = set(piscesdf.PDBchain)
         chainhits = set()
         for c in chains:
            chits = set([_ + c for _ in hits])
            chits &= pdbchains
            chainhits.update(chits)
         hits = {h.decode() for h in chainhits}

      return hits

   def __init__(self):
      self.urls = wu.Bunch(
         author='https://ftp.wwpdb.org/pub/pdb/derived_data/index/author.idx',
         compound='https://ftp.wwpdb.org/pub/pdb/derived_data/index/compound.idx',
         resl='https://ftp.wwpdb.org/pub/pdb/derived_data/index/resolu.idx',
         xtal='https://ftp.wwpdb.org/pub/pdb/derived_data/index/crystal.idx',
         entries='https://ftp.wwpdb.org/pub/pdb/derived_data/index/entries.idx',
         onhold='https://ftp.wwpdb.org/pub/pdb/derived_data/index/on_hold.list',
         entrytypes='https://ftp.wwpdb.org/pub/pdb/derived_data/pdb_entry_type.txt',
         seqres='https://ftp.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz',
         source='https://ftp.wwpdb.org/pub/pdb/derived_data/index/source.idx',
      )
      self.metadata = wu.Bunch()

   def get_full_seq(self):
      chainseq = self.chainseq
      seq = dict()
      for code, seqs in chainseq.items():
         seq[code] = str.join('', seqs.values())
      return seq

   def _load_cached(self, name, loadfunc):
      if not name in self.metadata:
         try:
            val = wu.load_package_data(f'pdb/meta/{name}.pickle')
         except FileNotFoundError:
            val = loadfunc()
            wu.storage.save_package_data(val, f'pdb/meta/{name}.pickle')
         self.metadata[name] = val
      return self.metadata[name]

   def update_source_files(self):
      for name, url in self.urls.items():
         fname = wu.storage.package_data_path(f'pdb/meta/{name}.txt')
         if name == 'seqres':
            fname += '.gz'
         urllib.request.urlretrieve(url, fname)
         log.info(f'downloading {fname}')
         assert os.path.exists(fname)
         fn = wu.storage.package_data_path(f'pdb/meta/seqres.txt')
      assert os.path.exists(fn + '.xz')
      with gzip.open(fn + '.gz') as inp:
         with lzma.open(fn + '.xz', 'wb') as out:
            out.write(inp.read())
      os.remove(fn + '.gz')

      for name in ('author', 'compound', 'resl', 'xtal', 'entries', 'onhold', 'entrytypes',
                   'source'):
         fname = wu.storage.package_data_path(f'pdb/meta/{name}.txt')
         lod.info(f'running xz {fname}')
         os.system(f'xz {fname}')

   def load_pdb_xtal_data(self):
      xtal = dict()
      fname = wu.storage.package_data_path('pdb/meta/xtal.txt.xz')
      count = 0
      with wu.storage.open_lzma_cached(fname) as inp:
         for line in inp:
            count += 1
            if count < 5:
               continue
            line = line.decode().strip()
            code = line[:4]
            cryst1 = line[5:].strip()
            xtal[code] = cryst1
      return xtal

   def load_pdb_seqres_data(self):
      pdbseq = collections.defaultdict(dict)
      fname = wu.storage.package_data_path('pdb/meta/seqres.txt.xz')
      pdb = None
      with wu.storage.open_lzma_cached(fname) as inp:
         for line in inp:
            line = line.decode()
            if line.startswith('>'):
               pdb = line[1:7]
               # print(pdb)
            else:
               code = pdb[:4].upper()
               chain = pdb[5]
               pdbseq[code][chain] = line.strip()
      return pdbseq

   def get_nres(self):
      import pandas as pd, numpy as np
      nres = {k: len(v) for k, v in self.seq.items()}
      nres = pd.Series(nres)
      # nres = nres[np.argsort(nres)]
      return nres

   def load_pdb_resl_data(self):
      pdbresl = dict()
      fname = wu.storage.package_data_path('pdb/meta/resl.txt.xz')
      with wu.storage.open_lzma_cached(fname) as inp:
         count = 0
         countnoresl = 0
         for line in inp:
            line = line.decode()
            count += 1
            if count < 7:
               continue
            splt = line.split()
            code = splt[0]
            assert isinstance(code, str)
            if code in pdbresl:
               log.debug(f'duplicate code {code}')
            assert len(code) == 4
            if len(splt) == 3:
               resl = float(splt[2])
            else:
               resl = -1
               log.debug(f'bad resolu.idx line {line.strip()}')
            if resl == -1:
               countnoresl += 1
               resl = 9e9
            pdbresl[code] = resl
      import pandas as pd, numpy as np
      pdbresl = pd.Series(pdbresl)
      # pdbresl = pdbresl[np.argsort(pdbresl)]
      return pdbresl

sys.modules[__name__] = PDBMetadata()
