import urllib.request, logging
import willutil as wu

log = logging.getLogger(__name__)

_pdbmeta = None

def pdb_metadata():
   global _pdbmeta
   if not _pdbmeta:
      _pdbmeta = wu.storage.load_package_data('pdb/pdb_meta.pickle')

   return _pdbmeta

def get_pdb_resl(code):
   if code not in pdbresl:
      raise ValueError('no resl info for pdb code ' + code)
   return pdb_metadata().resl[code]

def update_local_pdb_metadata():
   meta = fetch_pdb_metadata()
   wu.storage.save_package_data(meta, 'pdb/pdb_meta.pickle')
   meta2 = wu.storage.load_package_data('pdb/pdb_meta.pickle')
   assert meta == meta2

def fetch_pdb_metadata(limit=None, timeout=10):
   url = 'https://ftp.wwpdb.org/pub/pdb/derived_data/index/resolu.idx'
   log.info('updating pdb resl from', url)
   pdbresl = dict()
   with urllib.request.urlopen(url, timeout=timeout) as inp:
      count = 0
      countnoresl = 0
      for line in inp:
         count += 1
         if count < 7:
            continue
         if limit and count - 6 > limit:
            break
         splt = line.split()
         code = splt[0].decode()
         assert isinstance(code, str)
         if code in pdbresl:
            log.debug(f'duplicate code {code}')
         assert len(code) == 4
         if len(splt) == 3:
            resl = float(splt[2])
         else:
            resl = -1
            log.debug(f'bad resolu.idx line {line.strip().decode()}')
         if resl == -1: countnoresl += 1
         pdbresl[code] = resl
   return wu.Bunch(resl=pdbresl)

# update_local_pdb_metadata()
