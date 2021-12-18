import os, gzip, io, glob, collections, logging, tqdm, time
import willutil as wu

log = logging.getLogger(__name__)

class PDBFile:
   def __init__(self, df, meta):
      df.reset_index(inplace=True, drop=True)
      self.df = df
      self.code = meta.code
      self.resl = meta.resl
      self.cryst1 = meta.cryst1
      self.chainseq = atomrecords_to_chainseq(df)
      self.seq = str.join('', self.chainseq.values())

   @property
   def nres(self):
      return len(self.seq)

   @property
   def nchain(self):
      return len(self.chainseq)

   def subfile(self, chain):
      df = self.df[self.df.ch == wu.misc.tobytes(chain)]
      return PDBFile(df, meta=self)

def pdb_code(fname):
   if len(fname) > 100:
      return 'none'
   fname = os.path.basename(fname)
   if len(fname.split('.')[0]) == 4:
      return fname[:4].upper()
   else:
      return 'none'

def read_pdb_atoms(fname_or_buf):
   atomlines, cryst, expdata = list(), None, None

   if wu.storage.is_pdb_fname(fname_or_buf):
      opener = gzip.open if fname_or_buf.endswith('.gz') else open
      with opener(fname_or_buf) as inp:
         contents = str(inp.read()).replace(r'\n', '\n')
   else:
      contents = fname_or_buf

   for line in contents.splitlines():
      if line.startswith(('ATOM ', 'HETATM ')):
         atomlines.append(line)
      elif line.startswith('CRYST1 '):
         assert not cryst
         cryst = line.strip()

   return '\n'.join(atomlines), wu.Bunch(cryst1=cryst)

def parse_pdb_atoms(atomstr):
   import pandas as pd
   import numpy as np

   n = 'het ai an rn ch ri rixpad x y z occ bfac elem'.split()
   w = (6, 5, 5, 4, 2, 4, 4, 8, 8, 8, 6, 6, 99)
   assert len(n) is len(w)

   df = pd.read_fwf(io.StringIO(atomstr), widths=w, names=n)
   df = df[np.logical_or(df.het == 'ATOM', df.het == 'HETATM')]
   df.het = df.het == 'HETATM'
   df.ai = df.ai.astype('i4')
   df.an = df.an.astype('S4')
   df.rn = df.rn.astype('S3')
   df.ch = df.ch.astype('S1')
   df.ri = df.ri.astype('i4')
   df.rixpad = df.rixpad.astype('S4')
   df.x = df.x.astype('f4')
   df.y = df.y.astype('f4')
   df.z = df.z.astype('f4')
   df.occ = df.occ.astype('f4')
   df.bfac = df.bfac.astype('f4')
   df.elem = df.elem.astype('S4')

   # # df = pd.DataFrame(dict(an=df.an))
   # # print(df.head(60))
   # df.reset_index()
   # print('-----------------')
   # print(df.dtypes)
   # print('-----------------')
   # print(df.memory_usage())
   # print('-----------------')
   # # print(df.loc[62])
   return df

def readpdb(fname_or_buf):
   pdbatoms, meta = read_pdb_atoms(fname_or_buf)
   df = parse_pdb_atoms(pdbatoms)
   code = pdb_code(fname_or_buf)
   resl = -1.0
   if code != 'none':
      resl = metadb = wu.pdb.pdbmeta.resl[code]
   meta.update(code=code, resl=resl)
   return PDBFile(df, meta)

def format_atom(atomi=0, atomn='ATOM', idx=' ', resn='RES', chain='A', resi=0, insert=' ', x=0,
                y=0, z=0, occ=1, b=0):
   return _atom_record_format.format(**locals())

def atomrecords_to_chainseq(df):
   seq = collections.defaultdict(list)

   prevri = 123456789
   prevhet = False
   for i in range(len(df)):
      ri = df.ri[i]
      if ri == prevri: continue
      rn = df.rn[i]
      ch = df.ch[i]
      rn = rn.decode() if isinstance(rn, bytes) else rn
      ch = ch.decode() if isinstance(ch, bytes) else ch
      het = df.het[i]
      if not het and not prevhet:
         # mark missing sequence residues if not HET
         for _ in range(prevri + 1, ri):
            seq[ch].append('-')
      prevri = ri
      prevhet = het
      if het:
         if not rn == 'HOH':
            seq[ch].append('Z')
         continue
      try:

         seq[ch].append(wu.chemical.aa321[rn])
      except KeyError:
         seq[ch].append('X')
   return {c: str.join('', s) for c, s in seq.items()}

def find_pdb_files(files_or_pattern, maxsize=99e99):
   if isinstance(files_or_pattern, str):
      files_or_pattern = [files_or_pattern]
   candidates = list()
   for f in files_or_pattern:
      if not os.path.exists(f):
         candidates.extend(glob.glob(f))
      else:
         candidates.append(f)
   files = list()
   for f in candidates:
      if os.path.getsize(f) > maxsize:
         continue
      if not os.path.exists(f):
         raise ValueError(f'pdb file {f} does not exist')
      files.append(f)
   return files

def load_pdbs(files_or_pattern, cache=True, skip_errors=False, pbar=True, **kw):
   files = find_pdb_files(files_or_pattern, **kw)
   pdbs = dict()
   for fname in (tqdm.tqdm(files) if pbar else files):
      t = time.perf_counter()
      try:
         fname = fname.replace('.pickle', '')
         if cache:
            try:
               pdbfile = wu.load(fname + '.pickle')
               log.info(f'loaded {fname + ".pickle"}')
            except (FileNotFoundError, EOFError, AttributeError) as e:
               if not isinstance(e, FileNotFoundError):
                  log.warning(f'cache failure, loading {fname}')
               else:
                  log.info(f'cache failure, loading {fname}')
               pdbfile = readpdb(fname)
               wu.save(pdbfile, fname + '.pickle')
         else:
            log.info(f'loading {fname}')
            pdbfile = readpdb(fname)
         pdbs[fname] = pdbfile
         if time.perf_counter() - t > 0.1:
            print(fname)
      except (FileNotFoundError, ValueError) as e:
         if not skip_errors:
            raise e

   return pdbs
