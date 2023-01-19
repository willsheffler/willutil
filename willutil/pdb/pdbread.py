import os, gzip, io, glob, collections, logging, tqdm, time
import numpy as np
import willutil as wu

log = logging.getLogger(__name__)

class PDBFile:
   def __init__(
      self,
      df,
      meta,
      original_contents,
      renumber_by_model=True,
      renumber_from_0=False,
      remove_het=False,
   ):
      self.init(df, meta, original_contents, renumber_by_model, renumber_from_0)

   def init(self, df, meta, original_contents, renumber_by_model=False, renumber_from_0=False, remove_het=False):
      self.original_contents = original_contents
      self.meta = meta.copy()
      self.code = meta.code
      self.resl = meta.resl
      self.cryst1 = meta.cryst1
      df = df.copy()
      df.reset_index(inplace=True, drop=True)
      self.df = df
      if renumber_by_model:
         self.renumber_by_model()
      if renumber_from_0:
         self.renumber_from_0()
      if remove_het:
         self.remove_het()
      self.chainseq = atomrecords_to_chainseq(df)
      self.seqhet = str.join('', self.chainseq.values())
      self.seq = self.seqhet.replace('Z', '')
      # ic(self.seq)
      self.nres = len(self.seq)
      self.nreshet = len(self.seqhet)
      self.nchain = len(self.chainseq)
      self.fname = meta.fname
      self.aamask = self.atommask('CA', aaonly=False)

   def copy(self, **kw):
      return PDBFile(self.df, self.meta, self.original_contents, **kw)

   def __getattr__(self, k):
      'allow dot access for fields of self.df from self'
      if k == 'df':
         raise AttributeError
      elif k in self.df.columns:
         return getattr(self.df, k)
      else:
         raise AttributeError(k)

   def renumber_by_model(self):
      ri_per_model = np.max(self.df.ri) - np.min(self.df.ri) + 1
      ai_per_model = np.max(self.df.ai) - np.min(self.df.ai) + 1
      for m in self.models():
         i = self.modelidx(m)
         idx = self.df.mdl == m
         self.df.ri += np.where(idx, i * ri_per_model, 0)
         self.df.ai += np.where(idx, i * ai_per_model, 0)
      return self

   def natom(self):
      return len(self.df)

   def getres(self, ri):
      r = self.df[self.df.ri == ri].copy()
      r.reset_index(inplace=True, drop=True)
      return r

   def xyz(self, ir, ia):
      r = self.getres(ir)
      if isinstance(ia, int):
         return r.x[ia], r.y[ia], r.z[ia]
      if isinstance(ia, str):
         ia = ia.encode()
      if isinstance(ia, bytes):
         return float(r.x[r.an == ia]), float(r.y[r.an == ia]), float(r.z[r.an == ia])
      raise ValueError(ia)

   def renumber_from_0(self):
      assert np.all(self.het == np.sort(self.het))
      d = {ri: i for i, ri in enumerate(np.unique(self.ri))}
      self.df['ri'] = [d[ri] for ri in self.df['ri']]
      return self

   def remove_het(self):
      self.subset(het=False, inplace=True)
      return self

   def subset(
      self,
      chain=None,
      het=None,
      removeres=None,
      atomnames=[],
      chains=[],
      model=None,
      modelidx=None,
      inplace=False,
      removeatoms=[],
   ):
      import numpy as np
      import pandas as pd
      df = self.df
      if chain is not None:
         if isinstance(chain, int):
            chain = list(self.chainseq.keys())[chain]
         df = df.loc[df.ch == wu.misc.tobytes(chain)]
         # have no idea why, but dataframe gets corrupted  without this
         df = pd.DataFrame(df.to_dict())
         assert len(df) > 0
      if het is False:
         df = df.loc[df.het == False]
         df = pd.DataFrame(df.to_dict())
      if het is True:
         df = df.loc[df.het == True]
         df = pd.DataFrame(df.to_dict())
      if removeres is not None:
         if isinstance(removeres, (str, bytes)):
            removeres = [removeres]
         for res in removeres:
            res = wu.misc.tobytes(res)
            df = df.loc[df.rn != res]
            df = pd.DataFrame(df.to_dict())
      if atomnames:
         atomnames = [a.encode() for a in atomnames]
         df = df.loc[np.isin(df.an, atomnames)]
         df = pd.DataFrame(df.to_dict())
      if chains:
         if isinstance(chains, str) and len(chains) == 1: chains = [chains]
         chains = [c.encode() for c in chains]
         df = df.loc[np.isin(df.ch, chains)]
         df = pd.DataFrame(df.to_dict())
      if model is not None:
         df = df.loc[df.mdl == model]
         df = pd.DataFrame(df.to_dict())
      if modelidx is not None:
         df = df.loc[df.mdl == self.models()[modelidx]]
         df = pd.DataFrame(df.to_dict())
      if removeatoms:
         idx = np.isin(df.ai, removeatoms)
         # ic(df.loc[idx].an)
         # ic(df.loc[idx].ri)
         df = df.loc[~idx]
         df = pd.DataFrame(df.to_dict())

      df.reset_index(inplace=True, drop=True)
      assert len(df) > 0

      if inplace:
         self.init(
            df,
            self.meta,
            original_contents=self.original_contents,
            renumber_by_model=True,
         )
         return self
      else:
         return PDBFile(df, meta=self.meta, original_contents=self.original_contents)

   def isonlyaa(self):
      return np.sum(self.het) == 0

   def isonlyhet(self):
      return np.sum(self.het) == len(self.df)

   def models(self):
      return list(np.sort(np.unique(self.df.mdl)))

   def modelidx(self, m):
      models = self.models()
      return models.index(m)

   def atommask(self, atomname, aaonly=True):
      if not isinstance(atomname, (str, bytes)):
         return np.stack([self.atommask(a) for a in atomname]).T
      an = atomname.encode() if isinstance(atomname, str) else atomname
      an = an.upper()
      mask = list()
      for i, (ri, g) in enumerate(self.df.groupby(['ri', 'ch'])):
         assert np.sum(g.an == an) <= 1
         # assert np.sum(g.an == an) <= np.sum(g.an == b'CA') # e.g. O in HOH
         hasatom = np.sum(g.an == an) > 0
         mask.append(hasatom)
      mask = np.array(mask, dtype=bool)
      if aaonly:
         aaonly = self.aamask
         mask = mask[aaonly]
      return mask

   def coords(self, atomname=['N', 'CA', 'C', 'O', 'CB'], aaonly=True):
      if atomname is None:
         atomname = self.df.an.unique()
      if not self.isonlyaa():
         self = self.subset(het=False)  # sketchy?
      if not isinstance(atomname, (str, bytes)):
         coords, masks = zip(*[self.coords(a) for a in atomname])
         # ic(len(coords))
         # ic([len(_) for _ in coords])
         coords = np.stack(coords).swapaxes(0, 1)
         # ic(coords.shape)
         masks = np.stack(masks).T
         return coords, masks
      an = atomname.encode() if isinstance(atomname, str) else atomname
      an = an.upper().strip()
      mask = self.atommask(an)
      df = self.df
      idx = self.df.an == an
      df = df.loc[idx]
      xyz = np.stack([df['x'], df['y'], df['z']]).T
      if np.sum(~mask) > 0:
         coords = 9e9 * np.ones((len(mask), 3))
         coords[mask] = xyz
         xyz = coords
      return xyz, mask

   def camask(self, aaonly=False):
      return self.atommask('ca', aaonly=aaonly)
      # return np.array([np.any(g.an == b'CA') for i, g in self.df.groupby(self.df.ri)])

   def cbmask(self, aaonly=True):
      return self.atommask('cb', aaonly=aaonly)
      # mask = list()
      # for i, (ri, g) in enumerate(self.df.groupby(self.df.ri)):
      #    assert np.sum(g.an == b'CB') <= 1
      #    assert np.sum(g.an == b'CB') <= np.sum(g.an == b'CA')
      #    hascb = np.sum(g.an == b'CB') > 0
      #    mask.append(hascb)
      # mask = np.array(mask)
      # if aaonly:
      #    aaonly = self.aamask
      #    # ic(aaonly)
      #    mask = mask[aaonly]
      # return mask

   def bb(self):
      crd, mask = self.coords('n ca c o cb'.split())
      return crd

   def ca(self):
      crd, mask = self.coords('ca')
      return crd

   def ncac(self):
      crd, mask = self.coords('n ca c'.split())
      return crd
      # pdb = self.subset(het=False, atomnames=['N', 'CA', 'C'])
      # xyz = np.stack([pdb.df['x'], pdb.df['y'], pdb.df['z']]).T.reshape(-1, 3, 3)
      # return xyz

   def ncaco(self):
      crd, mask = self.coords('n ca c o'.split())
      return crd
      # pdb = self.subset(het=False, atomnames=['N', 'CA', 'C', 'O'])
      # xyz = np.stack([pdb.df['x'], pdb.df['y'], pdb.df['z']])
      # xyz = xyz.T.reshape(-1, 4, 3)
      # return xyz

   def sequence(self):
      return self.seq

   def chain(self, ires):
      ires = ires - 1
      for i, (ch, seq) in enumerate(self.chainseq.items()):
         if ires >= len(seq):
            ires -= len(seq)
         else:
            return i + 1
      else:
         return None
      return self

   def num_chains(self):
      return len(self.chainseq)

def pdb_code(fname):
   if len(fname) > 100:
      return 'none'
   fname = os.path.basename(fname)
   if len(fname.split('.')[0]) == 4:
      return fname[:4].upper()
   else:
      return 'none'

def read_pdb_atoms(fname_or_buf):
   atomlines, meta = dict(), wu.Bunch(fname=None, cryst1=None)

   if wu.storage.is_pdb_fname(fname_or_buf):
      meta.fname = fname_or_buf
      opener = gzip.open if fname_or_buf.endswith('.gz') else open
      with opener(fname_or_buf) as inp:
         contents = str(inp.read()).replace(r'\n', '\n')
         # contents = str(inp.read())
   else:
      contents = fname_or_buf
      if contents.count('ATOM  ') == 0 and contents.count('HETATM') == 0:
         raise ValueError(f'bad pdb: {contents}')
   if contents.startswith("b'"):
      contents = contents[2:]

   modelnum = 0
   atomlines[0] = list()
   for i, line in enumerate(contents.splitlines()):

      if line.startswith(('ATOM', 'HETATM')):
         atomlines[modelnum].append(line)
      elif line.startswith('MODEL '):
         modelnum = int(line[6:])
         atomlines[modelnum] = list()
      elif line.startswith('CRYST1 '):
         assert not meta.cryst1
         meta.cryst1 = line.strip()

   # ic(len(atomlines))
   assert atomlines

   return {k: '\n'.join(v) for k, v in atomlines.items()}, meta, contents

def parse_pdb_atoms(atomstr):
   import pandas as pd
   import numpy as np
   assert (atomstr)

   dt = pdbcoldtypes.copy()
   del dt['het']
   del dt['ai']
   cr = pdbcolrange.copy()
   cr[0] = 0, cr[1][1]
   cr[1] = 0, cr[1][1]
   converters = dict(
      het=lambda x: x.startswith('HETATM'),
      # logic below allows entries line 'ATOM 123456'
      ai=lambda x: np.int32(x[4:]) if x.startswith('ATOM') else np.int32(x[6:]),
   )

   mdf = dict()
   for m in atomstr.keys():
      df = pd.read_fwf(
         io.StringIO(atomstr[m]),
         names=pdbcolnames,
         colspecs=cr,
         header=None,
         dtype=dt,
         converters=converters,
         na_filter=False,
      )
      # don't understand why pandas doesn't respect the str dtypes from "dt"
      df.an = df.an.astype('a4')
      df.al = df.al.astype('a1')
      df.ch = df.ch.astype('a1')
      df.rn = df.rn.astype('a3')
      # df.rins = df.rins.astype('a1')
      # df.seg = df.seg.astype('a4')
      df.elem = df.elem.astype('a2')
      # df.charge = df.charge.astype('a2')
      # print(df.dtypesb)
      # print(df.memory_usage())

      notalt = np.logical_or(df.al == b'', df.al == b'A')
      df = df[notalt]
      df.drop('al', axis=1, inplace=True)
      mdf[m] = df
   return mdf

def concatenate_models(df):
   import pandas as pd
   assert isinstance(df, dict)
   df = {k: v for k, v in df.items() if len(v)}
   for m, d in df.items():
      d['mdl'] = m
   df = pd.concat(df.values())
   return df

def readpdb(fname_or_buf, indatabase=False):
   pdbatoms, meta, original_contents = read_pdb_atoms(fname_or_buf)
   df = parse_pdb_atoms(pdbatoms)
   code = pdb_code(fname_or_buf) if indatabase else 'none'
   resl = -1.0
   if code != 'none':
      resl = metadb = wu.pdb.pdbmeta.resl[code]
   meta.update(code=code, resl=resl)
   df = concatenate_models(df)
   pdb = PDBFile(df, meta, original_contents)
   return pdb

def format_atom(atomi=0, atomn='ATOM', idx=' ', resn='RES', chain='A', resi=0, insert=' ', x=0, y=0, z=0, occ=1, b=0):
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

         seq[ch].append(wu.chem.aa321[rn])
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

def load_pdb(
   fname,
   cache=True,
):
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
   return pdbfile

def load_pdbs(
   files_or_pattern,
   cache=True,
   skip_errors=False,
   pbar=True,
   maxfiles=9e9,
   **kw,
):
   files = find_pdb_files(files_or_pattern, **kw)
   pdbs = dict()
   for fname in (tqdm.tqdm(files) if pbar else files):
      t = time.perf_counter()
      try:
         pdbs[fname] = load_pdb(fname, cache)
         if len(pdbs) >= maxfiles:
            break
      except (FileNotFoundError, ValueError) as e:
         if not skip_errors:
            raise e

   return pdbs

def gen_pdbs(
   files_or_pattern,
   cache=True,
   skip_errors=False,
   pbar=True,
   maxfiles=9e9,
   **kw,
):
   files = find_pdb_files(files_or_pattern, **kw)
   pdbs = dict()
   for fname in (tqdm.tqdm(files) if pbar else files):
      t = time.perf_counter()
      try:
         yield fname, load_pdb(fname, cache)
         if len(pdbs) >= maxfiles:
            break
      except (FileNotFoundError, ValueError) as e:
         if not skip_errors:
            raise e

# pdb format
# COLUMNS        DATA TYPE       CONTENTS
# --------------------------------------------------------------------------------
#  1 -  6        Record name     "ATOM  "
#  7 - 11        Integer         Atom serial number.
# 13 - 16        Atom            Atom name.
# 17             Character       Alternate location indicator.
# 18 - 20        Residue name    Residue name.
# 22             Character       Chain identifier.
# 23 - 26        Integer         Residue sequence number.
# 27             AChar           Code for insertion of residues.
# 31 - 38        Real(8.3)       Orthogonal coordinates for X in Angstroms.
# 39 - 46        Real(8.3)       Orthogonal coordinates for Y in Angstroms.
# 47 - 54        Real(8.3)       Orthogonal coordinates for Z in Angstroms.
# 55 - 60        Real(6.2)       Occupancy.
# 61 - 66        Real(6.2)       Temperature factor (Default = 0.0).
# 73 - 76        LString(4)      Segment identifier, left-justified.
# 77 - 78        LString(2)      Element symbol, right-justified.
# 79 - 80        LString(2)      Charge on the atom.

pdbcolnames = [
   'het',  # "ATOM  "                   
   'ai',  # Atom serial number.        
   'an',  # Atom name.                 
   'al',  # Alternate location indicato
   'rn',  # Residue name.              
   'ch',  # Chain identifier.          
   'ri',  # Residue sequence number.   
   # 'rins',  # Code for insertion of resid
   'x',  # Orthogonal coordinates for 
   'y',  # Orthogonal coordinates for 
   'z',  # Orthogonal coordinates for 
   'occ',  # Occupancy.                 
   'bfac',  # Temperature factor (Default
   # 'seg',  # Segment identifier, left-ju
   'elem',  # Element symbol, right-justi
   # 'charge',  # Charge on the atom.
]

pdbcolrange = [
   (1 - 1, 6),
   (7 - 1, 11),
   (13 - 1, 16),
   (17 - 1, 17),
   (18 - 1, 20),
   (22 - 1, 22),
   (23 - 1, 26),
   # (27 - 1, 27),
   (31 - 1, 38),
   (39 - 1, 46),
   (47 - 1, 54),
   (55 - 1, 60),
   (61 - 1, 66),
   # (73 - 1, 76),
   (77 - 1, 78),
   # (79 - 1, 80),
]

pdbcoldtypes = dict(
   het='b',  # Record name
   ai='i4',  # Integer
   an='a4',  # Atom
   al='a1',  # Character
   rn='a3',  # Residue name
   ch='a1',  # Character
   ri='i4',  # Integer
   # rins='a1',  # AChar
   x='f4',  # Real(8.3)
   y='f4',  # Real(8.3)
   z='f4',  # Real(8.3)
   occ='f4',  # Real(6.2)
   bfac='f4',  # Real(6.2)
   # seg='a4',  # Lstring(4)
   elem='a2',  # Lstring(2)
   # charge='a2',  # LString(2)
)
