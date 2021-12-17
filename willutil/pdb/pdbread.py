import os, gzip, io, glob

import willutil as wu

class PDBFile:
   def __init__(self, df, meta):
      self.df = df
      self.code = meta.code
      self.resl = meta.resl
      self.sequence = atomrecords_to_sequence(df)
      self.meta = meta

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

   return '\n'.join(atomlines), wu.Bunch(cryst=cryst)

def parse_pdb_atoms(atomstr):
   from pandas import read_fwf

   n = 'het ai an rn ch ri x y z occ bfac elem'.split()
   w = (6, 5, 5, 4, 2, 4, 12, 8, 8, 6, 6, 99)
   assert len(n) is len(w)

   df = read_fwf(io.StringIO(atomstr), widths=w, names=n)
   # df = df[np.logical_or(df.het == 'ATOM', df.het == 'HETATM')]
   df.het = df.het == 'HETATM'
   df.ai = df.ai.astype('i4')
   # df.an = df.an.astype('S4')  # f*ck you, pandas!
   # df.rn = df.rn.astype('S3')  # f*ck you, pandas!
   # df.ch = df.ch.astype('S1')  # f*ck you, pandas!
   df.ri = df.ri.astype('i4')
   df.x = df.x.astype('f4')
   df.y = df.y.astype('f4')
   df.z = df.z.astype('f4')
   df.occ = df.occ.astype('f4')
   df.bfac = df.bfac.astype('f4')
   # df.elem = df.elem.astype('S4')  # f*ck you, pandas!
   return df

def readpdb(fname_or_buf):
   pdbatoms, meta = read_pdb_atoms(fname_or_buf)
   df = parse_pdb_atoms(pdbatoms)
   code = pdb_code(fname_or_buf)
   resl = -1.0
   if code != 'none':
      metadb = wu.pdb.pdb_metadata()
      resl = metadb.resl[code]
   meta.update(code=code, resl=resl)

   return PDBFile(df, meta)

def format_atom(atomi=0, atomn='ATOM', idx=' ', resn='RES', chain='A', resi=0, insert=' ', x=0,
                y=0, z=0, occ=1, b=0):
   return _atom_record_format.format(**locals())

def atomrecords_to_sequence(df):
   seq = list()

   prevri = 123456789
   prevhet = False
   for i in range(len(df)):
      ri = df.ri[i]
      if ri == prevri: continue
      rn = df.rn[i]
      het = df.het[i]
      if not het and not prevhet:
         # mark missing sequence residues if not HET
         for _ in range(prevri + 1, ri):
            seq.append('-')
      prevri = ri
      prevhet = het
      if het:
         if not rn == 'HOH':
            seq.append('Z')
         continue
      try:
         seq.append(wu.chemical.aa321[rn])
      except KeyError:
         seq.append('X')
   return str.join('', seq)

def find_pdb_files(files_or_pattern):
   if isinstance(files_or_pattern, str):
      files = glob.glob(files_or_pattern)
   else:
      files = files_or_pattern
   for f in files:
      if not os.path.exists(f):
         raise ValueError(f'pdb file {f} does not exist')
   return files

def load_pdbs(files_or_pattern, max_seq_identity=0.3):
   files = find_pdb_files(files_or_pattern)
   pdbs = list()
   for f in files:
      pdbs.append(readpdb(f))
   return pdbs
