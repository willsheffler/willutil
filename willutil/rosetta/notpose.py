import numpy as np
import willutil as wu

class NotPose:
   #@timed
   def __init__(self, fname=None, coords=None, **kw):
      if coords is None:
         _init_NotPose_pdb(self, fname, **kw)
      else:
         assert fname is None
         _init_NotPose_coords(self, coords, **kw)

   def __len__(self):
      return self.size()

   def size(self):
      return self.nres

   def sequence(self):
      return self.seq

   def secstruct(self):
      return self.ss

   def chain(self, ires):
      return self._chain[ires - 1]

   def extract(self, chain, **kw):
      if self.coordsonly:
         c = np.array(list(self._chain))
         if isinstance(chain, int):
            chain = np.unique(c)[chain]
         w = c == chain
         return NotPose(coords=self.coords[w], chain=chain, **kw)
      else:
         return NotPose(self.fname, pdb=self.pdb, chain=chain, **kw)

   def pdb_info(self):
      return self.info

   def ncaco(self):
      return self.bbcoords

   def residue(self, ir):
      return NotResidue(self, ir)

   def get_sc_coords(self, which_resi=None, recenter_input=False, **kw):
      assert which_resi is None
      assert recenter_input == False
      # kw = Bunch(kw, _strict=False)
      if which_resi is None:
         which_resi = list(range(1, self.size() + 1))
      resaname, resacrd = list(), list()
      for ir in which_resi:
         r = self.residue(ir)
         if not r.is_protein():
            raise ValueError("non-protein residue %s at position %i" % (r.name(), ir))
         anames, crd = list(), list()
         for ia in range(r.natoms()):
            anames.append(r.atom_name(ia + 1))
            xyz = r.xyz(ia + 1)
            crd.append([xyz.x, xyz.y, xyz.z])
         resaname.append(anames)
         hcrd = np.ones((len(anames), 4), dtype='f4')
         hcrd[:, :3] = np.array(crd)
         resacrd.append(hcrd)
      # if recenter_input:
      #    bb = get_bb_coords(self, which_resi, **kw.sub(recenter_input=False))
      #    cen = np.mean(bb.reshape(-1, 4)[:, :3], 0)
      #    for xyz in resacrd:
      #       xyz[:, :3] -= cen
      return resaname, resacrd

class NotResidue:
   def __init__(self, nopo, ir):
      self.nopo = nopo
      self.ir = ir - 1
      self.rdf = None if nopo.coordsonly else nopo.pdb.getres(self.ir)
      self.anames = ['N', 'CA', 'C', 'O', 'CB'][:nopo.coords.shape[1]] if nopo.coordsonly else None
      self.anamemap = dict(N=0, CA=1, C=2, O=3, CB=4)

   def xyz(self, ia):

      if isinstance(ia, int):
         ia -= 1
         if self.nopo.coordsonly:
            return NotXYZ(self.nopo.ncaco[self.ir, ia])
      if isinstance(ia, int):
         xyz = self.rdf.x[ia], self.rdf.y[ia], self.rdf.z[ia]
      if isinstance(ia, str):
         if self.nopo.coordsonly:
            if ia in self.anamemap:
               return NotXYZ(self.nopo.ncaco[self.ir, self.anamemap[ia]])
         ia = ia.encode()
      if isinstance(ia, bytes):
         if self.nopo.coordsonly:
            return self.nopo.coords[self.ir, self.anamemap[ia.decode()]]
         xyz = (
            float(self.rdf.x[self.rdf.an == ia]),
            float(self.rdf.y[self.rdf.an == ia]),
            float(self.rdf.z[self.rdf.an == ia]),
         )

      return NotXYZ(xyz)
      raise ValueError(ia)

   def has(self, aname):
      if self.nopo.coordsonly:
         return aname in ['N', 'CA', 'C', 'O']
      # could be more efficient
      if isinstance(aname, str):
         aname = aname.encode()
      return aname in set(self.rdf.an)

   def is_protein(self):
      return self.nopo.camask[self.ir]

   def natoms(self):
      return self.nheavyatoms()

   def nheavyatoms(self):
      if self.nopo.coordsonly: return 4
      return len(self.rdf)

   def atom_name(self, ia):
      if self.nopo.coordsonly:
         return ['N', 'CA', 'C', 'O'][ia - 1]
      r = self.rdf
      aname = r.an[ia - 1]
      return aname.decode()

   def name(self):
      if self.nopo.coordsonly:
         return self.nopo.seq[ir]
      r = self.rdf.rn[0]
      return r.decode()

class NotXYZ(list):
   def __init__(self, xyz):
      super().__init__(xyz)
      self.x = self[0]
      self.y = self[1]
      self.z = self[2]

class NotPDBInfo:
   """Mimicks rosetta PDBInfo class"""
   def __init__(self, nopo):
      self.nopo = nopo

   def name(self):
      return self.nopo.name

   def crystinfo(self):
      return self.nopo.crystinfo

class CrystInfo:
   """mimicks rosetta CrystInfo class"""
   @classmethod
   def from_cryst1(cls, cryst1):
      if cryst1 is None: return None
      s = cryst1.split()
      a, b, c, alpha, beta, gamma = (float(x) for x in s[1:7])
      spacegroup = ' '.join(s[7:])
      ci = CrystInfo(a, b, c, alpha, beta, gamma, spacegroup)
      # ic(ci.cryst1())
      return ci

   def __init__(self, a, b, c, alpha, beta, gamma, spacegroup):
      super(CrystInfo, self).__init__()
      self._A = a
      self._B = b
      self._C = c
      self._alpha = alpha
      self._beta = beta
      self._gamma = gamma
      self._spacegroup = spacegroup

   def spacegroup(self):
      return self._spacegroup

   def A(self):
      return self._A

   def B(self):
      return self._B

   def C(self):
      return self._C

   def alpha(self):
      return self._alpha

   def beta(self):
      return self._beta

   def gamma(self):
      return self._gamma

   def cryst1(self):
      return wu.sym.cryst1_pattern_full % (
         self.A(),
         self.B(),
         self.C(),
         self.alpha(),
         self.beta(),
         self.gamma(),
         self.spacegroup(),
      )

def _init_NotPose_pdb(self, fname=None, pdb=None, chain=None, secstruct=None, **kw):
   self.fname = fname

   if pdb is None:
      ic(fname)
      assert isinstance(fname, str)
      pdb = wu.pdb.readpdb(fname, **kw)
   self.rawpdb = pdb
   if chain is not None:
      pdb = self.rawpdb.subset(chain=chain)
   self.pdb = pdb.subset(het=False)
   self.pdbhet = pdb.subset(het=True)
   # self.bbcoords = wu.hpoint(self.pdb.bb(**kw))
   try:
      ncaco, _mask = self.pdb.atomcoords(['n', 'ca', 'c', 'o'], nomask=True, **kw)
   except ValueError:
      ncaco, _mask = self.pdb.atomcoords(['n', 'ca', 'c'], nomask=True, **kw)
      ncaco = wu.chem.add_bb_o_guess(ncaco)
   self.ncaco = wu.hpoint(ncaco)
   self.ncac = self.ncaco[:, :3]
   self.camask = self.pdb.camask()
   self.seq = self.pdb.sequence()
   self._chain = str.join('', [x.decode() for x in self.pdb.df.ch[self.pdb.df.an == b'CA']])
   self.nres = pdb.nres
   if secstruct is not None:
      self.ss = secstruct
      if len(self.ss) == 1:
         self.ss = self.ss * len(self.ncaco)
   else:
      try:
         self.ss = wu.dssp(self.ncaco)
      except ImportError:
         self.ss = 'L' * len(self.seq)
   self.crystinfo = CrystInfo.from_cryst1(pdb.cryst1)
   self.info = NotPDBInfo(self)
   self.pdb.renumber_from_0()
   self.coordsonly = False
   self.name = self.pdb.meta.fname

def _init_NotPose_coords(self, coords, seq=None, name=None, chain=None, secstruct=None, **kw):
   kw = wu.Bunch(kw, _strict=False)
   self.coordsonly = True
   assert kw.pdb is None
   assert kw.chain is None
   self.fname = None
   self.pdb = None
   self.rawpdb = None
   self.pdbhet = None
   assert coords.ndim in (3, 4)
   if len(coords) == 0:
      raise ValueError(f'Can\'t create NotPose from empty coordinates')
   coords = wu.hpoint(coords)
   self._chain = 'A' * len(coords)
   if coords.ndim == 4:
      self._chain = str.join('', [wu.pdb.all_pymol_chains[i] * coords.shape[1] for i in range(len(coords))])
      coords = coords.reshape(-1, *coords.shape[-2:])
   self._chain = chain or self._chain
   assert coords.shape[1] > 2
   if coords.shape[1] == 3:
      coords = wu.chem.add_bb_o_guess(coords)
   assert len(coords) > 0
   self.coords = wu.hpoint(coords)
   self.ncaco = self.coords[:, :4]
   self.ncac = self.coords[:, :3]
   self.camask = np.ones(len(coords), dtype=bool)
   self.nres = len(coords)
   self.seq = seq or ('G' * len(coords))
   if secstruct is not None:
      self.ss = secstruct
      if len(self.ss) == 1:
         self.ss = self.ss * len(self.ncaco)
   else:
      try:
         self.ss = wu.dssp(self.ncaco)
      except ImportError:
         self.ss = 'L' * len(self.seq)

   # self.crystinfo = CrystInfo.from_cryst1(cryst1)
   self.crystinfo = None
   self.info = NotPDBInfo(self)
   # self.pdb.renumber_from_0()
   self.name = name or 'NONAME'
