import willutil as wu

class NotPose:
   def __init__(self, fname=None, pdb=None, chain=None):
      self.fname = fname
      if pdb is None:
         assert isinstance(fname, str)
         pdb = wu.pdb.readpdb(fname)
      if chain is not None:
         pdb = pdb.subset(chain=chain)
      self.pdb = pdb.subset(het=False)
      self.pdbhet = pdb.subset(het=True)
      self.bbcoords = wu.hpoint(self.pdb.bb())
      self.ncac = self.bbcoords[:, :3]
      self.camask = self.pdb.camask()
      try:
         self.ss = wu.dssp(self.bbcoords)
      except ImportError:
         self.ss = None
      self.seq = self.pdb.sequence()
      self.info = NotPDBInfo(self)
      self.pdb.renumber_from_0()

   def __len__(self):
      return self.size()

   def size(self):
      return self.pdb.nres

   def sequence(self):
      return self.pdb.seq

   def secstruct(self):
      return self.ss

   def chain(self, ires):
      return self.pdb.chain(ires)

   def extract(self, chain=None):
      return NotPose(self.fname, pdb=self.pdb, chain=chain)

   def pdb_info(self):
      return self.info

   def bbcoords(self):
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
      self.rdf = nopo.pdb.getres(self.ir)

   def xyz(self, ia):
      if isinstance(ia, int):
         ia -= 1
      if isinstance(ia, int):
         xyz = self.rdf.x[ia], self.rdf.y[ia], self.rdf.z[ia]
      if isinstance(ia, str):
         ia = ia.encode()
      if isinstance(ia, bytes):
         xyz = (
            float(self.rdf.x[self.rdf.an == ia]),
            float(self.rdf.y[self.rdf.an == ia]),
            float(self.rdf.z[self.rdf.an == ia]),
         )
      return NotXYZ(xyz)
      raise ValueError(ia)

   def has(self, aname):
      # could be more efficient
      if isinstance(aname, str):
         aname = aname.encode()
      return aname in set(self.rdf.an)

   def is_protein(self):
      return self.nopo.camask[self.ir]

   def natoms(self):
      return self.nheavyatoms()

   def nheavyatoms(self):
      return len(self.rdf)

   def atom_name(self, ia):
      r = self.rdf
      aname = r.an[ia - 1]
      return aname.decode()

   def name(self):
      r = self.rdf.rn[0]
      return r.decode()

class NotXYZ(list):
   def __init__(self, xyz):
      super().__init__(xyz)
      self.x = self[0]
      self.y = self[1]
      self.z = self[2]

class NotPDBInfo:
   def __init__(self, nopo):
      self.nopo = nopo

   def name(self):
      return self.nopo.pdb.meta.fname
