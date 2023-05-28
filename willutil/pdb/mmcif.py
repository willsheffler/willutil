import os, io, tempfile

from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.mmcifio import MMCIFIO

import willutil as wu

def readcif(fname):
   finfo = wu.storage.fname_extensions(fname)
   parser = MMCIFParser()
   with tempfile.TemporaryDirectory() as td:
      readfname = f'{td}/tmp.cif'
      if finfo.compression == '.gz':
         os.system(f'zcat {fname} > {td}/tmp.cif')
      elif finfo.compression == '.xz':
         os.system(f'xzcat {fname} > {td}/tmp.cif')
      else:
         readfname = fname
         assert not finfo.compression
      # ic(readfname)
      # os.system(f'du -h {td}/tmp.cif')
      mmcif_dict = MMCIF2Dict(readfname)
      struct = parser.get_structure('????', readfname)
   cryst1 = wu.sym.cryst1_pattern_full % (
      float(mmcif_dict['_cell.length_a'][0]),
      float(mmcif_dict['_cell.length_b'][0]),
      float(mmcif_dict['_cell.length_c'][0]),
      float(mmcif_dict['_cell.angle_alpha'][0]),
      float(mmcif_dict['_cell.angle_beta'][0]),
      float(mmcif_dict['_cell.angle_gamma'][0]),
      mmcif_dict['_symmetry.space_group_name_H-M'][0],
   )
   pdb_io = PDBIO()
   pdb_io.set_structure(struct)
   strfile = io.StringIO()
   strfile.write(cryst1)
   pdb_io.save(strfile)

   pdb = wu.readpdb(strfile.getvalue())
   pdb.set_cif_info(mmcif_dict)

   return pdb

def dumpcif(fname, pdb, cifdict=None, **kw):
   '''creates a cif file from a PDBFile object
   uses biopython. first dumps a cif from biopython 'Structure"
   to get structural info in the dict format. kinda dumb, but not
   clear how to add annotations to a biopython Structure
   '''
   finfo = wu.storage.fname_extensions(fname)
   if not finfo.ext == '.cif': fname += '.cif'

   pdbout = io.StringIO()
   pdb.dump_pdb(pdbout, **kw)
   struct = PDBParser().get_structure('????', io.StringIO(pdbout.getvalue()))

   cifdict = cifdict or dict()
   if pdb.cryst1:
      # ic(pdb.cryst1)
      s = pdb.cryst1.split()
      a, b, c, alpha, beta, gamma = (float(x) for x in s[1:7])
      spacegroup = ' '.join(s[7:])
      cifdict['_cell.length_a'] = [str(a)]
      cifdict['_cell.length_b'] = [str(b)]
      cifdict['_cell.length_c'] = [str(c)]
      cifdict['_cell.angle_alpha'] = [str(alpha)]
      cifdict['_cell.angle_beta'] = [str(beta)]
      cifdict['_cell.angle_gamma'] = [str(gamma)]
      cifdict['_symmetry.space_group_name_H-M'] = [spacegroup]

   cifio = MMCIFIO()
   cifio.set_structure(struct)
   cifout = io.StringIO()
   cifio.save(cifout)
   cifdict.update(MMCIF2Dict(io.StringIO(cifout.getvalue())))

   cifio2 = MMCIFIO()
   cifio2.set_dict(cifdict)
   cifio2.save(fname)
