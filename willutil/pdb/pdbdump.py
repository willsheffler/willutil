from collections import defaultdict
import deferred_import, os
import willutil as wu

import numpy as np

all_pymol_chains = ("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz" * 10)

def pdb_format_atom(
   ia=0,
   an="CA",
   idx=" ",
   rn="RES",
   c="A",
   ir=0,
   insert=" ",
   x=0,
   y=0,
   z=0,
   occ=1,
   b=1,
   elem=" ",
   xyz=None,
):
   if xyz is not None:
      x, y, z, *_ = xyz.squeeze()
   if rn in wu.chem.aa1:
      rn = wu.chem.aa123[rn]
   if not isinstance(c, str):
      c = all_pymol_chains[c]

   format_str = _pdb_atom_record_format
   if ia >= 100000:
      format_str = format_str.replace("ATOM  {ia:5d}", "ATOM {ia:6d}")
   if ir >= 10000:
      format_str = format_str.replace("{ir:4d}{insert:1s}", "{ir:5d}")

   return format_str.format(**locals())

_dumppdb_seenit = defaultdict(lambda: -1)

def dumppdb(fname, stuff, *a, namereset=False, **kw):
   global _dumppdb_seenit
   if not fname.endswith(('.pdb', '.pdb.gz')) or fname.count('%04i'):
      if namereset:
         del _dumppdb_seenit[fname]
      if not fname.count('%04i'):
         fname += '_%04i.pdb'
      _dumppdb_seenit[fname] += 1
      fname = fname % _dumppdb_seenit[fname]
   if hasattr(stuff, 'dumppdb'):
      return stuff.dumppdb(fname, *a, **kw)
   else:
      return dump_pdb_from_points(fname, stuff, *a, **kw)

def dump_pdb_from_points(
   fname,
   pts,
   mask=None,
   anames=['N', 'CA', 'C', 'O', 'CB'],
   resnames=[],
   nres=-1,
   nresatom=-1,
   header='',
   frames=None,
   nchain=-1,
   dumppdbscale=1,
   spacegroup=None,
   cellsize=None,
   **kw,
):
   pts = np.asarray(pts)
   if frames is not None:
      pts = wu.hxform(frames, pts)
   if mask is None:
      mask = np.ones(pts.shape[:-1], dtype=bool)
   if not (pts.ndim in (2, 3, 4) and pts.shape[-1] in (3, 4)):
      raise ValueError(f'bad shape for points {pts.shape}')
   if os.path.dirname(fname):
      os.makedirs(os.path.dirname(fname), exist_ok=True)
   if nresatom == -1 and nres == -1 and pts.ndim == 3:
      if pts.shape[-2] < 30:  # assume nres not natom
         nresatom = pts.shape[-2]
      else:
         nresatom = 1
         pts = np.expand_dims(pts, 2)
         mask = None if mask is None else np.expand_dims(mask, 2)
   if pts.ndim == 4:
      assert nchain < 0 or pts.shape[0] == nchain
      assert nres < 0 or pts.shape[1] == nres
      assert nresatom < 0 or pts.shape[2] == nresatom
   elif pts.ndim == 3:
      nres = len(pts)
      if nchain > 0:
         assert not nres % nchain
         nres = nres // nchain
      pts = pts.reshape(nchain, nres, nresatom, pts.shape[-1])
      mask = mask.reshape(nchain, nres, nresatom)
   elif pts.ndim == 2:
      nres = len(pts)
      if nresatom < 0: nresatom = 1
      nres = nres // nresatom
      pts = pts.reshape(nchain, nres, nresatom, pts.shape[-1])
      mask = mask.reshape(nchain, nres, nresatom)
   else:
      raise ValueError(f'bad pts shape {pts.shape}')
   nchain, nres, nresatom, _ = pts.shape

   anames = anames[:nresatom]
   if not resnames:
      resnames = ['ALA'] * nres
   if nres < 0:
      nres = len(resnames)
   else:
      assert len(resnames) == nres

   if spacegroup is not None:
      if not isinstance(cellsize, (int, float)):
         raise ValueError(f'bad cellsize {cellsize}')
      cryst1 = wu.sym.cryst1_pattern % (*(dumppdbscale * cellsize, ) * 3, spacegroup)
      header += cryst1 + os.linesep
   pts = np.clip(pts, -999.999, 9999.999)
   atomconut = 1
   with open(fname, "w") as out:
      out.write(header)
      # for ic1, f in enumerate(pts):
      for ichain, chainpts in enumerate(pts):
         for ires, respts in enumerate(chainpts):
            for iatom, p in enumerate(respts):
               if mask[ichain, ires, iatom]:
                  s = pdb_format_atom(
                     ia=atomconut,
                     x=p[0],
                     y=p[1],
                     z=p[2],
                     ir=ires,
                     rn=resnames[ires],
                     an=anames[iatom],
                     c=ichain,
                  )
                  out.write(s)
                  atomconut += 1

def dump_pdb_from_ncac_points(fname, pts, nchain=1):
   if os.path.dirname(fname):
      os.makedirs(os.path.dirname(fname), exist_ok=True)
   if pts.ndim == 3: pts = pts[np.newaxis]
   # print(pts.shape)
   pts = pts.reshape(nchain * len(pts), -1, 3, pts.shape[-1])
   # print(pts.shape)
   # assert 0
   # if len(pts) > 1:
   # print(pts.shape)
   # assert 0
   ia = 0
   with open(fname, "w") as out:
      for ichain, chainpts in enumerate(pts):
         for i, p in enumerate(chainpts):
            a = pdb_format_atom(ia + 0, an='N', x=p[0, 0], y=p[0, 1], z=p[0, 2], ir=i, c=ichain)
            b = pdb_format_atom(ia + 1, an='CA', x=p[1, 0], y=p[1, 1], z=p[1, 2], ir=i, c=ichain)
            c = pdb_format_atom(ia + 2, an='C', x=p[2, 0], y=p[2, 1], z=p[2, 2], ir=i, c=ichain)
            ia += 3
            out.write(a)
            out.write(b)
            out.write(c)

   # assert 0

def dump_pdb_from_ncaco_points(fname, pts, nchain=1):
   if os.path.dirname(fname):
      os.makedirs(os.path.dirname(fname), exist_ok=True)
   if pts.ndim == 3: pts = pts[np.newaxis]
   # print(pts.shape)
   pts = pts.reshape(nchain * len(pts), -1, *pts.shape[-2:])
   # print(pts.shape)
   # assert 0
   # if len(pts) > 1:
   # print(pts.shape)
   # assert 0
   ia = 0
   with open(fname, "w") as out:
      for ichain, pc in enumerate(pts):
         chain = all_pymol_chains[ichain]
         # print(ichain, pc.shape, chain)
         for i, p in enumerate(pc):
            a = pdb_format_atom(ia + 0, rn='GLY', an='N', x=p[0, 0], y=p[0, 1], z=p[0, 2], ir=i, c=chain)
            b = pdb_format_atom(ia + 1, rn='GLY', an='CA', x=p[1, 0], y=p[1, 1], z=p[1, 2], ir=i, c=chain)
            c = pdb_format_atom(ia + 2, rn='GLY', an='C', x=p[2, 0], y=p[2, 1], z=p[2, 2], ir=i, c=chain)
            d = pdb_format_atom(ia + 3, rn='GLY', an='O', x=p[3, 0], y=p[3, 1], z=p[3, 2], ir=i, c=chain)
            ia += 4
            out.write(a)
            out.write(b)
            out.write(c)
            out.write(d)

   # assert 0

_pdb_atom_record_format = ("ATOM  {ia:5d} {an:^4}{idx:^1}{rn:3s} {c:1}{ir:4d}{insert:1s}   "
                           "{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{b:6.2f}           {elem:1s}\n")

def aname_to_elem(aname):
   "return based on first occurance of element letter"
   aname = aname.upper()
   elems = "COHNS"
   pos = [aname.find(e) for e in elems]
   poselem = sorted([(p, e) for p, e in zip(pos, elems) if p >= 0])
   return poselem[0][1]
