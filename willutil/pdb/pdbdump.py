import deferred_import, os, willutil as wu

np = deferred_import.deferred_import('numpy')

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

def dump_pdb_from_points(fname, pts, header='', frames=[np.eye(4)]):
   pts = np.asarray(pts)

   if not (pts.ndim == 2 and pts.shape[-1] in (3, 4)):
      raise ValueError(f'bad shape for points {pts.shape}')
   if os.path.dirname(fname):
      os.makedirs(os.path.dirname(fname), exist_ok=True)
   ia = 1
   with open(fname, "w") as out:
      out.write(header)
      for ic, f in enumerate(frames):
         for i, p in enumerate(pts):
            p = wu.hxform(f, p)
            s = pdb_format_atom(ia=ia, x=p[0], y=p[1], z=p[2], ir=i, c=ic)
            out.write(s)
            ia += 1

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
      for ic, pc in enumerate(pts):
         chain = all_pymol_chains[ic]
         # print(ic, pc.shape, chain)
         for i, p in enumerate(pc):
            a = pdb_format_atom(ia + 0, an='N', x=p[0, 0], y=p[0, 1], z=p[0, 2], ir=i, c=chain)
            b = pdb_format_atom(ia + 1, an='CA', x=p[1, 0], y=p[1, 1], z=p[1, 2], ir=i, c=chain)
            c = pdb_format_atom(ia + 2, an='C', x=p[2, 0], y=p[2, 1], z=p[2, 2], ir=i, c=chain)
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
      for ic, pc in enumerate(pts):
         chain = all_pymol_chains[ic]
         # print(ic, pc.shape, chain)
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
