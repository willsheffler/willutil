import deferred_import

np = deferred_import.deferred_import('numpy')
from willutil.chem import aa123, aa321, aa1

all_pymol_chains = ("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz")

def pdb_format_atom(
    ia=0,
    an="ATOM",
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
    if rn in aa1:
        rn = aa123[rn]
    if not isinstance(c, str):
        c = all_pymol_chains[c]

    format_str = _pdb_atom_record_format
    if ia >= 100000:
        format_str = format_str.replace("ATOM  {ia:5d}", "ATOM {ia:6d}")
    if ir >= 10000:
        format_str = format_str.replace("{ir:4d}{insert:1s}", "{ir:5d}")

    return format_str.format(**locals())

def dump_pdb_from_points(fname, pts):
    with open(fname, "w") as out:
        for i, p in enumerate(pts):
            s = pdb_format_atom(x=p[0], y=p[1], z=p[2], ir=i)
            out.write(s)

def dump_pdb_from_ncac_points(fname, pts):
    with open(fname, "w") as out:
        for i, p in enumerate(pts):
            a = pdb_format_atom(an='N', x=p[0, 0], y=p[0, 1], z=p[0, 2], ir=i)
            b = pdb_format_atom(an='CA', x=p[1, 0], y=p[1, 1], z=p[1, 2], ir=i)
            c = pdb_format_atom(an='C', x=p[2, 0], y=p[2, 1], z=p[2, 2], ir=i)
            out.write(a)
            out.write(b)
            out.write(c)

_pdb_atom_record_format = ("ATOM  {ia:5d} {an:^4}{idx:^1}{rn:3s} {c:1}{ir:4d}{insert:1s}   "
                           "{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{b:6.2f}           {elem:1s}\n")

def aname_to_elem(aname):
    "return based on first occurance of element letter"
    aname = aname.upper()
    elems = "COHNS"
    pos = [aname.find(e) for e in elems]
    poselem = sorted([(p, e) for p, e in zip(pos, elems) if p >= 0])
    return poselem[0][1]
