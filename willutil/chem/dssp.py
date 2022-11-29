import numpy as np
import tempfile
import willutil as wu

def add_bb_o_guess(coords):
   # pdb = wu.pdb.readpdb('/home/sheffler/src/rpxdock/rpxdock/data/pdb/DHR14.pdb.gz')
   # xyz = pdb.ncaco()
   # stubs = wu.hframe(xyz[:-1, 2], xyz[:-1, 1], xyz[1:, 2])
   # ocoord = wu.hxform(wu.hinv(stubs), xyz[:-1, 3])
   # _OCOORD = np.median(ocoord[4:-4], axis=0)
   # ic(ocoord)
   # opos = ocoord
   # ohat = wu.hxform(stubs, _OCOORD)
   # ic(_OCOORD)
   # ic(ohat[:10])
   # ic(xyz[:10, 3])
   # assert np.mean(wu.hnorm(ohat - xyz[:-1, 3])) < 0.1
   # assert np.allclose(ohat, xyz[:-1, 3], atol=0.1)
   # assert 0
   # assert coords.shape[-2:] == (3, 3)
   _OCOORD = np.array([0.63033436, -0.52888702, 0.91491776])
   stubs = wu.hframe(coords[:-1, 2], coords[:-1, 1], coords[1:, 2])
   o = wu.hxform(stubs, _OCOORD, homogout=False)
   lasto = np.array([[0, 0, 0]])
   # ic(o.shape, lasto.shape)
   o = np.concatenate([o, lasto]).reshape(-1, 1, 3)
   # ic(o.shape, coords.shape)
   newcoords = np.concatenate([coords, o], axis=1)
   return newcoords

def dssp(coords):
   if coords.shape[-2] == 3:
      coords = add_bb_o_guess(coords)
   import mdtraj
   with tempfile.TemporaryDirectory() as d:
      ic(coords.shape)
      wu.pdb.dump_pdb_from_ncaco_points(d + '/tmp.pdb', coords)
      t = mdtraj.load(d + '/tmp.pdb')
      ss = ''.join(mdtraj.compute_dssp(t)[0]).replace('C', 'L')
      return ss
