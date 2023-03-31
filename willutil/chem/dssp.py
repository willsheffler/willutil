import numpy as np
import tempfile
import willutil as wu

def add_bb_o_guess(coords):
   origshape = coords.shape
   coords = wu.hpoint(coords)
   assert coords.shape[-2:] == (3, 4)
   coords = coords.reshape(-1, 3, 4)
   _OCOORD = np.array([0.63033436, -0.52888702, 0.91491776])
   stubs = wu.hframe(coords[:-1, 2], coords[:-1, 1], coords[1:, 2])
   o = wu.hxform(stubs, _OCOORD, homogout=True)
   lasto = np.array([[0, 0, 0, 1]])
   o = np.concatenate([o, lasto]).reshape(-1, 1, 4)
   newcoords = np.concatenate([coords, o], axis=1)
   ic(newcoords.shape)
   newcoords = newcoords.reshape(*origshape[:-2], origshape[-2] + 1, 4)
   newcoords = newcoords[..., :origshape[-1]]
   return newcoords

def dssp(coords):
   assert len(coords) > 0
   if coords.shape[-2] == 3:
      coords = add_bb_o_guess(coords)
   import mdtraj
   with tempfile.TemporaryDirectory() as d:
      wu.pdb.dump_pdb_from_ncaco_points(d + '/tmp.pdb', coords)
      t = mdtraj.load(d + '/tmp.pdb')
      ss = ''.join(mdtraj.compute_dssp(t)[0]).replace('C', 'L')
      return ss
