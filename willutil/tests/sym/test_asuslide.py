import numpy as np
import willutil as wu
from willutil.sym.asuslide import asuslide

def main():
   # test_asuslide_oct()
   test_asuslide_p213()

def test_asuslide_p213():
   sym = 'P 21 3'
   xtal = wu.sym.Xtal(sym)
   scale = 100

   xyz = wu.tests.point_cloud(300, std=30, outliers=50)
   asucen = xtal.asucen(use_olig_nbrs=True, cellsize=scale)
   xyz += wu.hvec(asucen)
   # xyz[:, 2] += 30

   # fname = '/home/sheffler/src/willutil/blob5h.pdb'
   # pdb = wu.pdb.readpdb(fname)
   # # xyz = np.stack([pdb.df['x'], pdb.df['y'], pdb.df['z']]).T
   # xyz, mask = pdb.coords()
   # xyz = xyz[:, :4].reshape(-1, 3)
   # # ic(xyz.shape)
   # # assert 0
   # xyz[:, :3] -= wu.hcom(xyz)[:3]
   # xyz[:, :3] += asucen[:3]
   # cendis = np.argsort(wu.hnorm(xyz - wu.hcom(xyz)[:3])**2)
   # w = cendis[:int(len(xyz) * 0.6)]
   # xyz_contact = xyz[w]
   wu.showme(xyz)

   primary_frames = np.stack([
      wu.hscaled(scale, np.eye(4)),
      xtal.symelems[0].operators[1],
      xtal.symelems[1].operators[1],
      xtal.symelems[0].operators[2],
      xtal.symelems[1].operators[2],
   ])
   primary_frames = wu.hscaled(scale, primary_frames)

   # frames = wu.sym.frames(sym, ontop=primary_frames, cells=[(-1, 0), (-1, 0), (-1, 0)])
   # frames = wu.sym.frames(sym, ontop=primary_frames, cells=3, center=
   frames = wu.sym.frames(
      sym,
      ontop=primary_frames,
      cells=(-1, 1),
      cellsize=scale,
      center=asucen,
      asucen=asucen,
      radius=scale * 0.8,
   )
   frames = primary_frames

   slid = asuslide(sym, xyz, frames, showme=True, maxstep=14, step=3, sphereradius=2, towardaxis=True, alongaxis=True,
                   checksubsets=True)
   ic(slid.cellsize)

def test_asuslide_oct():
   sym = 'oct'
   ax2 = wu.sym.axes(sym)[2]
   ax3 = wu.sym.axes(sym)[3]
   # axisinfo = [(2, ax2, (2, 3)), (3, ax3, 1)]
   axesinfo = [(ax2, [0, 0, 0]), (ax3, [0, 0, 0])]
   primary_frames = [np.eye(4), wu.hrot(ax2, 180), wu.hrot(ax3, 120), wu.hrot(ax3, 240)]
   frames = primary_frames
   # frames = wu.sym.frames(sym, ontop=primary_frames)

   fname = '/home/sheffler/src/willutil/blob4h.pdb'
   pdb = wu.pdb.readpdb(fname)
   pdb = pdb.subset(atomnames=['CA'], chains=['A'])
   xyz = np.stack([pdb.df['x'], pdb.df['y'], pdb.df['z']]).T
   xyz[:, :3] -= wu.hcom(xyz)[:3]
   xyz[:, :3] += wu.hnormalized(ax2 * 1.2 + ax3)[:3] * 70
   # wu.showme(xyz)

   # slid1 = asuslide(sym, xyz, frames, axes=axesinfo, showme=True, sphereradius=2)
   # slid2 = asuslide(sym, xyz, frames, alongaxis=True, showme=True, sphereradius=2)
   slid1 = asuslide(sym, xyz, frames, axes=axesinfo, alongaxis=True)
   slid2 = asuslide(sym, xyz, frames, alongaxis=True)

   assert np.all(np.abs(slid1.frames()[:, :3, 3]) < 0.0001)
   assert np.allclose(np.eye(3), slid1.asym.position[:3, :3])
   assert np.allclose(wu.hcart3(slid1.asym.globalposition), [-21.07033885, -21.07033885, -6.92820323])

   assert np.all(np.abs(slid2.frames()[:, :3, 3]) < 0.0001)
   assert np.allclose(np.eye(3), slid2.asym.position[:3, :3])
   assert np.allclose(wu.hcart3(slid2.asym.globalposition), [-21.07033885, -21.07033885, -6.92820323])

if __name__ == '__main__':
   main()
   ic('test_aluslide DONE')
