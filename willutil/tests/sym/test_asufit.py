import copy
import pytest
import numpy as np
import willutil as wu

# ic.configureOutput(includeContext=True)

def prune_radial_outliers(xyz, nprune=10):
   npoints = len(xyz) - nprune
   for i in range(nprune):
      com = wu.hcom(xyz)
      r = wu.hnorm(xyz - com)
      w = np.argsort(r)
      xyz = xyz[w[:-1]]
   return xyz

def point_cloud(npoints=100, std=10, outliers=0):
   xyz = wu.hrandpoint(npoints + outliers, std=10)
   xyz = prune_radial_outliers(xyz, outliers)
   assert len(xyz) == npoints
   xyz = xyz[np.argsort(xyz[:, 0])]
   xyz -= wu.hvec(wu.hcom(xyz))
   return xyz

def main():
   # test_asufit_I4132(showme=True)
   # test_asufit_P213(showme=True)
   test_asufit_L6m322(showme=True)
   # test_asufit_L632(showme=True)
   # test_asufit_oct(showme=True)
   # test_asufit_icos(showme=True)
   ic('TEST asufit DONE')

@pytest.mark.xfail()
def test_asufit_oct(showme=False):
   sym = 'oct'
   fname = '/home/sheffler/src/willutil/blob4h.pdb'
   pdb = wu.pdb.readpdb(fname)
   pdb = pdb.subset(atomnames=['CA'], chains=['A'])
   xyz = np.stack([pdb.df['x'], pdb.df['y'], pdb.df['z']]).T
   xyz[:, :3] -= wu.hcom(xyz)[:3]
   xyz_contact = None

   cendis = np.argsort(wu.hnorm(xyz - wu.hcom(xyz)[:3])**2)
   w = cendis[:int(len(xyz) * 0.5)]
   xyz_contact = xyz[w]
   # wu.showme(xyz)
   # wu.showme(xyz_contact)
   # assert 0

   ax2 = wu.sym.axes(sym)[2]
   ax3 = wu.sym.axes(sym)[3]
   # xyz = point_cloud(100, std=10, outliers=0)
   # xyz[:, :3] += 60 * (wu.sym.axes(sym)[2] + wu.sym.axes(sym)[3])[:3]
   # wu.showme(xyz)
   primary_frames = [np.eye(4), wu.hrot(ax2, 180), wu.hrot(ax3, 120)]  #, wu.hrot(ax3, 240)]
   frames = wu.sym.frames(sym, ontop=primary_frames)
   lever = wu.hrog(xyz) * 1.5
   '''

   '''
   with wu.Timer():
      ic('symfit')
      # np.random.seed(7)
      mc = wu.sym.asufit(
         sym,
         xyz,
         xyz_contact,
         symaxes=[ax3, ax2],
         frames=frames,
         showme=True,
         showme_accepts=True,
         fresh=True,
         headless=False,
         contactfrac=0.3,
         contactdist=10,
         clashdist=4,
         clashpenalty=0.1,
         cartsd=2.5,
         temperature=1.0,
         resetinterval=100,
         correctionfactor=1.5,
         iterations=1000,
         driftpenalty=0.0,
         anglepenalty=0.5,
         thresh=0.0,
         spreadpenalty=0.1,
         biasradial=4,
         usebvh=True,
         sphereradius=3,
         scoreframes=[(0, 1), (0, 2)],
         clashframes=[(1, 2), (1, 3), (2, 3)],
      )
   assert np.allclose(
      mc.beststate.position,
      np.array([[9.63284623e-01, 2.49202698e-01, -9.99037016e-02, -1.32625492e+01],
                [-2.65707884e-01, 9.38228112e-01, -2.21646860e-01, 5.46007841e+01],
                [3.84974658e-02, 2.40054213e-01, 9.69995835e-01, 1.16772927e+01],
                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]))

@pytest.mark.xfail()
def test_asufit_I4132(showme=False):
   sym = 'I4132_C322'
   xtal = wu.sym.Xtal(sym)

   scale = 140
   asucen = xtal.asucen(cellsize=scale)
   # np.random.seed(2)
   # asucen = xtal.asucen(cellsize=scale)
   # xyz = point_cloud(10, std=20, outliers=0)
   # xyz += wu.hvec(asucen)
   # xyz[:, 0] += +0.000 * scale
   # xyz[:, 1] += -0.030 * scale
   # xyz[:, 2] += -0.020 * scale

   fname = '/home/sheffler/src/willutil/blob5h.pdb'
   pdb = wu.pdb.readpdb(fname)
   # xyz = np.stack([pdb.df['x'], pdb.df['y'], pdb.df['z']]).T
   xyz, mask = pdb.coords()
   xyz = xyz[:, :4].reshape(-1, 3)
   xyz[:, :3] -= wu.hcom(xyz)[:3]
   xyz[:, :3] += asucen[:3]

   cendis = np.argsort(wu.hnorm(xyz - wu.hcom(xyz)[:3])**2)
   w = cendis[:int(len(xyz) * 0.8)]
   xyz_contact = xyz[w]

   primary_frames = np.stack([
      np.eye(4),
      xtal.symelems[0].operators[1],
      xtal.symelems[1].operators[1],
      xtal.symelems[2].operators[1],
      xtal.symelems[0].operators[2],
   ])
   primary_frames = wu.hscaled(scale, primary_frames)
   # wu.showme(wu.hxform(primary_frames[0], xyz))
   # wu.showme(wu.hxform(primary_frames[1], xyz))
   # wu.showme(wu.hxform(primary_frames[2], xyz))
   # wu.showme(wu.hxform(primary_frames[3], xyz))
   # wu.showme(wu.hxform(primary_frames[4], xyz))
   # wu.showme(xtal.symelems, scale=scale, symelemscale=0.5, name='cenelems')
   # assert 0
   # frames = wu.sym.frames(sym, ontop=primary_frames, cells=(-1, 1), cellsize=scale, center=asucen, asucen=asucen,
   # radius=scale * 3)
   # ic(scale)
   frames = wu.sym.frames(sym, ontop=primary_frames, cells=(-1, 1), cellsize=scale, center=asucen, asucen=asucen,
                          radius=scale / 3)
   # ic(frames.shape)
   # ic(frames)
   # wu.showme(primary_frames, scale=1)
   # wu.showme(frames)
   # assert 0

   if 0:
      # cenelems = xtal.central_symelems(target=[-0.1, -0.05, 0.1])
      # ic(xtal.symelems)
      # ic(cenelems)
      # assert 0
      # wu.showme(cenelems, scale=scale, symelemscale=0.5, name='cenelems')

      wu.showme(xtal.symelems, scale=scale, symelemscale=2)

      wu.showme(wu.hxform(primary_frames[0], xyz), sphere=10, col=(1, 1, 1))
      wu.showme(wu.hxform(primary_frames[1], xyz), sphere=10, col=(1, 0, 0))
      wu.showme(wu.hxform(primary_frames[2], xyz), sphere=10, col=(0, 1, 0))
      wu.showme(wu.hxform(primary_frames[3], xyz), sphere=10, col=(0, 0, 1))
      wu.showme(wu.hxform(primary_frames[4], xyz), sphere=10, col=(1, 1, 0))

      from willutil.tests.sym.test_xtal import test_hxtal_viz
      test_hxtal_viz(
         spacegroup='I4132_C322',
         headless=False,
         # showpoints=wu.hcom(xyz),
         cells=(-1, 0),
         symelemscale=0.3,
         fansize=np.array([1.7, 1.2, 0.7]) / 3,
         fancover=10,
         symelemtwosided=True,
         showsymelems=True,
         scale=scale,
         pointradius=17,
      )

      # frames = wu.sym.frames(sym, ontop=primary_frames, cells=3, cellsize=scale, center=asucen, radius=scale * 2)
      # ic(frames.shape)
      # assert 0
      # lever = wu.hrog(xyz) * 1.5
      # assert 0

      # wu.showme(
      #    list(wu.hxform(frames, xyz)),
      #    sphere=10,
      #    name='framepts',
      #    topcolors=[(0, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)],
      #    chainbow=True,
      # )
      # assert 0

   # asucen = wu.hcom(xyz) / scale
   # xyz[:, :3] += scale

   frames = wu.hscaled(1 / scale, frames)
   lever = wu.hrog(xyz) * 1.5
   with wu.Timer():
      ic('symfit')
      # np.random.seed(14)
      mc = wu.sym.asufit(
         sym,
         xyz,
         xyz_contact,
         # symaxes=[3, 2],
         frames=frames,
         dumppdb=False,
         dumppdbscale=1,
         showme=False,
         showme_accepts=True,
         fresh=True,
         headless=False,
         spacegroup='I 41 3 2',
         # png='I4132_322',
         contactfrac=0.1,
         contactdist=10,
         clashdist=5,
         clashpenalty=10.1,
         cartsd=1.5,
         temperature=0.6,
         resetinterval=10000,
         correctionfactor=1.5,
         iterations=1000,
         nresatom=4,
         driftpenalty=0.2,
         anglepenalty=0.5,
         thresh=0.0,
         spreadpenalty=0.1,
         biasradial=1,
         usebvh=True,
         sphereradius=2,
         scale=scale,
         scalesd=1,
         scoreframes=[(0, 1), (0, 2), (0, 3)],
         clashframes=[(1, 2), (1, 3), (2, 3)],
         # topcolors=[(1, 1, 1)] + [(.9, .9, .9)] * 5),
         topcolors=[(1, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)],
         chainbow=False,
      )

   assert np.allclose(
      mc.beststate.position,
      np.array([[9.63284623e-01, 2.49202698e-01, -9.99037016e-02, -1.32625492e+01],
                [-2.65707884e-01, 9.38228112e-01, -2.21646860e-01, 5.46007841e+01],
                [3.84974658e-02, 2.40054213e-01, 9.69995835e-01, 1.16772927e+01],
                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]))

@pytest.mark.xfail()
def test_asufit_P213(showme=False):
   sym = 'P 21 3'
   xtal = wu.sym.Xtal(sym)
   scale = 100

   # xyz = point_cloud(100, std=30, outliers=20)
   asucen = xtal.asucen(use_olig_nbrs=True, cellsize=scale)
   # xyz += wu.hvec(asucen)
   # xyz[:, 2] += 30

   fname = '/home/sheffler/src/willutil/blob5h.pdb'
   pdb = wu.pdb.readpdb(fname)
   # xyz = np.stack([pdb.df['x'], pdb.df['y'], pdb.df['z']]).T
   xyz, mask = pdb.coords()
   xyz = xyz[:, :4].reshape(-1, 3)
   # ic(xyz.shape)
   # assert 0
   xyz[:, :3] -= wu.hcom(xyz)[:3]
   xyz[:, :3] += asucen[:3]
   cendis = np.argsort(wu.hnorm(xyz - wu.hcom(xyz)[:3])**2)
   w = cendis[:int(len(xyz) * 0.6)]
   xyz_contact = xyz[w]

   primary_frames = np.stack([
      wu.hscaled(scale, np.eye(4)),
      xtal.symelems[0].operators[1],
      xtal.symelems[1].operators[1],
      xtal.symelems[0].operators[2],
      xtal.symelems[1].operators[2],
   ])
   primary_frames = wu.hscaled(scale, primary_frames)
   # ic(xtal.symelems[0].cen)
   # ic(xtal.symelems[1].cen)
   # ic(asucen)

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
   # ic(frames.shape)
   # wu.showme(xtal.symelems, scale=scale, symelemscale=2)
   # wu.showme(wu.hxform(primary_frames[0], xyz), sphere=10, col=(1, 1, 1))
   # wu.showme(wu.hxform(primary_frames[1], xyz), sphere=10, col=(1, 0, 0))
   # wu.showme(wu.hxform(primary_frames[2], xyz), sphere=10, col=(0, 1, 0))
   # wu.showme(wu.hxform(primary_frames[3], xyz), sphere=10, col=(0, 0, 1))
   # wu.showme(wu.hxform(primary_frames[4], xyz), sphere=10, col=(1, 1, 0))
   # wu.showme(
   # list(wu.hxform(frames, xyz)),
   # sphere=10,
   # name='aoiresnt',
   # topcolors=[(1, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)],
   # )
   # assert 0

   frames = wu.hscaled(1 / scale, frames)
   lever = wu.hrog(xyz) * 1.5
   '''

   '''
   with wu.Timer():
      ic('symfit')
      # np.random.seed(7)
      mc = wu.sym.asufit(
         sym,
         xyz,
         xyz_contact,
         spacegroup='P 21 3',
         nresatom=4,
         # symaxes=[3, 2],
         frames=frames,
         showme=True,
         showme_accepts=True,
         fresh=True,
         headless=False,
         contactfrac=0.1,
         contactdist=12,
         clashdist=6,
         clashpenalty=10.1,
         cartsd=1,
         temperature=0.5,
         resetinterval=200,
         correctionfactor=1.5,
         iterations=1000,
         driftpenalty=0.0,
         anglepenalty=0.5,
         thresh=0.0,
         spreadpenalty=0.1,
         biasradial=1,
         usebvh=True,
         sphereradius=2,
         scale=scale,
         scalesd=4,
         scoreframes=[(0, 1), (0, 2)],
         clashframes=[(1, 2), (1, 3), (2, 3)],
         # topcolors=[(1, 1, 1)] + [(.9, .9, .9)] * 5),
         topcolors=[(1, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)],
         # chainbow=True,
      )
   assert np.allclose(
      mc.beststate.position,
      np.array([[9.63284623e-01, 2.49202698e-01, -9.99037016e-02, -1.32625492e+01],
                [-2.65707884e-01, 9.38228112e-01, -2.21646860e-01, 5.46007841e+01],
                [3.84974658e-02, 2.40054213e-01, 9.69995835e-01, 1.16772927e+01],
                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]))

@pytest.mark.xfail()
def test_asufit_L6m322(showme=False):
   sym = 'L6m322'
   xtal = wu.sym.Xtal(sym)
   scale = 70
   # xyz = point_cloud(100, std=30, outliers=20)
   # xyz += wu.hvec(xtal.asucen(cellsize=scale))
   # xyz[:, 2] += 20

   fname = '/home/sheffler/src/willutil/blob6h.pdb'
   pdb = wu.pdb.readpdb(fname)
   # xyz = np.stack([pdb.df['x'], pdb.df['y'], pdb.df['z']]).T
   xyz, mask = pdb.coords()
   xyz = xyz[:, :4].reshape(-1, 3)
   xyz[:, :3] -= wu.hcom(xyz)[:3]
   xyz[:, 0] += scale * 0.3
   xyz[:, 1] += scale * 0.1
   xyz[:, 2] += 18
   ss = np.array(list(wu.chem.dssp(xyz.reshape(-1, 4, 3))))
   xyz_contact = xyz.reshape(-1, 4, 3)[ss == 'H'].reshape(-1, 3)
   ic(xyz_contact.shape)

   primary_frames = np.stack([
      np.eye(4),
      xtal.symelems[0].operators[1],
      xtal.symelems[1].operators[1],
      xtal.symelems[2].operators[1],
   ])
   frames = wu.sym.frames(sym, ontop=primary_frames)
   lever = wu.hrog(xyz) * 1.5
   '''

   '''
   for i in range(10):
      with wu.Timer():
         ic('symfit')
         # np.random.seed(7)
         mc = wu.sym.asufit(
            sym,
            xyz,
            xyz_contact,
            # symaxes=[3, 2],
            dumppdb=f'P6m32_{i:04}.pdb',
            frames=frames,
            showme=False,
            showme_accepts=False,
            fresh=True,
            headless=False,
            contactfrac=0.1,
            contactdist=12,
            clashdist=5,
            clashpenalty=10.1,
            cartsd=2,
            temperature=0.8,
            resetinterval=20000,
            correctionfactor=1.5,
            iterations=1000,
            driftpenalty=0.0,
            anglepenalty=0.5,
            thresh=0.0,
            spreadpenalty=0.1,
            biasradial=1,
            usebvh=True,
            sphereradius=2,
            scale=scale,
            scalesd=4,
            scoreframes=[(0, 1), (0, 2), (0, 3)],
            clashframes=[(1, 2), (1, 3), (2, 3)],
         )
   assert np.allclose(
      mc.beststate.position,
      np.array([[9.63284623e-01, 2.49202698e-01, -9.99037016e-02, -1.32625492e+01],
                [-2.65707884e-01, 9.38228112e-01, -2.21646860e-01, 5.46007841e+01],
                [3.84974658e-02, 2.40054213e-01, 9.69995835e-01, 1.16772927e+01],
                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]))

@pytest.mark.xfail()
def test_asufit_L632(showme=False):
   sym = 'L632'
   xtal = wu.sym.Xtal(sym)
   scale = 100

   xyz = point_cloud(100, std=10, outliers=20)
   xyz += wu.hvec(xtal.asucen()) * scale

   primary_frames = np.stack([np.eye(4), xtal.symelems[0].operators[1], xtal.symelems[1].operators[1]])
   frames = wu.sym.frames(sym, ontop=primary_frames)
   lever = wu.hrog(xyz) * 1.5
   '''

   '''
   with wu.Timer():
      ic('symfit')
      # np.random.seed(7)
      mc = wu.sym.asufit(
         sym,
         xyz,
         # symaxes=[3, 2],
         frames=frames,
         showme=True,
         showme_accepts=True,
         fresh=True,
         headless=False,
         contactfrac=0.3,
         contactdist=16,
         cartsd=2,
         temperature=0.5,
         resetinterval=200,
         correctionfactor=1.5,
         iterations=1000,
         driftpenalty=0.0,
         anglepenalty=0.5,
         thresh=0.0,
         spreadpenalty=0.1,
         biasradial=1,
         usebvh=True,
         sphereradius=6,
         scale=scale,
         scalesd=4,
         scoreframes=[(0, 1), (0, 2)],
         clashframes=[(1, 2), (1, 3), (2, 3)],
      )
   assert np.allclose(
      mc.beststate.position,
      np.array([[9.63284623e-01, 2.49202698e-01, -9.99037016e-02, -1.32625492e+01],
                [-2.65707884e-01, 9.38228112e-01, -2.21646860e-01, 5.46007841e+01],
                [3.84974658e-02, 2.40054213e-01, 9.69995835e-01, 1.16772927e+01],
                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]))

@pytest.mark.xfail()
def test_asufit_icos(showme=False):
   sym = 'icos'
   # fname = wu.tests.testdata.test_data_path('pdb/x012.pdb')
   # pdb = wu.pdb.readpdb(fname)
   # pdb = pdb.subset(atomnames=['CA'], chains=['A'])
   # xyz = np.stack([pdb.df['x'], pdb.df['y'], pdb.df['z']]).T
   # xyz[:, :3] += wu.hcom(xyz)[:3]
   xyz = point_cloud(100, std=10, outliers=10)
   xyz[:, :3] += 140 * wu.hnormalized(wu.sym.axes(sym)[2] * 4 + wu.sym.axes(sym)[3])[:3]
   ax2 = wu.sym.axes('icos')[2]
   ax3 = wu.sym.axes('icos')[3]
   primary_frames = [np.eye(4), wu.hrot(ax2, 180), wu.hrot(ax3, 120)]  #, wu.hrot(ax3, 240)]
   frames = wu.sym.frames(sym, ontop=primary_frames)

   lever = wu.hrog(xyz) * 1.5
   with wu.Timer():
      ic('symfit')
      # np.random.seed(7)
      mc = wu.sym.asufit(
         sym,
         xyz,
         symaxes=[ax3, ax2],
         frames=frames,
         showme=showme,
         showme_accepts=True,
         fresh=True,
         contactfrac=0.2,
         contactdist=12,
         cartsd=1,
         temperature=1,
         resetinterval=100,
         correctionfactor=1.5,
         iterations=1000,
         sphereradius=10,
         driftpenalty=0.1,
         anglepenalty=0.1,
         thresh=0.0,
         spreadpenalty=0.1,
         biasradial=4,
         usebvh=True,
         scoreframes=[(0, 1), (0, 2)],
         clashframes=[(1, 2), (1, 3), (2, 3)],
      )
      ref = np.array([[0.99880172, 0.01937787, 0.04494025, -3.38724054],
                      [-0.02529149, 0.99052266, 0.13500072, 1.91723184],
                      [-0.04189831, -0.13597556, 0.98982583, 2.10409862], [0., 0., 0., 1.]])
      assert np.allclose(ref, mc.beststate.position)
      ic('test_asufit_icos PASS!')

if __name__ == '__main__':
   main()