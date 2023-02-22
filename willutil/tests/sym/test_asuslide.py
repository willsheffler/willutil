import numpy as np, functools as ft
import willutil as wu
from willutil.sym.asuslide import asuslide

# ic.configureOutput(includeContext=True, contextAbsPath=True)

def main():

   test_asuslide_helix_case1()

   test_asuslide_case2()
   test_asuslide_helix_nfold1_2()
   test_asuslide_helix_nfold5()
   test_asuslide_helix_nfold3()
   test_asuslide_helix_nfold1()
   test_asuslide_I4132_clashframes()
   test_asuslide_P432_44()
   test_asuslide_P432_43()
   test_asuslide_F432()
   test_asuslide_I432()
   test_asuslide_p213()
   test_asuslide_oct()
   test_asuslide_I4132()
   test_asuslide_I213()
   test_asuslide_L632()
   test_asuslide_L632_2()
   test_asuslide_L442()

   # test_asuslide_case2()
   # assert
   # test_asuslide_helix_nfold1_2()
   # test_asuslide_oct()
   # test_asuslide_L632_2(showme=True)
   # test_asuslide_P432_44(showme=True)
   # test_asuslide_P432_43(showme=True)
   assert 0

   test_asuslide_P432()
   test_asuslide_F432()
   test_asuslide_P4132()

   # asuslide_case3()

   # asuslide_case2()
   # asuslide_case1()
   # assert 0
   test_asuslide_L442()

   test_asuslide_I4132()

   test_asuslide_L632()

   test_asuslide_I213()

   ic('DONE')

def test_asuslide_L632_2(showme=False):
   sym = 'L6_32'
   xtal = wu.sym.Xtal(sym)
   csize = 160
   np.random.seed(7)
   xyz = wu.tests.point_cloud(100, std=30, outliers=0)
   asucen = xtal.asucen(use_olig_nbrs=True, cellsize=csize)
   xyz += wu.hvec(asucen)
   xyz[:, 1] -= 2

   primary_frames = np.stack([
      wu.hscaled(csize, np.eye(4)),
      xtal.symelems[0].operators[1],
      xtal.symelems[0].operators[2],
      xtal.symelems[1].operators[1],
   ])
   primary_frames = wu.hscaled(csize, primary_frames)
   frames = primary_frames

   slid = asuslide(sym, xyz, frames, showme=showme, maxstep=30, step=10, iters=5, clashiters=0, clashdis=8,
                   contactdis=16, contactfrac=0.2, vizsphereradius=6, cellsize=csize, towardaxis=True, alongaxis=False,
                   vizfresh=False, centerasu=False, scaleslides=1, resetonfail=True)
   # wu.showme(slid, vizsphereradius=6)
   ic(slid.cellsize)
   ic(slid.asym.com())
   assert np.allclose(slid.cellsize, 95)
   assert np.allclose(slid.asym.com(), [25.1628825, -1.05965433, 0, 1])

def asuslide_case4():

   sym = 'P432'
   xtal = wu.sym.Xtal(sym)
   # cellsize = 99.417
   cellsize = 76.38867528392643

   pdbfile = '/home/sheffler/project/diffusion/unbounded/preslide.pdb'
   pdb = wu.pdb.readpdb(pdbfile).subset(chain='A')
   xyz = pdb.ca()
   fracremains = 1.0
   primaryframes = xtal.primary_frames(cellsize)
   cen = wu.th_com(xyz.reshape(-1, xyz.shape[-1]))
   frames = wu.sym.frames(sym, ontop=primaryframes, cells=(-1, 1), cellsize=cellsize, center=cen,
                          xtalrad=cellsize * 0.9)
   # frames = primaryframes
   cfracmin = 0.7
   cfracmax = 0.7
   cdistmin = 14.0
   cdistmax = 14.0
   t = 1
   slid = wu.sym.asuslide(
      sym=sym,
      coords=xyz,
      frames=frames,
      # tooclosefunc=tooclose,
      cellsize=cellsize,
      maxstep=50,
      step=4,
      iters=4,
      subiters=4,
      clashiters=0,
      receniters=0,
      clashdis=4 * t + 4,
      contactdis=14,
      contactfrac=0.1,
      cellscalelimit=1.5,
      # vizsphereradius=2,
      towardaxis=False,
      alongaxis=True,
      # vizfresh=False,
      # centerasu=None,
      centerasu='toward_other',
      # centerasu='closert',
      # centerasu_at_start=t > 0.8
      showme=False,
   )
   # wu.showme(slid)

def asuslide_case3():

   sym = 'P213_33'
   xtal = wu.sym.Xtal(sym)
   # cellsize = 99.417
   cellsize = 115

   pdbfile = '/home/sheffler/project/diffusion/unbounded/preslide.pdb'
   pdb = wu.pdb.readpdb(pdbfile).subset(chain='A')
   xyz = pdb.ca()
   fracremains = 1.0
   primaryframes = xtal.primary_frames(cellsize)
   # frames = wu.sym.frames(sym, ontop=primaryframes, cells=(-1, 1), cellsize=cellsize, center=cen, xtalrad=cellsize * 0.5)
   frames = primaryframes
   cfracmin = 0.7
   cfracmax = 0.7
   cdistmin = 14.0
   cdistmax = 14.0
   t = 1
   slid = wu.sym.asuslide(
      sym='P213_33',
      coords=xyz,
      frames=frames,
      # tooclosefunc=tooclose,
      cellsize=cellsize,
      maxstep=100,
      step=4 * t + 2,
      iters=6,
      subiters=4,
      clashiters=0,
      receniters=0,
      clashdis=4 * t + 4,
      contactdis=t * (cdistmax - cdistmin) + cdistmin,
      contactfrac=t * (cfracmax - cfracmin) + cfracmin,
      cellscalelimit=1.5,
      # vizsphereradius=2,
      towardaxis=True,
      alongaxis=False,
      # vizfresh=False,
      # centerasu=None,
      centerasu='toward_other',
      # centerasu='closert',
      # centerasu_at_start=t > 0.8
      showme=False,
   )
   # wu.showme(slid)

def test_asuslide_helix_case1(showme=False):
   showmeopts = wu.Bunch(vizsphereradius=4)

   np.random.seed(7084203)
   xyz = wu.tests.point_cloud(100, std=30, outliers=20)

   h = wu.sym.Helix(turns=15, phase=0.5, nfold=1)
   spacing = 50
   rad = 70
   hgeom = wu.Bunch(radius=rad, spacing=spacing, turns=2)
   cellsize = [hgeom.radius, hgeom.radius, hgeom.spacing]
   rb1 = wu.sym.helix_slide(h, xyz, cellsize, iters=0, closest=9)
   rb2 = wu.sym.helix_slide(h, xyz, cellsize, contactfrac=0.1, closest=9)
   # rb3 = wu.sym.helix_slide(h, xyz, cellsize, contactfrac=0.1, closest=20)
   # ic(cellsize, rb1.cellsize, rb2.cellsize, rb3.cellsize)
   # assert 0

   # wu.showme(rb1, **showmeopts)
   # wu.showme(rb2, **showmeopts)
   # wu.showme(rb3, **showmeopts)

   # ic(rb1.cellsize)
   ic(rb2.cellsize)
   assert np.allclose(rb1.cellsize, [70, 70, 50])
   # assert np.allclose(rb2.cellsize, rb3.cellsize)
   assert np.allclose(rb2.cellsize, [113.7143553, 113.7143553, 44.31469973])

def test_asuslide_helix_nfold1(showme=False):
   showmeopts = wu.Bunch(vizsphereradius=4)

   np.random.seed(7)
   xyz = wu.tests.point_cloud(100, std=30, outliers=0)

   h = wu.sym.Helix(turns=15, phase=0.5, nfold=1)
   spacing = 70
   rad = h.turns * 0.8 * h.nfold * spacing / 2 / np.pi
   hgeom = wu.Bunch(radius=rad, spacing=spacing, turns=2)
   cellsize = [hgeom.radius, hgeom.radius, hgeom.spacing]

   rb1 = wu.sym.helix_slide(h, xyz, cellsize, iters=0, closest=9)
   rb2 = wu.sym.helix_slide(h, xyz, cellsize, contactfrac=0.1, closest=9, showme=False, step=5)
   rb3 = wu.sym.helix_slide(h, xyz, cellsize, contactfrac=0.1, closest=20, step=5)

   # ic(cellsize, rb1.cellsize, rb2.cellsize, rb3.cellsize)
   # assert 0

   # wu.showme(rb1, **showmeopts)
   # wu.showme(rb2, **showmeopts)
   # wu.showme(rb3, **showmeopts)

   ic(rb1.cellsize)
   ic(rb2.cellsize)
   ic(rb3.cellsize)
   assert np.allclose(rb1.cellsize, [133.6901522, 133.6901522, 70.])
   assert np.allclose(rb2.cellsize, rb3.cellsize)
   assert np.allclose(rb2.cellsize, [109.21284284, 109.21284284, 43.59816075])

def test_asuslide_helix_nfold1_2():
   showmeopts = wu.Bunch(vizsphereradius=6)

   np.random.seed(7)
   xyz = wu.tests.point_cloud(100, std=30, outliers=0)

   h = wu.sym.Helix(turns=8, phase=0.5, nfold=1)
   spacing = 70
   rad = h.turns * spacing / 2 / np.pi * 1.3
   hgeom = wu.Bunch(radius=rad, spacing=spacing, turns=2)
   cellsize = [hgeom.radius, hgeom.radius, hgeom.spacing]

   rb1 = wu.sym.helix_slide(h, xyz, cellsize, iters=0, closest=20)
   rb2 = wu.sym.helix_slide(h, xyz, cellsize, contactfrac=0.3, closest=20, steps=30, step=8.7, iters=5, showme=False,
                            **showmeopts)

   # ic(rb2.frames())

   # wu.showme(rb1, **showmeopts)
   # wu.showme(rb2, **showmeopts)

   ic(rb1.cellsize)
   ic(rb2.cellsize)
   assert np.allclose(rb1.cellsize, [115.86479857, 115.86479857, 70.])
   assert np.allclose(rb2.cellsize, [55.93962805, 55.93962805, 38.53925788])

def test_asuslide_helix_nfold3():
   showmeopts = wu.Bunch(vizsphereradius=4)

   np.random.seed(7)
   xyz = wu.tests.point_cloud(100, std=30, outliers=0)

   h = wu.sym.Helix(turns=6, phase=0.5, nfold=3)
   spacing = 50
   rad = h.turns * spacing / 2 / np.pi
   hgeom = wu.Bunch(radius=rad, spacing=spacing, turns=2)
   cellsize = [hgeom.radius, hgeom.radius, hgeom.spacing]

   rb1 = wu.sym.helix_slide(h, xyz, cellsize, iters=0, closest=20)
   rb2 = wu.sym.helix_slide(h, xyz, cellsize, contactfrac=0.1, closest=20, step=10, iters=5, showme=False)

   # wu.showme(rb1, **showmeopts)
   # wu.showme(rb2, **showmeopts)

   # ic(rb1.cellsize)
   ic(rb2.cellsize)
   assert np.allclose(rb1.cellsize, [47.74648293, 47.74648293, 50.])
   assert np.allclose(rb2.cellsize, [44.70186644, 44.70186644, 146.78939426])

def test_asuslide_helix_nfold5():
   showmeopts = wu.Bunch(vizsphereradius=4)

   np.random.seed(7)
   xyz = wu.tests.point_cloud(100, std=30, outliers=0)

   h = wu.sym.Helix(turns=4, phase=0.1, nfold=5)
   spacing = 40
   rad = h.turns * h.nfold * spacing / 2 / np.pi
   hgeom = wu.Bunch(radius=rad, spacing=spacing, turns=2)
   cellsize = [hgeom.radius, hgeom.radius, hgeom.spacing]

   rb = wu.sym.helix_slide(h, xyz, cellsize, iters=0, closest=0)
   rb2 = wu.sym.helix_slide(h, xyz, cellsize, contactfrac=0.1, closest=9)
   rb3 = wu.sym.helix_slide(h, xyz, rb2.cellsize, iters=0, closest=0)

   # wu.showme(rb, **showmeopts)
   # wu.showme(rb2, **showmeopts)
   # wu.showme(rb3, **showmeopts)

   ic(rb.cellsize)
   ic(rb2.cellsize)
   assert np.allclose(rb.cellsize, [127.32395447, 127.32395447, 40.])
   assert np.allclose(rb2.cellsize, [153.14643468, 153.14643468, 49.28047224])
   assert np.allclose(rb3.cellsize, rb2.cellsize)

def test_asuslide_L442():
   sym = 'L4_42'
   xtal = wu.sym.Xtal(sym)
   csize = 160
   np.random.seed(7)
   xyz = wu.tests.point_cloud(100, std=30, outliers=0)
   asucen = xtal.asucen(use_olig_nbrs=True, cellsize=csize)
   xyz += wu.hvec(asucen)
   xyz[:, 1] -= 2

   # pdbfile = '/home/sheffler/project/diffusion/unbounded/step10Bsym.pdb'
   # pdb = wu.pdb.readpdb(pdbfile).subset(chain='A')
   # xyz = pdb.ca()

   primary_frames = np.stack([
      wu.hscaled(csize, np.eye(4)),
      xtal.symelems[0].operators[1],
      xtal.symelems[0].operators[2],
      xtal.symelems[0].operators[3],
      xtal.symelems[1].operators[1],
   ])
   primary_frames = wu.hscaled(csize, primary_frames)
   frames = primary_frames

   slid = asuslide(sym, xyz, frames, showme=False, maxstep=30, step=10, iters=10, clashiters=0, clashdis=8,
                   contactdis=16, contactfrac=0.2, vizsphereradius=2, cellsize=csize, towardaxis=True, alongaxis=False,
                   vizfresh=False, centerasu=False, cellscalelimit=1.2)
   # wu.showme(slid)
   ic(slid.cellsize, slid.asym.com())
   assert np.allclose(slid.cellsize, 99.16625977)
   assert np.allclose(slid.asym.com(), [2.86722158e+01, -1.14700730e+00, 4.03010958e-16, 1.00000000e+00])

def test_asuslide_I4132_clashframes():
   sym = 'I4132_322'
   xtal = wu.sym.Xtal(sym)
   csize = 200
   np.random.seed(7)
   xyz = wu.tests.point_cloud(100, std=30, outliers=0)
   asucen = xtal.asucen(use_olig_nbrs=True, cellsize=csize)
   xyz += wu.hvec(asucen)
   xyz[:, :3] -= 2

   primaryframes = np.stack([
      wu.hscaled(csize, np.eye(4)),
      xtal.symelems[0].operators[1],
      xtal.symelems[0].operators[2],
      xtal.symelems[1].operators[1],
      xtal.symelems[2].operators[1],
   ])

   primaryframes = wu.hscaled(csize, primaryframes)
   frames = wu.sym.frames(sym, ontop=primaryframes, cells=(-1, 1), cellsize=csize, center=wu.hcom(xyz),
                          xtalrad=csize * 0.5)
   # frames = primaryframes

   tooclose = ft.partial(wu.rigid.tooclose_primary_overlap, nprimary=len(primaryframes))
   # tooclose = wu.rigid.tooclose_overlap

   slid = asuslide(sym, xyz, frames, showme=False, maxstep=30, step=5, iters=5, clashiters=0, clashdis=8, contactdis=16,
                   contactfrac=0.3, vizsphereradius=6, cellsize=csize, towardaxis=True, alongaxis=False, vizfresh=False,
                   centerasu=False)  #, tooclosefunc=tooclose)
   # xtal.dump_pdb('test0.pdb', slid.asym.coords, cellsize=slid.cellsize, cells=0)
   # xtal.dump_pdb('test1.pdb', slid.asym.coords, cellsize=slid.cellsize, cells=(-1, 0), ontop='primary')
   # wu.showme(slid)
   ic(slid.cellsize, slid.asym.com())
   assert np.allclose(slid.cellsize, 180.390625)
   assert np.allclose(slid.asym.com(), [-4.80305991, 11.55346709, 28.23302801, 1.])

   slid = asuslide(sym, xyz, frames, showme=False, maxstep=30, step=5, iters=5, clashiters=0, clashdis=8, contactdis=16,
                   contactfrac=0.2, vizsphereradius=6, cellsize=csize, towardaxis=True, alongaxis=False, vizfresh=False,
                   centerasu=False, tooclosefunc=tooclose)
   # xtal.dump_pdb('test0.pdb', slid.asym.coords, cellsize=slid.cellsize, cells=0)
   # xtal.dump_pdb('test1.pdb', slid.asym.coords, cellsize=slid.cellsize, cells=(-1, 0), ontop='primary')
   # wu.showme(slid)
   ic(slid.cellsize, slid.asym.com())
   assert np.allclose(slid.cellsize, 241.25)
   assert np.allclose(slid.asym.com(), [-3.44916815, 14.59051223, 37.75725345, 1.])

   # assert 0

def asuslide_case2():
   sym = 'I4132_322'
   xtal = wu.sym.Xtal(sym)
   # cellsize = 99.417
   cellsize = 115

   pdbfile = '/home/sheffler/project/diffusion/unbounded/step12Bsym.pdb'
   pdb = wu.pdb.readpdb(pdbfile).subset(chain='A')
   xyz = pdb.ca()
   fracremains = 1.0
   primaryframes = xtal.primary_frames(cellsize)
   # frames = wu.sym.frames(sym, ontop=primaryframes, cells=(-1, 1), cellsize=cellsize, center=cen, xtalrad=cellsize * 0.5)
   frames = primaryframes
   slid = wu.sym.asuslide(
      sym=sym,
      coords=xyz,
      showme=True,
      frames=xtal.primary_frames(cellsize),
      cellsize=cellsize,
      maxstep=100,
      step=6 * fracremains + 2,
      iters=6,
      clashiters=0,
      receniters=3,
      clashdis=4 * fracremains + 2,
      contactdis=8 * fracremains + 8,
      contactfrac=fracremains * 0.3 + 0.3,
      # vizsphereradius=2,
      towardaxis=True,
      alongaxis=False,
      # vizfresh=False,
      # centerasu=None,
      centerasu='toward_other',
      # centerasu='closert',
      # centerasu_at_start=fracremains > 0.8
      # showme=True,
   )

   assert 0

def asuslide_case1():
   sym = 'I4132_322'
   xtal = wu.sym.Xtal(sym)
   # csize = 20
   # fname = '/home/sheffler/src/willutil/step2A.pdb'
   fname = '/home/sheffler/project/diffusion/unbounded/step-9Ainput.pdb'
   pdb = wu.pdb.readpdb(fname)
   chainA = pdb.subset(chain='A')
   chainD = pdb.subset(chain='D')

   cachains = pdb.ca().reshape(xtal.nprimaryframes, -1, 4)
   csize = wu.hnorm(wu.hcom(chainD.ca()) * 2)
   ic(csize)
   csize, shift = xtal.fit_coords(cachains, noshift=True)
   ic(csize)

   # xyz = np.stack([pdb.df['x'], pdb.df['y'], pdb.df['z']]).T
   xyz = chainA.ca()
   # xyz = pdb.ca()
   # xyz = xyz[:, :4].reshape(-1, 3)
   # ic(xyz.shape)

   # primary_frames = np.stack([
   # wu.hscaled(csize, np.eye(4)),
   # xtal.symelems[0].operators[1],
   # xtal.symelems[0].operators[2],
   # xtal.symelems[1].operators[1],
   # ])
   # primary_frames = wu.hscaled(csize, primary_frames)
   primary_frames = xtal.primary_frames(cellsize=csize)
   frames = primary_frames

   slid = asuslide(
      sym,
      xyz,
      frames,
      showme=True,
      printme=False,
      maxstep=100,
      step=10,
      iters=6,
      clashiters=0,
      receniters=3,
      clashdis=8,
      contactdis=16,
      contactfrac=0.5,
      vizsphereradius=2,
      cellsize=csize,
      towardaxis=True,
      alongaxis=False,
      vizfresh=False,
      centerasu='toward_other',
      centerasu_at_start=True,
   )
   ic(slid.cellsize)
   assert 0
   # x = wu.sym.Xtal(sym)
   # x.dump_pdb('test.pdb', slid.asym.coords, cellsize=slid.cellsize)
   # print(x)
   # ic(wu.hcart3(slid.asym.globalposition))
   # assert np.allclose(slid.cellsize, 262.2992230399999)
   # assert np.allclose(wu.hcart3(slid.asym.globalposition), [67.3001427, 48.96971455, 60.86220864])

def test_asuslide_I213():
   sym = 'I213'
   xtal = wu.sym.Xtal(sym)
   csize = 200
   np.random.seed(7)
   xyz = wu.tests.point_cloud(100, std=30, outliers=0)
   asucen = xtal.asucen(method='closest', use_olig_nbrs=True, cellsize=csize)
   # asucen = xtal.asucen(method='stored', cellsize=csize)
   xyz += wu.hvec(asucen)
   # xyz[:, 1] -= 2

   # wu.showme(wu.rigid.RigidBodyFollowers(sym=sym, coords=xyz, cellsize=csize, xtalrad=0.7))
   # assert 0

   # primary_frames = np.stack([
   # wu.hscaled(csize, np.eye(4)),
   # xtal.symelems[0].operators[1],
   # xtal.symelems[0].operators[2],
   # xtal.symelems[1].operators[1],
   # ])
   # primary_frames = wu.hscaled(csize, primary_frames)
   frames = None  #xtal.primary_frames(cellsize=csize)

   slid = asuslide(sym, xyz, showme=False, frames=frames, maxstep=20, step=10, iters=10, clashiters=0, clashdis=8,
                   contactdis=16, contactfrac=0.2, vizsphereradius=6, cellsize=csize, towardaxis=True, alongaxis=False,
                   vizfresh=False, centerasu=False, xtalrad=0.3)
   # asym = wu.rigid.RigidBodyFollowers(sym=sym, coords=slid.asym.coords, cellsize=slid.cellsize,
   # frames=xtal.primary_frames(cellsize=slid.cellsize))
   # x = wu.sym.Xtal(sym)
   # x.dump_pdb('test.pdb', slid.asym.coords, cellsize=slid.cellsize)
   # print(x)
   # wu.showme(slid, vizsphereradius=6)
   # wu.showme(asym, vizsphereradius=6)

   ic(slid.cellsize)
   ic(slid.asym.com())
   assert np.allclose(slid.cellsize, 131.68197632)
   assert np.allclose(slid.asym.com(), [75.62589092, 49.98626644, 84.08747631, 1.])

   # frames = wu.sym.frames(sym, ontop=primary_frames, cells=(-1, 1), cellsize=csize, center=asucen, asucen=asucen,
   # xtalrad=csize * 0.5)
   # slid2 = asuslide(sym, xyz, frames, showme=False, maxstep=50, step=10, iters=10, clashiters=0, clashdis=8,
   # contactdis=16, contactfrac=0.2, vizsphereradius=2, cellsize=csize, extraframesradius=1.5 * csize,
   # towardaxis=True, alongaxis=False, vizfresh=False, centerasu=False)

def test_asuslide_L632():
   sym = 'L6_32'
   xtal = wu.sym.Xtal(sym)
   csize = 160
   np.random.seed(7)
   xyz = wu.tests.point_cloud(100, std=30, outliers=0)
   asucen = xtal.asucen(use_olig_nbrs=True, cellsize=csize)
   xyz += wu.hvec(asucen)
   xyz[:, 1] -= 2

   primary_frames = np.stack([
      wu.hscaled(csize, np.eye(4)),
      xtal.symelems[0].operators[1],
      xtal.symelems[0].operators[2],
      xtal.symelems[1].operators[1],
   ])
   primary_frames = wu.hscaled(csize, primary_frames)
   frames = primary_frames

   slid = asuslide(sym, xyz, frames, showme=False, maxstep=30, step=10, iters=10, clashiters=0, clashdis=8,
                   contactdis=16, contactfrac=0.2, vizsphereradius=2, cellsize=csize, towardaxis=True, alongaxis=False,
                   vizfresh=False, centerasu=False)
   # wu.showme(slid)
   ic(slid.cellsize, slid.asym.com())
   assert np.allclose(slid.cellsize, 89.95727539)
   assert np.allclose(slid.asym.com(), [2.38292265e+01, -1.00347068e+00, 3.69426711e-16, 1.00000000e+00])

def test_asuslide_I4132():
   sym = 'I4132_322'
   xtal = wu.sym.Xtal(sym)
   csize = 360
   np.random.seed(7)
   xyz = wu.tests.point_cloud(100, std=30, outliers=0)
   asucen = xtal.asucen(use_olig_nbrs=True, cellsize=csize)
   xyz += wu.hvec(asucen)
   xyz[:, 1] -= 2

   primary_frames = np.stack([
      wu.hscaled(csize, np.eye(4)),
      xtal.symelems[0].operators[1],
      xtal.symelems[0].operators[2],
      xtal.symelems[1].operators[1],
      xtal.symelems[2].operators[1],
   ])
   primary_frames = wu.hscaled(csize, primary_frames)
   frames = primary_frames

   slid = asuslide(sym, xyz, frames, showme=False, maxstep=30, step=5, iters=3, clashiters=0, clashdis=8, contactdis=16,
                   contactfrac=0.2, vizsphereradius=2, cellsize=csize, towardaxis=True, alongaxis=False, vizfresh=False,
                   centerasu=False)
   # wu.showme(slid, vizsphereradius=6)
   ic(slid.cellsize)
   ic(slid.asym.com())
   # ic(wu.hcart3(slid.asym.globalposition))
   # x = wu.sym.Xtal(sym)
   # x.dump_pdb('test.pdb', slid.asym.coords, cellsize=slid.cellsize)
   assert np.allclose(slid.cellsize, 196.875)
   assert np.allclose(slid.asym.com(), [-5.5324486, 12.47020588, 32.41759047, 1.])

   slid2 = asuslide(sym, xyz, showme=False, maxstep=50, step=10, iters=10, clashiters=0, clashdis=8, contactdis=16,
                    contactfrac=0.2, vizsphereradius=2, cellsize=csize, towardaxis=True, alongaxis=False,
                    vizfresh=False, centerasu=False, xtalrad=0.3)
   # wu.showme(slid2)
   # ic(slid.cellsize)
   # ic(slid.asym.com())
   # ic(wu.hcart3(slid.asym.globalposition))
   assert np.allclose(slid.cellsize, 196.875)
   assert np.allclose(slid.asym.com(), [-5.5324486, 12.47020588, 32.41759047, 1.])

def test_asuslide_p213():
   sym = 'P 21 3'
   xtal = wu.sym.Xtal(sym)
   csize = 180
   np.random.seed(7)
   xyz = wu.tests.point_cloud(100, std=30, outliers=0)
   asucen = xtal.asucen(use_olig_nbrs=True, cellsize=csize)
   xyz += wu.hvec(asucen)
   xyz[:, 1] -= 2

   primary_frames = xtal.primary_frames(cellsize=csize)
   slid = asuslide(showme=0, sym=sym, coords=xyz, frames=primary_frames, maxstep=30, step=7, iters=5, subiters=3,
                   contactdis=16, contactfrac=0.1, vizsphereradius=6, cellsize=csize, towardaxis=True, alongaxis=False,
                   vizfresh=False, centerasu=False)
   # wu.showme(slid)
   # slid.dump_pdb('test1.pdb')
   # ic(slid.bvh_op_count, len(slid.bodies))
   ic(slid.cellsize, slid.asym.com())
   assert np.allclose(slid.cellsize, 161.5703125)
   assert np.allclose(slid.asym.com(), [81.45648685, 41.24336469, 62.20570401, 1.])

   frames = xtal.frames(cells=(-1, 1), cellsize=csize, xtalrad=0.9)

   slid = asuslide(showme=0, sym=sym, coords=xyz, frames=frames, maxstep=30, step=7, iters=5, subiters=3, contactdis=16,
                   contactfrac=0.1, vizsphereradius=6, cellsize=csize, towardaxis=True, alongaxis=False, vizfresh=False,
                   centerasu=False)
   # slid.dump_pdb('test2.pdb')
   # ic(slid.bvh_op_count, len(slid.bodies))
   ic(slid.cellsize, slid.asym.com())
   assert np.allclose(slid.cellsize, 161.5703125)
   assert np.allclose(slid.asym.com(), [81.45648685, 41.24336469, 62.20570401, 1.])

   slid = asuslide(showme=0, sym=sym, coords=xyz, maxstep=30, step=7, iters=5, subiters=3, contactdis=16,
                   contactfrac=0.1, vizsphereradius=6, cellsize=csize, towardaxis=True, alongaxis=False, vizfresh=False,
                   centerasu=False)
   slid.dump_pdb('test3.pdb')
   # ic(slid.bvh_op_count, len(slid.bodies))
   # ic(slid.cellsize, slid.asym.com())
   # ic(wu.hcart3(slid.asym.globalposition))
   # ic(slid.asym.tolocal)
   assert np.allclose(slid.cellsize, 161.5703125)
   assert np.allclose(slid.asym.com(), [81.45648685, 41.24336469, 62.20570401, 1.])

def test_asuslide_oct():
   sym = 'oct'
   ax2 = wu.sym.axes(sym)[2]
   ax3 = wu.sym.axes(sym)[3]
   # axisinfo = [(2, ax2, (2, 3)), (3, ax3, 1)]
   axesinfo = [(ax2, [0, 0, 0]), (ax3, [0, 0, 0])]
   primary_frames = [np.eye(4), wu.hrot(ax2, 180), wu.hrot(ax3, 120), wu.hrot(ax3, 240)]
   # frames = primary_frames
   frames = wu.sym.frames(sym, ontop=primary_frames)

   np.random.seed(7)
   xyz = wu.tests.point_cloud(100, std=30, outliers=0)
   xyz += ax2 * 20
   xyz += ax3 * 20
   xyz0 = xyz.copy()

   slid = asuslide(showme=0, sym=sym, coords=xyz, frames=frames, axes=axesinfo, alongaxis=True, towardaxis=False,
                   iters=3, subiters=3, contactfrac=0.1, contactdis=16, vizsphereradius=6)
   ic(slid.asym.com(), slid.cellsize)
   assert np.all(np.abs(slid.frames()[:, :3, 3]) < 0.0001)
   assert np.allclose(slid.asym.com(), [67.39961966, 67.39961966, 25.00882048, 1.])
   assert np.allclose(slid.cellsize, [1, 1, 1])
   assert np.allclose(np.eye(3), slid.asym.position[:3, :3])
   # slid.dump_pdb('ref.pdb')

   slid = asuslide(showme=0, sym=sym, coords=xyz, frames=primary_frames, axes=axesinfo, alongaxis=True,
                   towardaxis=False, iters=3, subiters=3, contactfrac=0.1, contactdis=16, vizsphereradius=6)
   # ic(slid.asym.com(), slid.cellsize)
   assert np.all(np.abs(slid.frames()[:, :3, 3]) < 0.0001)
   assert np.allclose(slid.asym.com(), [67.39961966, 67.39961966, 25.00882048, 1.])
   assert np.allclose(slid.cellsize, [1, 1, 1])
   assert np.allclose(np.eye(3), slid.asym.position[:3, :3])
   # slid.dump_pdb('test0.pdb')

   xyz = xyz0 - ax2 * 30
   slid2 = asuslide(showme=0, sym=sym, coords=xyz, frames=frames, alongaxis=True, vizsphereradius=6, contactdis=12,
                    contactfrac=0.1, maxstep=20, iters=3, subiters=3, towardaxis=False, along_extra_axes=[[0, 0, 1]])
   # ic(slid.asym.com(), slid.cellsize)
   assert np.all(np.abs(slid2.frames()[:, :3, 3]) < 0.0001)
   assert np.allclose(np.eye(3), slid2.asym.position[:3, :3])
   assert np.allclose(slid.asym.com(), [67.39961966, 67.39961966, 25.00882048, 1.])
   # slid.dump_pdb('test1.pdb')

   xyz = xyz0 - ax2 * 20
   slid2 = asuslide(showme=0, sym=sym, coords=xyz, frames=primary_frames, alongaxis=True, vizsphereradius=6,
                    contactdis=12, contactfrac=0.1, maxstep=20, iters=3, subiters=3, towardaxis=False)
   # ic(slid.asym.com(), slid.cellsize)
   assert np.all(np.abs(slid2.frames()[:, :3, 3]) < 0.0001)
   assert np.allclose(np.eye(3), slid2.asym.position[:3, :3])
   assert np.allclose(slid.asym.com(), [67.39961966, 67.39961966, 25.00882048, 1.])
   # slid.dump_pdb('test2.pdb')

def test_asuslide_P432_44(showme=False):
   sym = 'P_4_3_2'
   xtal = wu.sym.Xtal(sym)
   csize = 180
   np.random.seed(7)
   xyz = wu.tests.point_cloud(100, std=30, outliers=0)
   asucen = xtal.asucen(method='stored', cellsize=csize)
   xyz += wu.hvec(asucen)
   primary_frames = xtal.primary_frames(cellsize=csize)
   cen = wu.hcom(xyz)
   frames = wu.sym.frames(sym, ontop=primary_frames, cells=(-1, 1), cellsize=csize, center=cen, asucen=asucen,
                          xtalrad=0.5, strict=False)

   # rbprimary = wu.RigidBodyFollowers(coords=xyz, frames=primary_frames)
   # wu.showme(rbprimary)
   # frames = primary_frames

   slid = asuslide(sym, xyz, frames, showme=False, maxstep=10, step=10.123, iters=3, clashiters=0, clashdis=8,
                   contactdis=16, contactfrac=0.2, vizsphereradius=6, cellsize=csize, towardaxis=True, alongaxis=False,
                   vizfresh=False, centerasu=False, iterstepscale=0.5, resetonfail=True)

   # slid = asuslide(sym, xyz, frames, showme=True, maxstep=20, step=5, scalestep=10, iters=5, clashiters=0, clashdis=8,
   # contactdis=16, contactfrac=0.2, vizsphereradius=6, cellsize=csize, towardaxis=True, alongaxis=False,
   # vizfresh=False, centerasu=False, iterstepscale=0.7, resetonfail=True, receniters=2,
   # centerasu_at_start=False)
   # wu.showme(slid)

   # ic(slid.frames())
   # ic(slid.cellsize, slid.asym.com())
   # ic(wu.hcart3(slid.asym.globalposition))
   # x = wu.sym.Xtal(sym)
   # x.dump_pdb(sym + '.pdb', slid.asym.coords, cellsize=slid.cellsize)

   assert np.allclose(slid.cellsize, 144.5695)
   assert np.allclose(wu.hcart3(slid.asym.globalposition), [1.0123, 0.62081512, 0.93117363])
   assert np.allclose(slid.asym.com(), [19.0123, 36.62081512, 54.93117363, 1])

   # cen = asucen
   # cen = wu.hcom(xyz + [0, 0, 0, 0])
   # frames = wu.sym.frames(sym, ontop=primary_frames, cells=(-1, 1), cellsize=csize, center=asucen, asucen=cen,
   # xtalrad=csize * 0.7)
   # ic(len(frames))
   # assert 0

def test_asuslide_P432_43(showme=False):
   sym = 'P_4_3_2_43'
   xtal = wu.sym.Xtal(sym)
   csize = 180
   np.random.seed(7)
   xyz = wu.tests.point_cloud(100, std=30, outliers=0)
   asucen = xtal.asucen(method='stored', cellsize=csize)
   xyz += wu.hvec(asucen)
   primary_frames = xtal.primary_frames(cellsize=csize)
   cen = wu.hcom(xyz)
   frames = wu.sym.frames(sym, ontop=primary_frames, cells=(-1, 1), cellsize=csize, center=cen, asucen=asucen,
                          xtalrad=0.6, strict=False)

   slid = asuslide(sym, xyz, frames, showme=showme, maxstep=10, step=10.123, iters=3, clashiters=0, clashdis=8,
                   contactdis=16, contactfrac=0.2, vizsphereradius=6, cellsize=csize, towardaxis=True, alongaxis=False,
                   vizfresh=False, centerasu=False, iterstepscale=0.5, resetonfail=True)
   # wu.showme(slid)
   # ic(slid.cellsize)
   # ic(slid.asym.com())
   # ic(wu.hcart3(slid.asym.globalposition))
   # x = wu.sym.Xtal(sym)
   # x.dump_pdb(sym + '.pdb', slid.asym.coords, cellsize=slid.cellsize)

   assert np.allclose(slid.cellsize, 147.10025)
   assert np.allclose(wu.hcart3(slid.asym.globalposition), [1.0123, 2.0246, 3.0369])
   assert np.allclose(slid.asym.com(), [19.0123, 38.0246, 57.0369, 1.])

def test_asuslide_F432():
   sym = 'F_4_3_2'
   xtal = wu.sym.Xtal(sym)
   csize = 300
   np.random.seed(7)
   xyz = wu.tests.point_cloud(100, std=30, outliers=0)
   asucen = xtal.asucen(method='stored', cellsize=csize)
   # ic(asucen)
   xyz += wu.hvec(asucen)

   # frames = wu.sym.frames(sym, cells=None, cellsize=csize)
   frames = wu.sym.frames(sym, cellsize=csize, cen=wu.hcom(xyz), xtalrad=0.3, strict=False)
   # ic(frames.shape)
   # assert 0

   slid = asuslide(sym, xyz, frames, showme=0, maxstep=30, step=10.3, iters=5, subiters=3, contactdis=10,
                   contactfrac=0.1, vizsphereradius=6, cellsize=csize, towardaxis=True, alongaxis=False, vizfresh=False,
                   centerasu=False, along_extra_axes=[[0, 0, 1]], iterstepscale=0.7)
   # wu.showme(slid, vizsphereradius=6)
   # ic(slid.cellsize, slid.asym.com())
   assert np.allclose(slid.cellsize, 207.3)
   assert np.allclose(slid.asym.com(), [156.96760967, 19.95527561, 75.49680577, 1.])

   # cen = asucen
   # cen = wu.hcom(xyz + [0, 0, 0, 0])
   # frames = wu.sym.frames(sym, ontop=primary_frames, cells=(-1, 1), cellsize=csize, center=asucen, asucen=cen,
   # xtalrad=csize * 0.7)
   # ic(len(frames))
   # assert 0

   # wu.showme(slid2)

def test_asuslide_I432():
   sym = 'I_4_3_2'
   xtal = wu.sym.Xtal(sym)
   csize = 250
   np.random.seed(7)
   xyz = wu.tests.point_cloud(100, std=30, outliers=0)
   # ic(xyz.shape)
   xyz += wu.hvec(xtal.asucen(cellsize=csize))

   # wu.showme(wu.rigid.RigidBodyFollowers(sym=sym, coords=xyz, cellsize=csize, xtalrad=0.7))

   slid = asuslide(sym, xyz, showme=0, maxstep=30, step=10, iters=3, subiters=3, clashiters=0, clashdis=8,
                   contactdis=12, contactfrac=0.1, vizsphereradius=6, cellsize=csize, towardaxis=True, alongaxis=False,
                   vizfresh=False, centerasu=False, along_extra_axes=[], xtalrad=0.4, iterstepscale=0.666)

   # ic(slid.bvh_op_count)
   # wu.showme(slid)
   # ic(slid.cellsize, slid.asym.com())
   assert np.allclose(slid.cellsize, 181.12888)
   assert np.allclose(slid.asym.com(), [52.2085891, 31.69795832, 16., 1.])

   # cen = asucen
   # cen = wu.hcom(xyz + [0, 0, 0, 0])
   # frames = wu.sym.frames(sym, ontop=primary_frames, cells=(-1, 1), cellsize=csize, center=asucen, asucen=cen,
   # xtalrad=csize * 0.7)
   # ic(len(frames))
   # assert 0

   # wu.showme(slid2)

def test_asuslide_from_origin():
   from willutil.tests.testdata.misc.asuslide_misc import test_asuslide_case2_coords

   def boundscheck_L632(bodies):
      return True

   sym = 'L632'
   kw = {'maxstep': 40, 'clashdis': 5.68, 'contactdis': 12.0, 'contactfrac': 0.05, 'cellscalelimit': 1.5}
   csize = 1
   slid = asuslide(showme=1, sym=sym, coords=test_asuslide_case2_coords, axes=None, existing_olig=None, alongaxis=0,
                   towardaxis=True, printme=False, cellsize=csize, isxtal=False, nbrs='auto', doscale=True, iters=2,
                   subiters=2, clashiters=0, receniters=0, step=5.26, scalestep=None, closestfirst=True,
                   centerasu='toward_other', centerasu_at_start=False, scaleslides=1.0, iterstepscale=0.75,
                   coords_to_asucen=False, boundscheck=boundscheck_L632, nobadsteps=True, vizsphereradius=6, **kw)
   ic(slid.asym.com(), slid.cellsize)

def test_asuslide_case2():
   from willutil.tests.testdata.misc.asuslide_misc import test_asuslide_case2_coords

   sym = 'L632'
   kw = {'maxstep': 40, 'clashdis': 5.68, 'contactdis': 12.0, 'contactfrac': 0.05, 'cellscalelimit': 1.5}
   xtal = wu.sym.Xtal(sym)
   csize = 80

   frames = xtal.primary_frames(cellsize=csize)  #xtal.frames(cellsize=csize)
   slid = asuslide(showme=0, sym=sym, coords=test_asuslide_case2_coords, frames=frames, axes=None, existing_olig=None,
                   alongaxis=0, towardaxis=True, printme=False, cellsize=csize, isxtal=False, nbrs='auto', doscale=True,
                   iters=2, subiters=2, clashiters=0, receniters=0, step=5.26, scalestep=None, closestfirst=True,
                   centerasu='toward_other', centerasu_at_start=False, scaleslides=1.0, iterstepscale=0.75,
                   coords_to_asucen=False, nobadsteps=True, vizsphereradius=6, **kw)
   # wu.showme(slid)
   # ic(slid.asym.com(), slid.cellsize)
   assert np.allclose(slid.asym.com(), [18.33744584, 0.30792098, 3.55403141, 1])
   assert np.allclose(slid.cellsize, [58.96, 58.96, 58.96])
   slid = asuslide(showme=0, sym=sym, coords=test_asuslide_case2_coords, axes=None, existing_olig=None, alongaxis=0,
                   towardaxis=True, printme=False, cellsize=csize, isxtal=False, nbrs='auto', doscale=True, iters=2,
                   subiters=2, clashiters=0, receniters=0, step=5.26, scalestep=None, closestfirst=True,
                   centerasu='toward_other', centerasu_at_start=False, scaleslides=1.0, iterstepscale=0.75,
                   coords_to_asucen=False, nobadsteps=True, vizsphereradius=6, **kw)
   # wu.showme(slid)
   # ic(slid.asym.com(), slid.cellsize)
   assert np.allclose(slid.asym.com(), [18.33744584, 0.30792098, 3.55403141, 1])
   assert np.allclose(slid.cellsize, [58.96, 58.96, 58.96])

   # ic(test_asuslide_case2_coords.shape)
   # ic(slid.asym.coords.shape)
   # ic(slid.coords.shape)
   # slid.dump_pdb('ref.pdb')

   def boundscheck_L632(bodies):
      com = bodies.asym.com()
      if com[0] < 0: return False
      if com[0] > 4 and abs(np.arctan2(com[1], com[0])) > np.pi / 6: return False
      com2 = bodies.bodies[3].com()
      if com[0] > com2[0]: return False
      return True

   # coords = test_asuslide_case2_coords
   coords = wu.hcentered(test_asuslide_case2_coords, singlecom=True)
   coords[..., 0] += 5
   # wu.showme(test_asuslide_case2_coords[:, 1])
   # ic(wu.hcom(coords))
   slid = asuslide(showme=0, sym=sym, coords=coords, axes=None, existing_olig=None, alongaxis=0, towardaxis=True,
                   printme=False, cellsize=csize, isxtal=False, nbrs='auto', doscale=True, iters=2, subiters=2,
                   clashiters=0, receniters=0, step=5.26, scalestep=None, closestfirst=True, centerasu='toward_other',
                   centerasu_at_start=False, scaleslides=1.0, iterstepscale=0.75, coords_to_asucen=True,
                   nobadsteps=True, vizsphereradius=6, boundscheck=boundscheck_L632, **kw)
   # slid.dump_pdb('test.pdb')
   # wu.showme(slid)
   ic(slid.asym.com(), slid.cellsize)
   ic('=======')
   # don't know why this is unstable... generally off by a few thou
   assert np.allclose(slid.asym.com(), [1.81500000e+01, -4.17462713e-04, 4.31305757e-15, 1.00000000e+00], atol=0.1)
   assert np.allclose(slid.cellsize, [58.96, 58.96, 58.96], atol=0.01)

   coords = wu.hcentered(test_asuslide_case2_coords, singlecom=True)
   coords[..., 0] += 5
   csize = 10
   slid = asuslide(showme=False, sym=sym, coords=coords, axes=None, existing_olig=None, alongaxis=0, towardaxis=True,
                   printme=False, cellsize=csize, isxtal=False, nbrs='auto', doscale=True, iters=2, subiters=2,
                   clashiters=0, receniters=0, step=5.26, scalestep=None, closestfirst=True, centerasu='toward_other',
                   centerasu_at_start=False, scaleslides=1.0, iterstepscale=0.75, coords_to_asucen=False,
                   nobadsteps=True, vizsphereradius=6, boundscheck=boundscheck_L632, **kw)
   # wu.showme(slid)
   ic(slid.asym.com(), slid.cellsize)
   assert np.allclose(slid.asym.com(), [1.81500000e+01, -4.17462713e-04, 4.31305757e-15, 1.00000000e+00], atol=0.1)

   assert np.allclose(slid.cellsize, 57.34, atol=0.01)

if __name__ == '__main__':
   main()
   print('test_aluslide DONE')
