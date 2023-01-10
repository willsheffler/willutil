import numpy as np, functools as ft
import willutil as wu
from willutil.sym.asuslide import asuslide

ic.configureOutput(includeContext=True, contextAbsPath=True)

def main():
   # asuslide_case2()
   # asuslide_case1()
   # assert 0
   test_asuslide_L442()
   # test_asuslide_I4132_clashframes()
   # test_asuslide_I4132()
   test_asuslide_oct()
   test_asuslide_L632()
   # test_asuslide_p213()
   test_asuslide_I213()

   test_asuslide_helix_nfold1()
   test_asuslide_helix_nfold3()
   test_asuslide_helix_nfold5()

def helix_slide(
   helix,
   coords,
   cellsize,
   turns=2,
   contactfrac=0,
   step=0.03,
   maxstep=10,
   iters=4,
   breathe=[2.5, 1, 0.5, 0],
   showme=False,
   closest=9,
   **kw,
):
   assert np.allclose(cellsize[0], cellsize[1])
   cellsize = cellsize.copy()

   hframes = helix.frames(maxdist=9e9, radius=cellsize[0], spacing=cellsize[2], turns=turns, closest=closest)
   rb = wu.RigidBodyFollowers(coords=coords, frames=hframes, symtype='H', cellsize=cellsize, clashdis=8, contactdis=16)
   hstep = np.array([0.00, 0.00, step])
   rstep = np.array([step, step, 0.00])
   sstep = np.array([step, step, step])
   tooclose = ft.partial(wu.rigid.tooclose_overlap, contactfrac=contactfrac)
   for i, expand in enumerate(breathe):
      for j in range(iters):
         for step in (hstep, rstep):
            scale = (1 + step * expand)
            wu.sym.slide_scale(
               rb,
               cellsize=1,
               step=step,
               tooclosefunc=tooclose,
               showme=showme,
               maxstep=maxstep,
               moveasymunit=False,
               **kw,
            )
            if expand > 0:
               rb.scale_frames(scale)
            if showme:
               wu.showme(rb, **kw)
   return rb

def test_asuslide_helix_nfold1():
   showmeopts = wu.Bunch(sphereradius=4)

   np.random.seed(7)
   xyz = wu.tests.point_cloud(100, std=30, outliers=0)

   h = wu.sym.Helix(turns=15, phase=0.5, nfold=1)
   spacing = 50
   rad = h.turns * h.nfold * spacing / 2 / np.pi
   hgeom = wu.Bunch(radius=rad, spacing=spacing, turns=2)
   cellsize = [hgeom.radius, hgeom.radius, hgeom.spacing]

   rb = helix_slide(h, xyz, cellsize, iters=0, closest=9)
   rb2 = helix_slide(h, xyz, cellsize, contactfrac=0.1, closest=9)
   rb3 = helix_slide(h, xyz, cellsize, contactfrac=0.1, closest=20)

   # ic(rb.)

   # wu.showme(rb, **showmeopts)
   # wu.showme(rb2, **showmeopts)

   # ic(rb.cellsize)
   # ic(rb2.cellsize)
   assert np.allclose(rb.cellsize, [119.36620732, 119.36620732, 50.])
   assert np.allclose(rb2.cellsize, rb3.cellsize)
   assert np.allclose(rb2.cellsize, [112.47430976, 112.47430976, 39.59507625])

def test_asuslide_helix_nfold3():
   showmeopts = wu.Bunch(sphereradius=4)

   np.random.seed(7)
   xyz = wu.tests.point_cloud(100, std=30, outliers=0)

   h = wu.sym.Helix(turns=6, phase=0.5, nfold=3)
   spacing = 50
   rad = h.turns * spacing / 2 / np.pi
   hgeom = wu.Bunch(radius=rad, spacing=spacing, turns=2)
   cellsize = [hgeom.radius, hgeom.radius, hgeom.spacing]

   rb = helix_slide(h, xyz, cellsize, iters=0, closest=20)
   rb2 = helix_slide(h, xyz, cellsize, contactfrac=0.1, closest=20, steps=10, iters=10)

   # wu.showme(rb, **showmeopts)
   # wu.showme(rb2, **showmeopts)
   # ic(rb.cellsize)
   # ic(rb2.cellsize)

   assert np.allclose(rb.cellsize, [47.74648293, 47.74648293, 50.])
   assert np.allclose(rb2.cellsize, [42.38059534, 42.38059534, 140.10179698])

def test_asuslide_helix_nfold5():
   showmeopts = wu.Bunch(sphereradius=4)

   np.random.seed(7)
   xyz = wu.tests.point_cloud(100, std=30, outliers=0)

   h = wu.sym.Helix(turns=4, phase=0.1, nfold=5)
   spacing = 40
   rad = h.turns * h.nfold * spacing / 2 / np.pi
   hgeom = wu.Bunch(radius=rad, spacing=spacing, turns=2)
   cellsize = [hgeom.radius, hgeom.radius, hgeom.spacing]

   rb = helix_slide(h, xyz, cellsize, iters=0, closest=0)
   rb2 = helix_slide(h, xyz, cellsize, contactfrac=0.1, closest=9)
   rb3 = helix_slide(h, xyz, rb2.cellsize, iters=0, closest=0)

   # wu.showme(rb, **showmeopts)
   # wu.showme(rb2, **showmeopts)
   # wu.showme(rb3, **showmeopts)
   # ic(rb.cellsize)
   # ic(rb2.cellsize)

   assert np.allclose(rb.cellsize, [127.32395447, 127.32395447, 40.])
   assert np.allclose(rb2.cellsize, [159.68665321, 159.68665321, 38.81661124])

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
                   contactdis=16, contactfrac=0.2, sphereradius=2, cellsize=csize, towardaxis=True, alongaxis=False,
                   fresh=False, centerasu=False, cellscalelimit=1.2)
   # ic(slid.cellsize)
   # ic(wu.hcart3(slid.asym.globalposition))
   assert np.allclose(slid.cellsize, 98.500111359)
   assert np.allclose(wu.hcart3(slid.asym.globalposition), [-2.15518344e+01, 1.91427678e+00, 0])

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
                          maxdist=csize * 0.5)
   # frames = primaryframes

   tooclose = ft.partial(wu.rigid.tooclose_primary_overlap, nprimary=len(primaryframes))
   # tooclose = wu.rigid.tooclose_overlap

   slid = asuslide(sym, xyz, frames, showme=False, maxstep=30, step=10, iters=10, clashiters=0, clashdis=8,
                   contactdis=16, contactfrac=0.2, sphereradius=2, cellsize=csize, towardaxis=True, alongaxis=False,
                   fresh=False, centerasu=False, tooclosefunc=tooclose)
   xtal.dump_pdb('test0.pdb', slid.asym.coords, cellsize=slid.cellsize, cells=0)
   xtal.dump_pdb('test1.pdb', slid.asym.coords, cellsize=slid.cellsize, cells=(-1, 0), ontop='primary')

   # ic(slid.cellsize)
   # ic(wu.hcart3(slid.asym.globalposition))
   assert np.allclose(slid.cellsize, 220.712217)
   assert np.allclose(wu.hcart3(slid.asym.globalposition), [1.97916456, -1.30299291, 5.65510368])

def asuslide_case2():
   sym = 'I4132_322'
   xtal = wu.sym.Xtal(sym)
   # cellsize = 99.417
   cellsize = 115

   pdbfile = '/home/sheffler/project/diffusion/unbounded/step12Bsym.pdb'
   pdb = wu.pdb.readpdb(pdbfile).subset(chain='A')
   xyz = pdb.ca()
   fracremains = 1.0
   primryframes = xtal.primary_frames(cellsize)
   # frames = wu.sym.frames(sym, ontop=primryframes, cells=(-1, 1), cellsize=cellsize, center=cen, maxdist=cellsize * 0.5)
   frames = primryframes
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
      # sphereradius=2,
      towardaxis=True,
      alongaxis=False,
      # fresh=False,
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
      printme=True,
      maxstep=100,
      step=10,
      iters=6,
      clashiters=0,
      receniters=3,
      clashdis=8,
      contactdis=16,
      contactfrac=0.5,
      sphereradius=2,
      cellsize=csize,
      towardaxis=True,
      alongaxis=False,
      fresh=False,
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
   csize = 100
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

   slid = asuslide(sym, xyz, frames, showme=False, maxstep=50, step=10, iters=10, clashiters=0, clashdis=8,
                   contactdis=16, contactfrac=0.2, sphereradius=2, cellsize=csize, towardaxis=True, alongaxis=False,
                   fresh=False, centerasu=False)
   # ic(slid.cellsize)
   # x = wu.sym.Xtal(sym)
   # x.dump_pdb('test.pdb', slid.asym.coords, cellsize=slid.cellsize)
   # print(x)
   # ic(wu.hcart3(slid.asym.globalposition))
   assert np.allclose(slid.cellsize, 271.61699839999915)
   assert np.allclose(wu.hcart3(slid.asym.globalposition), [73.52211643, 54.76770372, 63.50804849])

   # frames = wu.sym.frames(sym, ontop=primary_frames, cells=(-1, 1), cellsize=csize, center=asucen, asucen=asucen,
   # maxdist=csize * 0.5)
   # slid2 = asuslide(sym, xyz, frames, showme=False, maxstep=50, step=10, iters=10, clashiters=0, clashdis=8,
   # contactdis=16, contactfrac=0.2, sphereradius=2, cellsize=csize, extraframesradius=1.5 * csize,
   # towardaxis=True, alongaxis=False, fresh=False, centerasu=False)

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
                   contactdis=16, contactfrac=0.2, sphereradius=2, cellsize=csize, towardaxis=True, alongaxis=False,
                   fresh=False, centerasu=False)
   # ic(slid.cellsize)
   # ic(wu.hcart3(slid.asym.globalposition))
   assert np.allclose(slid.cellsize, 88.88678400000006)
   assert np.allclose(wu.hcart3(slid.asym.globalposition), [-2.40119093e+01, 1.42260466e+00, -2.03300189e-16])

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

   slid = asuslide(sym, xyz, frames, showme=False, maxstep=30, step=10, iters=10, clashiters=0, clashdis=8,
                   contactdis=16, contactfrac=0.2, sphereradius=2, cellsize=csize, towardaxis=True, alongaxis=False,
                   fresh=False, centerasu=False)
   # ic(slid.cellsize)
   # ic(wu.hcart3(slid.asym.globalposition))
   x = wu.sym.Xtal(sym)
   x.dump_pdb('test.pdb', slid.asym.coords, cellsize=slid.cellsize)

   assert np.allclose(slid.cellsize, 187.67194624000027)
   assert np.allclose(wu.hcart3(slid.asym.globalposition), [2.75203393, -12.89901091, -25.10636146])

   # cen = asucen
   cen = wu.hcom(xyz + [0, 0, 0, 0])
   frames = wu.sym.frames(sym, ontop=primary_frames, cells=(-1, 1), cellsize=csize, center=asucen, asucen=cen,
                          maxdist=csize * 0.5)
   # ic(len(frames))
   # assert 0

   slid2 = asuslide(sym, xyz, frames, showme=False, maxstep=50, step=10, iters=10, clashiters=0, clashdis=8,
                    contactdis=16, contactfrac=0.2, sphereradius=2, cellsize=csize, towardaxis=True, alongaxis=False,
                    fresh=False, centerasu=False)

def test_asuslide_p213():
   sym = 'P 21 3'
   xtal = wu.sym.Xtal(sym)
   csize = 180
   np.random.seed(7)
   xyz = wu.tests.point_cloud(100, std=30, outliers=0)
   asucen = xtal.asucen(use_olig_nbrs=True, cellsize=csize)
   xyz += wu.hvec(asucen)
   xyz[:, 1] -= 2

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
   # wu.showme(xyz)

   primary_frames = np.stack([
      wu.hscaled(csize, np.eye(4)),
      xtal.symelems[0].operators[1],
      xtal.symelems[0].operators[2],
      xtal.symelems[1].operators[1],
      xtal.symelems[1].operators[2],
   ])
   primary_frames = wu.hscaled(csize, primary_frames)

   # frames = wu.sym.frames(sym, ontop=primary_frames, cells=[(-1, 0), (-1, 0), (-1, 0)])
   # frames = wu.sym.frames(sym, ontop=primary_frames, cells=3, center=
   # frames = wu.sym.frames(
   #    sym,
   #    ontop=primary_frames,
   #    cells=(-1, 1),
   #    cellsize=csize,
   #    center=asucen,
   #    asucen=asucen,
   #     maxdist=csize * 0.8,
   # )
   frames = primary_frames

   slid = asuslide(sym, xyz, frames, maxstep=30, step=10, iters=10, clashiters=0, clashdis=8, contactdis=16,
                   contactfrac=0.2, sphereradius=2, cellsize=csize, towardaxis=True, alongaxis=False, fresh=False,
                   centerasu=False)
   # wu.showme(slid)
   assert np.allclose(slid.cellsize, 143.02490624)
   assert np.allclose(wu.hcart3(slid.asym.globalposition), [-19.20408823, -8.02949862, -13.88582565])

def test_asuslide_oct():
   sym = 'oct'
   ax2 = wu.sym.axes(sym)[2]
   ax3 = wu.sym.axes(sym)[3]
   # axisinfo = [(2, ax2, (2, 3)), (3, ax3, 1)]
   axesinfo = [(ax2, [0, 0, 0]), (ax3, [0, 0, 0])]
   primary_frames = [np.eye(4), wu.hrot(ax2, 180), wu.hrot(ax3, 120), wu.hrot(ax3, 240)]
   frames = primary_frames
   # frames = wu.sym.frames(sym, ontop=primary_frames)

   np.random.seed(7)
   xyz = wu.tests.point_cloud(100, std=30, outliers=0)
   xyz += ax2 * 20
   xyz += ax3 * 20

   # slid1 = asuslide(sym, xyz, frames, axes=axesinfo, showme=True, sphereradius=2)
   # slid2 = asuslide(sym, xyz, frames, alongaxis=True, showme=True, sphereradius=2)
   slid1 = asuslide(sym, xyz, frames, axes=axesinfo, alongaxis=True, clashdis=5)
   slid2 = asuslide(sym, xyz, frames, alongaxis=True, sphereradius=2, clashdis=5, showme=False)

   assert np.all(np.abs(slid1.frames()[:, :3, 3]) < 0.0001)
   assert np.allclose(np.eye(3), slid1.asym.position[:3, :3])
   # ic(wu.hcart3(slid1.asym.globalposition))
   assert np.allclose(wu.hcart3(slid1.asym.globalposition), np.array([39.02998607, 39.02998607, 14.36909663]))

   assert np.all(np.abs(slid2.frames()[:, :3, 3]) < 0.0001)
   assert np.allclose(np.eye(3), slid2.asym.position[:3, :3])
   # ic(wu.hcart3(slid2.asym.globalposition))
   assert np.allclose(wu.hcart3(slid2.asym.globalposition), np.array([22.97493765, 33.42625273, 10.78872307]))

if __name__ == '__main__':
   main()
   print('test_aluslide DONE')
