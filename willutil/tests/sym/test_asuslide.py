import numpy as np, functools as ft
import pytest
from numpy import array
import willutil as wu
from willutil.sym.asuslide import asuslide

pytest.skip(allow_module_level=True)

# ic.configureOutput(includeContext=True, contextAbsPath=True)

def main():

   test_asuslide_p213()
   assert 0

   test_asuslide_L632()
   test_asuslide_L632_2()
   test_asuslide_L442()

   test_asuslide_F432()
   test_asuslide_I4132()
   test_asuslide_L632_ignoreimmobile()
   test_asuslide_P432_43()
   test_asuslide_P432_44()
   test_asuslide_I432()
   test_asuslide_I213()
   test_asuslide_p213()

   assert 0
   manual_test()
   assert 0

   test_asuslide_helix_case1()

   test_asuslide_case2()
   test_asuslide_helix_nfold1_2()
   test_asuslide_helix_nfold5()
   test_asuslide_helix_nfold3()
   test_asuslide_helix_nfold1()
   test_asuslide_I4132_clashframes()
   test_asuslide_F432()

   test_asuslide_oct()

   # test_asuslide_case2()
   # assert
   # test_asuslide_helix_nfold1_2()
   # test_asuslide_oct()
   # test_asuslide_L632_2(showme=True)
   # test_asuslide_P432_44(showme=True)
   # test_asuslide_P432_43(showme=True)
   assert 0

   test_asuslide_P432()

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

def manual_test():

   #yapf: disable
   kw = {'maxstep': 10, 'clashdis': 7.2, 'contactdis': 15.2, 'contactfrac': 0.74, 'cellscalelimit': 1.5}
   coords=np.array([[[54.51119,29.409903,41.55905],[55.404022,30.199131,40.71904],[54.80475,31.564907,40.40925]],[[50.993343,30.501368,49.58386],[51.36628,29.573011,50.644382],[52.437237,30.170462,51.548016]],[[50.81645,31.18953,52.002556],[51.652683,30.111223,52.515972],[53.12805,30.487146,52.466667]],[[50.76109,45.387222,36.6931],[49.574318,46.10996,37.134537],[48.44742,45.992607,36.116314]],[[50.402847,30.614996,50.54183],[49.2313,29.935604,51.08177],[48.20577,30.934965,51.601406]],[[50.156784,28.430199,49.708775],[49.649155,28.881092,50.998985],[50.440693,30.07578,51.515347]],[[47.568687,30.346247,45.422497],[47.46511,30.643986,46.845963],[48.83805,30.888628,47.458855]],[[52.039886,26.259457,54.802845],[52.552208,27.210032,53.823273],[52.995693,26.502792,52.549076]],[[49.67048,30.243135,42.89007],[49.205063,29.327225,43.92455],[50.255432,29.145754,45.012768]],[[49.719425,31.44023,49.359226],[49.01798,30.30116,49.938995],[49.176334,30.269579,51.453716]],[[48.65437,32.22256,50.814247],[49.092762,30.833954,50.74214],[50.04858,30.61794,49.575867]],[[46.69054,31.525162,47.757866],[45.838898,30.54201,48.416473],[44.38849,31.006111,48.453102]],[[51.068855,29.403759,47.45503],[50.390015,29.80569,48.681103],[49.35607,30.890503,48.40802]],[[48.490772,30.510437,50.964443],[48.177532,29.455526,50.00806],[46.77778,28.900673,50.238823]],[[29.709919,46.317963,44.71841],[29.99214,44.994907,44.1748],[31.467758,44.644386,44.316673]],[[46.471992,27.998854,51.43776],[46.382023,29.266502,52.152332],[45.766384,29.078108,53.53289]],[[44.214825,36.826454,53.14749],[44.099888,37.321796,54.513897],[45.454407,37.757965,55.05746]],[[40.711914,11.792399,58.443165],[41.376827,12.430169,57.31322],[42.336136,11.46828,56.624073]],[[46.7976,26.363178,50.228165],[46.827892,27.797981,49.971134],[47.848133,28.14319,48.893936]],[[45.634583,31.482237,52.535007],[45.940495,31.430096,51.11046],[44.669273,31.327003,50.277504]],[[46.012135,31.091051,48.3927],[46.917515,30.814217,49.501434],[48.352142,30.65532,49.014587]],[[32.986732,29.966717,47.246475],[33.391026,30.376778,48.585888],[33.66751,31.87366,48.64374]],[[33.502666,14.853637,48.780735],[34.282253,15.957048,48.232674],[35.398876,15.447759,47.33034]],[[40.082268,26.190973,47.255383],[38.87397,26.969202,47.010387],[38.625446,27.96862,48.132816]],[[42.42685,30.589973,50.754574],[43.663067,31.362072,50.790157],[43.3833,32.85528,50.67843]],[[46.398476,29.847504,50.47195],[47.269993,30.610554,51.357304],[46.48524,31.66326,52.12958]],[[46.31977,29.85919,52.44138],[47.42738,30.154331,51.540417],[47.86578,31.607924,51.664196]],[[47.277813,28.991884,48.79292],[47.771015,30.14654,49.533962],[47.298767,31.449516,48.901764]],[[35.191467,44.29714,47.00056],[36.25365,43.88056,47.908237],[36.34115,42.36219,47.993763]],[[47.194786,31.910461,53.616863],[47.9863,31.894869,52.392574],[47.852116,33.208523,51.633137]],[[52.472107,48.008404,52.002567],[51.997284,49.386864,52.003445],[51.12485,49.66818,53.220078]],[[46.7191,32.949257,52.503937],[47.418682,32.156,53.507412],[46.493782,31.120302,54.13377]],[[44.63648,30.289768,50.601242],[46.02256,30.018358,50.962822],[46.491463,30.942041,52.079685]],[[46.951527,32.239796,49.816612],[46.19379,31.021242,49.558556],[45.15348,30.779085,50.64463]],[[44.125675,31.862848,49.362602],[44.744556,30.734777,50.04823],[44.347168,30.698387,51.518333]],[[50.294582,30.517284,48.943455],[49.480213,29.867598,49.96342],[50.303497,29.548111,51.204624]],[[48.466484,30.866653,51.12663],[49.144737,29.634232,50.74354],[49.315804,29.546465,49.232426]],[[45.54389,27.197826,48.88265],[46.21767,28.252945,49.629894],[45.213623,29.156569,50.334023]],[[37.88225,29.6959,67.0285],[38.679813,30.677744,67.75342],[40.06525,30.824993,67.137535]],[[47.23133,45.901657,53.742523],[48.19499,45.414234,54.722023],[47.53759,44.466564,55.71711]],[[45.42423,28.154987,48.459393],[44.12753,28.02204,47.80629],[43.02064,27.77799,48.823975]],[[48.02686,32.317562,34.984047],[48.138798,32.891502,36.319603],[47.498165,31.986053,37.36372]],[[50.919758,29.981022,49.595917],[49.667004,29.89365,50.3366],[49.072655,28.493732,50.250465]],[[46.37101,31.657902,50.985703],[46.723537,30.24435,51.042446],[46.699253,29.614489,49.655674]],[[32.974712,27.645676,42.05848],[33.567818,26.867233,43.13917],[35.06662,27.11941,43.241367]],[[47.956562,31.07771,50.788708],[46.567665,30.865738,50.399258],[45.63312,31.79336,51.165142]],[[51.902977,30.028152,49.67738],[53.127495,29.55674,49.041782],[53.839962,28.529812,49.91255]],[[46.852753,29.845186,41.01078],[47.586807,30.950294,41.61537],[47.195652,31.13794,43.075592]],[[50.55296,30.0599,45.51342],[51.030663,29.056011,46.45663],[51.97746,29.668709,47.480648]],[[49.02594,22.478662,45.896187],[49.528923,21.372475,46.701797],[48.89156,20.053185,46.28507]],[[36.86418,8.222951,68.56943],[36.031624,9.12927,69.351135],[36.801693,9.702785,70.533714]],[[49.51607,27.569708,50.968277],[48.37647,28.102577,51.705154],[47.071953,27.84982,50.96031]],[[49.03591,26.769102,51.488636],[48.115417,27.095097,52.571247],[48.072483,28.596596,52.824417]],[[31.38462,45.965652,66.288315],[31.332184,45.50645,64.905556],[31.090973,46.666813,63.948578]],[[45.181137,29.960491,50.44939],[46.36725,29.119968,50.33844],[47.21862,29.19951,51.59911]],[[48.340427,28.166515,54.500313],[48.18699,29.512632,55.03888],[46.937508,30.18738,54.48756]],[[44.924744,33.17005,57.525112],[45.047295,32.17692,56.46477],[46.036007,31.082758,56.846474]],[[54.441685,28.200527,58.672997],[53.754234,28.352888,57.396343],[52.92843,29.632532,57.364685]],[[46.27058,30.971676,51.169163],[46.57207,32.36717,50.873665],[48.05331,32.55931,50.574577]],[[59.63514,44.87081,51.99519],[59.223736,45.90732,51.05602],[59.68853,45.585022,49.641613]],[[45.67184,16.958412,48.651516],[46.069626,15.567487,48.83244],[46.935852,15.087295,47.675068]],[[45.777363,31.652958,49.56032],[46.656704,30.969278,50.50105],[45.85657,30.17431,51.524902]],[[34.517807,20.905317,44.245853],[35.331497,22.03519,44.6782],[36.038208,21.73371,45.993534]],[[47.48112,33.021828,48.499287],[48.379345,32.262527,49.360832],[47.726837,30.969978,49.834103]],[[51.70997,32.020325,40.994267],[51.622063,30.569407,40.88134],[50.634964,30.160204,39.795696]],[[51.5201,36.703354,50.835514],[51.026,35.575817,50.054382],[51.307514,34.25369,50.756657]],[[56.377827,33.016003,42.05814],[55.78306,31.786804,42.56897],[54.562286,32.08233,43.43083]],[[47.33628,30.031137,49.454967],[47.91413,28.933064,48.689476],[46.915096,28.378067,47.68233]],[[35.940598,37.360428,46.751423],[35.84982,36.12063,45.98966],[34.535557,35.40081,46.263557]],[[43.012424,34.40379,27.344557],[42.504784,33.7293,28.533257],[41.05034,33.314266,28.352234]],[[39.955437,35.716084,47.293755],[41.04108,36.657764,47.539215],[40.79615,37.461803,48.809635]],[[46.53096,30.68734,50.755527],[47.731167,30.07678,50.196632],[47.46222,29.499454,48.812866]],[[48.40207,30.523767,49.375404],[47.40459,30.65515,48.320217],[48.00142,31.298399,47.075043]],[[48.00724,29.411568,47.87233],[46.656746,29.601315,48.38787],[45.624672,29.525154,47.270073]],[[54.733658,41.58862,26.833849],[54.15404,42.55312,27.7609],[54.909096,42.567955,29.083818]]],dtype=np.float32)
   # yapf: enable
   frames = None
   slid = asuslide(
      sym='I213_32',
      coords=coords,
      frames=frames,
      axes=None,
      existing_olig=None,
      alongaxis=0,
      towardaxis=True,
      printme=False,
      cellsize=array([80.33195401, 80.33195401, 80.33195401]),
      isxtal=False,
      nbrs='auto',
      doscale=True,
      iters=1,
      subiters=1,
      clashiters=0,
      receniters=0,
      step=10,
      scalestep=None,
      closestfirst=True,
      centerasu='toward_other',
      centerasu_at_start=False,
      showme=True,
      scaleslides=1.0,
      iterstepscale=0.75,
      coords_to_asucen=False,
      xtalrad=0.6,
      vizsphereradius=6,
      # ignoreimmobile=True,
      **kw,
   )
   wu.showme(slid)

   ic(slid.asym.com())
   ic(slid.cellsize)

   wu.showme(slid)
   slid.dump_pdb(f'/home/sheffler/DEBUG_slid_willutil.pdb')

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
   assert np.allclose(slid.cellsize, 101.875)
   assert np.allclose(slid.asym.com(), [3.02441406e+01, -1.27343750e+00, 3.42139511e-16, 1.00000000e+00])
   # assert np.allclose(slid.cellsize, 95)
   # assert np.allclose(slid.asym.com(), [25.1628825, -1.05965433, 0, 1])

def test_asuslide_L632_ignoreimmobile(showme=False):
   sym = 'L6_32'
   xtal = wu.sym.Xtal(sym)
   csize = 160
   np.random.seed(7)
   xyz = wu.tests.point_cloud(100, std=30, outliers=0)
   asucen = xtal.asucen(use_olig_nbrs=True, cellsize=csize)
   xyz += wu.hvec(asucen)
   xyz[:, 0] -= 30

   frames = xtal.cellframes(cellsize=csize)

   slid = asuslide(sym, xyz, frames, showme=False, maxstep=30, step=9, iters=3, subiters=1, doscale=True,
                   doscaleiters=True, clashiters=0, clashdis=8, contactdis=12, contactfrac=0.1, vizsphereradius=6,
                   cellsize=csize, towardaxis=True, alongaxis=False, vizfresh=False, centerasu=False, scaleslides=1,
                   resetonfail=True, nobadsteps=False, ignoreimmobile=True, iterstepscale=0.5)
   # wu.showme(slid, vizsphereradius=6)
   ic(slid.cellsize)
   ic(slid.asym.com())
   assert np.allclose(slid.cellsize, 90.25)
   assert np.allclose(slid.asym.com(), [2.37973395e+01, 1.44932026e-15, 7.30711512e-16, 1.00000000e+00])

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
   # asucen = xtal.asucen(method='closest', use_olig_nbrs=True, cellsize=csize)
   asucen = xtal.asucen(method='stored', cellsize=csize)
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

   slid = asuslide(sym, xyz, showme=False, frames=frames, maxstep=13, step=10, iters=3, clashiters=0, clashdis=8,
                   contactdis=16, contactfrac=0.1, vizsphereradius=6, cellsize=csize, towardaxis=True, alongaxis=False,
                   vizfresh=False, centerasu=False, xtalrad=0.6, iterstepscale=0.5)
   # asym = wu.rigid.RigidBodyFollowers(sym=sym, coords=slid.asym.coords, cellsize=slid.cellsize,
   # frames=xtal.primary_frames(cellsize=slid.cellsize))
   # x = wu.sym.Xtal(sym)
   # x.dump_pdb('test.pdb', slid.asym.coords, cellsize=slid.cellsize)
   # print(x)
   # wu.showme(slid, vizsphereradius=6)
   # wu.showme(asym, vizsphereradius=6)

   ic(slid.cellsize)
   ic(slid.asym.com())
   assert np.allclose(slid.cellsize, 142.5)
   assert np.allclose(slid.asym.com(), [82.59726537, 52.63939034, 90.46451613, 1.])

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

   slid = asuslide(sym, xyz, frames, showme=False, maxstep=20, step=10, iters=3, clashiters=0, clashdis=8,
                   contactdis=10, contactfrac=0.1, vizsphereradius=6, cellsize=csize, towardaxis=True, alongaxis=False,
                   vizfresh=False, centerasu=False)
   # wu.showme(slid)
   ic(slid.cellsize, slid.asym.com())
   assert np.allclose(slid.cellsize, 97.5)
   assert np.allclose(slid.asym.com(), [2.89453125e+01, -1.21875000e+00, 3.27446403e-16, 1.00000000e+00])

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
   assert np.allclose(slid.cellsize, 183.75)
   assert np.allclose(slid.asym.com(), [0.26694229, 19.87146628, 36.37601256, 1.])

   slid2 = asuslide(sym, xyz, showme=False, maxstep=50, step=10, iters=3, clashiters=0, clashdis=8, contactdis=16,
                    contactfrac=0.2, vizsphereradius=2, cellsize=csize, towardaxis=True, alongaxis=False,
                    vizfresh=False, centerasu=False, xtalrad=0.5)
   # wu.showme(slid2, vizsphereradius=6)
   ic(slid.cellsize)
   ic(slid.asym.com())
   # ic(wu.hcart3(slid.asym.globalposition))
   assert np.allclose(slid.cellsize, 183.75)
   assert np.allclose(slid.asym.com(), [0.26694229, 19.87146628, 36.37601256, 1.])
   # assert 0

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
                   vizfresh=False, centerasu=False, iterstepscale=0.75)
   # wu.showme(slid)
   # slid.dump_pdb('test1.pdb')
   # ic(slid.bvh_op_count, len(slid.bodies))
   ic(slid.cellsize, slid.asym.com())
   assert np.allclose(slid.cellsize, 161.5703125)
   assert np.allclose(slid.asym.com(), [81.45648685, 41.24336469, 62.20570401, 1.])

   frames = xtal.frames(cells=(-1, 1), cellsize=csize, xtalrad=0.9)

   slid = asuslide(showme=0, sym=sym, coords=xyz, frames=frames, maxstep=30, step=7, iters=5, subiters=3, contactdis=16,
                   contactfrac=0.1, vizsphereradius=6, cellsize=csize, towardaxis=True, alongaxis=False, vizfresh=False,
                   centerasu=False, iterstepscale=0.75)
   # slid.dump_pdb('test2.pdb')
   # ic(slid.bvh_op_count, len(slid.bodies))
   ic(slid.cellsize, slid.asym.com())
   assert np.allclose(slid.cellsize, 161.5703125)
   assert np.allclose(slid.asym.com(), [81.45648685, 41.24336469, 62.20570401, 1.])

   slid = asuslide(showme=0, sym=sym, coords=xyz, maxstep=30, step=7, iters=5, subiters=3, contactdis=16,
                   contactfrac=0.1, vizsphereradius=6, cellsize=csize, towardaxis=True, alongaxis=False, vizfresh=False,
                   centerasu=False, iterstepscale=0.75)
   # wu.showme(slid)
   # slid.dump_pdb('test3.pdb')
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
   csize = 200
   np.random.seed(7)
   xyz = wu.tests.point_cloud(100, std=30, outliers=0)
   asucen = xtal.asucen(method='stored', cellsize=csize)
   xyz += wu.hvec(asucen)
   primary_frames = xtal.primary_frames(cellsize=csize)
   cen = wu.hcom(xyz)
   frames = wu.sym.frames(sym, ontop=primary_frames, cells=(-1, 1), cellsize=csize, center=cen, asucen=asucen,
                          xtalrad=0.9, strict=False)

   # rbprimary = wu.RigidBodyFollowers(coords=xyz, frames=primary_frames)
   # wu.showme(rbprimary)
   # rbstart = wu.RigidBodyFollowers(coords=xyz, frames=frames)
   # wu.showme(rbstart)
   # frames = primary_frames

   slid = asuslide(sym, xyz, frames, showme=False, maxstep=10, step=10.123, iters=3, clashiters=0, clashdis=8,
                   contactdis=16, contactfrac=0.2, vizsphereradius=6, cellsize=csize, towardaxis=True, alongaxis=False,
                   vizfresh=False, centerasu=False, iterstepscale=0.5, resetonfail=True)
   # wu.showme(slid)
   ic(slid.cellsize, slid.asym.com())
   assert np.allclose(slid.cellsize, 146.85425)
   assert np.allclose(slid.asym.com(), [18.666027, 37.33205399, 55.99808099, 1.])

   asucen = xtal.asucen(method='stored', cellsize=csize)
   csize = 100
   slid = asuslide(sym, xyz, showme=False, maxstep=10, step=10.123, iters=3, clashiters=0, clashdis=8, contactdis=16,
                   contactfrac=0.2, vizsphereradius=6, cellsize=csize, towardaxis=True, alongaxis=False, vizfresh=False,
                   centerasu=False, iterstepscale=0.5, resetonfail=True, xtalrad=0.6)
   # wu.showme(slid)
   ic(slid.cellsize, slid.asym.com())
   assert np.allclose(slid.cellsize, 145.5535)
   assert np.allclose(slid.asym.com(), [18.74043474, 37.48086949, 56.22130423, 1.])

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
   ic(slid.cellsize)
   ic(slid.asym.com())
   assert np.allclose(slid.cellsize, 147.10025)
   assert np.allclose(slid.asym.com(), [18.69073977, 37.38147955, 56.07221932, 1.])

def test_asuslide_F432():
   sym = 'F_4_3_2'
   xtal = wu.sym.Xtal(sym)
   csize = 150
   np.random.seed(7)
   xyz = wu.tests.point_cloud(100, std=30, outliers=0)
   asucen = xtal.asucen(method='stored', cellsize=csize)
   # ic(asucen / csize)
   # assert 0
   xyz += wu.hvec(asucen)

   # frames = wu.sym.frames(sym, cells=None, cellsize=csize)
   frames = wu.sym.frames(sym, cellsize=csize, cen=wu.hcom(xyz), xtalrad=0.5, strict=False)
   # ic(frames.shape)
   # assert 0

   slid = asuslide(sym, xyz, frames, showme=0, maxstep=30, step=10.3, iters=3, subiters=2, contactdis=10,
                   contactfrac=0.1, vizsphereradius=6, cellsize=csize, towardaxis=True, alongaxis=False, vizfresh=False,
                   centerasu=False, along_extra_axes=[[0, 0, 1]], iterstepscale=0.7)
   wu.showme(slid, vizsphereradius=6)
   ic(slid.cellsize, slid.asym.com())
   assert np.allclose(slid.cellsize, 201.5)
   assert np.allclose(slid.asym.com(), [154.9535, 15.5155, 77.5775, 1.])

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
   # xyz += wu.hvec(xtal.asucen(method='stored', cellsize=csize))
   xyz += wu.hvec(xtal.asucen(method='closest', cellsize=csize))

   # p = xtal.primary_frames(cellsize=csize)
   # ic(p.shape)
   # ic(wu.hcart3(p))
   # assert 0

   # wu.showme(wu.rigid.RigidBodyFollowers(sym=sym, coords=xyz, cellsize=csize, xtalrad=0.7, strict=False))

   slid = asuslide(sym, xyz, showme=0, maxstep=30, step=10, iters=2, subiters=1, clashiters=0, clashdis=8,
                   contactdis=12, contactfrac=0.2, vizsphereradius=6, cellsize=csize, towardaxis=True, alongaxis=False,
                   vizfresh=False, centerasu=False, along_extra_axes=[], xtalrad=0.5, iterstepscale=0.5)

   # ic(slid.bvh_op_count)
   # wu.showme(slid)
   ic(slid.cellsize, slid.asym.com())
   assert np.allclose(slid.cellsize, 165)
   assert np.allclose(slid.asym.com(), [48.91764706, 29.7, 13.97647059, 1.])

   # cen = asucen
   # cen = wu.hcom(xyz + [0, 0, 0, 0])
   # frames = wu.sym.frames(sym, ontop=primary_frames, cells=(-1, 1), cellsize=csize, center=asucen, asucen=cen,
   # xtalrad=csize * 0.7)
   # ic(len(frames))

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
