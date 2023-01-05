import itertools, tempfile, sys
import numpy as np
import pytest
import willutil as wu
# ic.configureOutput(includeContext=True, contextAbsPath=False)

def main():

   test_dump_pdb()
   assert 0

   if 0:
      test_hxtal_viz(
         spacegroup='I4132_322',
         headless=False,
         cells=2,
         symelemscale=0.3,
         fansize=np.array([1.7, 1.2, 0.7]) / 3,
         fancover=10,
         symelemtwosided=True,
         showsymelems=True,
         pointshift=(0.2, 0.2, 0.1),
         scaleptrad=1,
      )
      '''
run /home/sheffler/pymol3/misc/G222.py; gyroid(10,r=11,cen=Vec(5,5,5)); set light, [ -0.3, -0.30, 0.8 ]
   '''
      assert 0, 'aoisrtnoiarnsiot'

   test_asucen()
   noshow = True
   test_xtal_L6m322(headless=noshow)
   test_xtal_L6_32(headless=noshow)
   # assert 0, 'stilltest viz'

   test_hxtal_viz(spacegroup='I 21 3', headless=noshow)
   test_hxtal_viz(spacegroup='P 2 3', headless=noshow)
   test_hxtal_viz(spacegroup='P 21 3', headless=noshow)
   test_hxtal_viz(spacegroup='I 41 3 2', headless=noshow)
   test_hxtal_viz(spacegroup='I4132_322', headless=noshow)
   if not noshow: assert 0

   test_xtal_cryst1_I_21_3(dump_pdbs=False)
   test_xtal_cryst1_P_2_3(dump_pdbs=False)
   test_xtal_cryst1_P_21_3(dump_pdbs=False)
   test_xtal_cryst1_I_41_3_2(dump_pdbs=False)
   test_xtal_cryst1_I4132_322(dump_pdbs=False)
   test_xtal_cellframes()
   test_symelem()

   # _test_hxtal_viz_gyroid(headless=False)
   ic('test_xtal.py DONE')

def test_dump_pdb():
   sym = 'I213'
   xtal = wu.sym.Xtal(sym)
   csize = 150
   np.random.seed(7)
   xyz = wu.tests.point_cloud(100, std=30, outliers=0)
   asucen = xtal.asucen(use_olig_nbrs=True, cellsize=csize)
   xyz += wu.hvec(asucen)
   # xyz[:, 1] -= 2
   xtal.dump_pdb('test.pdb', xyz, cellsize=csize, cells=(-1, 0), radius=0.5, ontop='primary')

def test_asucen(headless=True):
   csize = 62.144
   xtal = wu.sym.Xtal('P 21 3')
   asucen = xtal.asucen(cellsize=csize, method='closest_approach')
   cellpts = xtal.symcoords(asucen, cellsize=csize, flat=True)
   frames = xtal.primary_frames(csize)
   wu.showme(xtal, scale=csize)
   wu.showme(asucen, sphere=4)
   wu.showme(cellpts, sphere=4)

def test_xtal_L6m322(headless=True):
   xtal = wu.sym.Xtal('L6m322')

   ic(xtal.symelems)
   ic(xtal.genframes.shape)
   # ic(len(xtal.coverelems))
   # ic(len(xtal.coverelems[0]))
   # ic(len(xtal.coverelems[1]))
   wu.showme(xtal.genframes, scale=3, headless=headless)
   # wu.showme(xtal.unitframes, name='arstn', scale=3)
   wu.showme(xtal, headless=headless, showgenframes=False, symelemscale=1, pointscale=0.8, fresh=True)

def test_xtal_L6_32(headless=False):
   xtal = wu.sym.Xtal('L6_32')
   # ic(xtal.symelems)fresh
   # ic(xtal.genframes.shape)
   # ic(len(xtal.coverelems))
   # ic(len(xtal.coverelems[0]))
   # ic(len(xtal.coverelems[1]))
   wu.showme(xtal.genframes, scale=3, headless=headless)
   # wu.showme(xtal.unitframes, name='arstn')
   # wu.showme(xtal, headless=False, showgenframes=False, symelemscale=1, pointscale=0.8, fresh=True)

def test_symelem(headless=True):
   elem1 = wu.sym.SymElem(2, [1, 0, 0], [0, 0, 0])
   elem2 = wu.sym.SymElem(2, [1, 0, 0], [0, 10, 0])

   x = wu.hrand()
   e2 = wu.hxform(x, elem1)
   assert np.allclose(e2.coords, wu.hxform(x, elem1.coords))

   # x = wu.hrand()
   # e2 = wu.hxform(x, elem1)
   # assert np.allclose(e2.coords, wu.hxform(x, elem1.coords))
   # wu.showme(elem1, headless=headless)
   # wu.showme(wu.hxform(wu.hrot([0, 1, 0]s, 120, [0, 0, 1]), elem1), headless=headless)
   # wu.showme([elem1], fancover=0.8)

def test_xtal_cellframes():
   xtal = wu.sym.Xtal('P 2 3')
   assert xtal.nsub == 12
   assert len(xtal.cellframes(cellsize=1, cells=1)) == xtal.nsub
   assert len(xtal.cellframes(cellsize=1, cells=None)) == 1
   assert len(xtal.cellframes(cellsize=1, cells=2)) == 8 * xtal.nsub
   assert len(xtal.cellframes(cellsize=1, cells=3)) == 27 * xtal.nsub
   assert len(xtal.cellframes(cellsize=1, cells=4)) == 64 * xtal.nsub
   assert len(xtal.cellframes(cellsize=1, cells=5)) == 125 * xtal.nsub

def test_xtal_cryst1_P_2_3(*args, **kw):
   helper_test_xtal_cryst1('P 2 3', *args, **kw)

def test_xtal_cryst1_I4132_322(*args, **kw):
   helper_test_xtal_cryst1('I4132_322', *args, **kw)

def test_xtal_cryst1_I_41_3_2(*args, **kw):
   helper_test_xtal_cryst1('I 41 3 2 ', *args, **kw)

def test_xtal_cryst1_I_21_3(*args, **kw):
   helper_test_xtal_cryst1('I 21 3', *args, **kw)

def test_xtal_cryst1_P_21_3(*args, **kw):
   helper_test_xtal_cryst1('P 21 3', *args, **kw)

def prune_bbox(coords, lb, ub):
   inboundslow = np.all(coords >= lb - 0.001, axis=-1)
   inboundshigh = np.all(coords <= ub + 0.001, axis=-1)
   inbounds = np.logical_and(inboundslow, inboundshigh)
   return inbounds

def helper_test_xtal_cryst1(spacegroup, dump_pdbs=False):
   pymol = pytest.importorskip('pymol')
   xtal = wu.sym.Xtal(spacegroup)

   cellsize = 99.12345
   crd = cellsize * np.array([
      [0.28, 0.13, 0.13],
      [0.28, 0.16, 0.13],
      [0.28, 0.13, 0.15],
   ])

   if dump_pdbs:
      xtal.dump_pdb(f'test_{spacegroup.replace(" ","_")}_1.pdb', crd, cellsize=cellsize, cells=1)
      xtal.dump_pdb(f'test_{spacegroup.replace(" ","_")}_2.pdb', crd, cellsize=cellsize, cells=None)

   with tempfile.TemporaryDirectory() as tmpdir:
      pymol.cmd.delete('all')
      fname = f'{tmpdir}/test.pdb'
      xtal.dump_pdb(fname, crd, cellsize=cellsize)
      pymol.cmd.load(fname)
      pymol.cmd.symexp('pref', 'test', 'all', 9e9)
      coords1 = pymol.cmd.get_coords()
      pymol.cmd.delete('all')
      coords2 = xtal.symcoords(crd, cellsize=cellsize, cells=(-2, 1), flat=True)
      assert len(coords1) == 27 * 3 * xtal.nsub
      assert len(coords2) == 64 * 3 * xtal.nsub

   if True:
      coords1 = coords1.round().astype('i')
      coords2 = coords2.round().astype('i')[..., :3]
      s1 = set([tuple(x) for x in coords1])
      s2 = set([tuple(x) for x in coords2])
      # ic(spacegroup, len(s1), len(coords1))
      # ic(spacegroup, len(s2), len(coords2))
      # ic(spacegroup, len(s2 - s1), (64 - 27) * 3 * xtal.nsub)
      expected_ratio = (4**3 - 3**3) / 3**3
      assert len(s2 - s1) == (64 - 27) * 3 * xtal.nsub
      assert len(s1 - s2) == 0, f'canonical frames mismatch {spacegroup}'
      assert len(s1.intersection(s2)) == len(coords1), f'canonical frames mismatch {spacegroup}'

   lb = -105
   ub = 155
   coords1 = coords1[coords1[:, 0] < ub]
   coords1 = coords1[coords1[:, 0] > lb]
   coords1 = coords1[coords1[:, 1] < ub]
   coords1 = coords1[coords1[:, 1] > lb]
   coords1 = coords1[coords1[:, 2] < ub]
   coords1 = coords1[coords1[:, 2] > lb]
   coords2 = coords2[coords2[:, 0] < ub]
   coords2 = coords2[coords2[:, 0] > lb]
   coords2 = coords2[coords2[:, 1] < ub]
   coords2 = coords2[coords2[:, 1] > lb]
   coords2 = coords2[coords2[:, 2] < ub]
   coords2 = coords2[coords2[:, 2] > lb]

   if dump_pdbs:
      wu.pdb.dump_pdb_from_points(f'test_{spacegroup.replace(" ","_")}_pymol.pdb', coords1)
      wu.pdb.dump_pdb_from_points(f'test_{spacegroup.replace(" ","_")}_wxtal.pdb', coords2)

   # ic(spacegroup)
   # ic(coords1.shape)
   # ic(coords2.shape)
   coords1 = coords1.round().astype('i')
   coords2 = coords2.round().astype('i')[..., :3]
   s1 = set([tuple(x) for x in coords1])
   s2 = set([tuple(x) for x in coords2])
   assert s1 == s2

def test_hxtal_viz(headless=True, spacegroup='P 2 3', symelemscale=0.7, cellsize=10, **kw):
   pymol = pytest.importorskip('pymol')
   xtal = wu.sym.Xtal(spacegroup)
   # ic(xtal.unitframes.shape)
   cen = xtal.asucen(cellsize=cellsize, method='closest_to_cen')

   # wu.showme(xtal.symelems, scale=cellsize)
   # wu.showme(cen, sphereradius=1)

   wu.showme(
      xtal,
      headless=headless,
      showgenframes=False,
      symelemscale=symelemscale,
      pointscale=0.8,
      scale=cellsize,
      showpoints=cen[None],
      pointradius=0.3,
      # fresh=True,
      **kw,
   )
   # sys.path.append('/home/sheffler/src/wills_pymol_crap')
   # pymol.cmd.do('@/home/sheffler/.pymolrc')
   # pymol.cmd.do('run /home/sheffler/pymol3/misc/G222.py; gyroid(10,r=8,c=Vec(5,5,5)); set light, [ -0.3, -0.30, 0.8 ]')
   # elem1 = wu.sym.SymElem(2, [1, 0, 0], [0, 0.25, 0.0])
   # elem2 = wu.sym.SymElem(3, [1, 1, -1], [0, 0, 0])
   # xtal = wu.sym.Xtal([elem1, elem2])
   # for a, b, c in itertools.product(*[(0, 1)] * 3):
   #    # wu.showme(xtal, cellshift=[a, b, c], showgenframes=a == b == c == 0)
   #    wu.showme(xtal, cellshift=[a, b, c], headless=headless)

def _test_hxtal_viz_gyroid(headless=True):
   # elem1 = wu.sym.SymElem(2, [1, 0, 0], [0, 0.25, 0.0])
   # elem2 = wu.sym.SymElem(3, [1, 1, 1], [0, 0, 0])
   # xtal = wu.sym.Xtal([elem1, elem2])

   # xtal = wu.sym.Xtal('I 21 3')

   # wu.showme(xtal, headless=headless, fanshift=[-0.03, 0.05], fansize=[0.15, 0.12])
   # wu.showme(xtal, headless=headless, fanshift=[-0.03, 0.05], fansize=[0.15, 0.12])
   wu.showme(xtal, headless=headless, showpoints=1)
   #for a, b, c in itertools.product(*[(0, 1)] * 3):
   #   # wu.showme(xtal, cellshift=[a, b, c], showgenframes=a == b == c == 0)
   #   wu.showme(xtal, cellshift=[a, b, c], headless=headless, fanshift=[-0.03, 0.05],
   #             fansize=[0.15, 0.12])

if __name__ == '__main__':
   main()