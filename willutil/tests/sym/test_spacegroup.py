import pytest
import numpy as np
import willutil as wu

def main():
   # test_subsym()

   # assert 0

   test_spacegroup_frames_tounitcell('P1', [15, 25, 35, 75, 85, 95], 4)
   test_spacegroup_frames_tounitcell('P-1', [15, 25, 35, 75, 85, 95], 4)
   test_spacegroup_frames_tounitcell('P121', [15, 25, 35, 90, 85, 90], 4)
   test_spacegroup_frames_tounitcell('I213', [10], 2)
   test_spacegroup_frames_tounitcell('P212121', [10, 11, 12], 4)
   test_spacegroup_frames_tounitcell('P43', [10, 10, 12], 4)
   test_spacegroup_frames_tounitcell('P3', [11, 11, 13, 90, 90, 120], 2)

   test_spacegroup_frames_P1()
   test_lattice_vectors()
   test_two_iface()
   test_spacegroup_frames_P213()
   test_spacegroup_frames_P23()
   test_spacegroup_frames_I213()
   test_spacegroup_frames_P4132()
   test_spacegroup_frames_I4132()
   test_spacegroup_frames_P432()
   test_spacegroup_frames_F432()
   test_spacegroup_frames_I432()
   ic('PASS test_spacegroup')

def helper_test_spacegroup_frames(sg):
   f1 = wu.sym.sgframes(sg, [1, 1, 1])
   f2 = wu.sym.frames(sg, cells=1, ontop=None)
   assert f1.shape == f2.shape
   f2[np.where(np.abs(f2) < 0.0001)] = 0
   f2[f2[:, 0, 3] > 0.999, 0, 3] = 0  # mod1
   f2[f2[:, 1, 3] > 0.999, 1, 3] = 0  # mod1
   f2[f2[:, 2, 3] > 0.999, 2, 3] = 0  # mod1
   for i, x in enumerate(f1):
      match = None
      for j, y in enumerate(f2):
         if np.allclose(x, y, atol=1e-4):
            assert match is None
            match = j
      assert match is not None
   # print(f'pass {sg}')

def test_lattice_vectors():
   print('''
      15.737   15.887   25.156  85.93  86.19  69.86 P 1     
      15.737000,   0.000000,   0.000000
       5.470136,  14.915575,   0.000000
       1.671566,   1.288705,  25.067297
   ''', flush=True)
   lvec = wu.sym.spacegroup.lattice_vectors('P1', [15.737, 15.887, 25.156, 85.93, 86.19, 69.86])
   assert np.allclose(lvec, np.array([[15.737, 0., 0.], [5.47013594, 14.91557514, 0.], [1.67156711, 1.28870451, 25.06729822]]))

def test_spacegroup_frames_P1():
   f = wu.sym.spacegroup.sgframes('P1', [15.737, 15.887, 25.156, 85.93, 86.19, 69.86], cells=3)
   ref = np.array([[-22.87870305, -16.20427966, -25.06729822], [-21.20713594, -14.91557514, 0.], [-19.53556883, -13.62687063, 25.06729822], [-17.40856711, -1.28870451, -25.06729822], [-15.737, 0., 0.], [-14.06543289, 1.28870451, 25.06729822], [-11.93843117, 13.62687063, -25.06729822], [-10.26686406, 14.91557514, 0.], [-8.59529695, 16.20427966, 25.06729822], [-7.14170305, -16.20427966, -25.06729822], [-5.47013594, -14.91557514, 0.], [-3.79856883, -13.62687063, 25.06729822], [-1.67156711, -1.28870451, -25.06729822], [0., 0., 0.], [1.67156711, 1.28870451, 25.06729822], [3.79856883, 13.62687063, -25.06729822], [5.47013594, 14.91557514, 0.], [7.14170305, 16.20427966, 25.06729822], [8.59529695, -16.20427966, -25.06729822], [10.26686406, -14.91557514, 0.], [11.93843117, -13.62687063, 25.06729822], [14.06543289, -1.28870451, -25.06729822], [15.737, 0., 0.], [17.40856711, 1.28870451, 25.06729822], [19.53556883, 13.62687063, -25.06729822], [21.20713594, 14.91557514, 0.], [22.87870305, 16.20427966, 25.06729822]])
   assert np.allclose(f[:, :3, 3], ref)
   # wu.showme(f)

@pytest.mark.parametrize('sgroup,cellgeom,ncell', [
   ('P1', [15, 25, 35, 75, 85, 95], 4),
   ('P-1', [15, 25, 35, 75, 85, 95], 4),
   ('P121', [15, 25, 35, 90, 85, 90], 4),
   ('I213', [10], 2),
   ('P212121', [10, 11, 12], 4),
   ('P43', [10, 10, 12], 4),
   ('P3', [11, 11, 13, 90, 90, 120], [1, 2, 1]),
])
def test_spacegroup_frames_tounitcell(sgroup, cellgeom, ncell):
   fcell = wu.sym.spacegroup.sgframes(sgroup, cellgeom, cells=ncell)
   # ic(fcell.shape)
   funit = wu.sym.spacegroup.sgframes(sgroup, cellgeom='unit', cells=ncell)
   # ic(funit.shape)
   ftest = wu.sym.spacegroup.tounitframes(fcell, cellgeom, sgroup)
   # ic(ftest.shape)
   # ic(funit)
   # ic(ftest)
   assert np.allclose(funit, ftest)

def test_two_iface():
   assert len(wu.sym.spacegroup_data.two_iface_spacegroups) == 31
   for sg in wu.sym.spacegroup_data.two_iface_spacegroups:
      if sg not in wu.sym.spacegroup_data.sg_lattice:
         assert sg == 'R3'

def test_spacegroup_frames_I213():
   helper_test_spacegroup_frames('I213')

def test_spacegroup_frames_P213():
   helper_test_spacegroup_frames('P213')

def test_spacegroup_frames_P4132():
   helper_test_spacegroup_frames('P4132')

def test_spacegroup_frames_I4132():
   helper_test_spacegroup_frames('I4132')

def test_spacegroup_frames_P432():
   helper_test_spacegroup_frames('P432')

def test_spacegroup_frames_F432():
   helper_test_spacegroup_frames('F432')

def test_spacegroup_frames_I432():
   helper_test_spacegroup_frames('I432')

def test_spacegroup_frames_P23():
   helper_test_spacegroup_frames('P23')

def ___test_spacegroup_frames_I4132():
   f1 = wu.sym.sgframes('I4132', [1, 1, 1])
   f2 = wu.sym.frames('I4132_322', cells=1, ontop=None)
   f2[np.where(np.abs(f2) < 0.0001)] = 0
   f2[f2[:, 0, 3] > 0.999, 0, 3] = 0
   f2[f2[:, 1, 3] > 0.999, 1, 3] = 0
   f2[f2[:, 2, 3] > 0.999, 2, 3] = 0

   # for x in f2:
   # print(list(x[:3, :3].T.reshape(9).round(6).astype('i')) + list(x[:3, 3].round(6)))
   # assert 0

   # f1[:, :3, 3] *= 10
   # f2[:, :3, 3] *= 10
   # wu.showme(f1)
   # wu.showme(f2)
   # assert 0
   for i, x in enumerate(f1[:24]):
      match = None
      for j, y in enumerate(f2):
         # if i == 25 and np.isclose(y[0, 1], -1):
         # if i == 25 and np.allclose(x[:3, :3], y[:3, :3]):
         # ic(i, j, x, y)
         if np.allclose(x, y, atol=1e-4):
            assert match is None
            match = j
      # if match is None:
      # ic(i)
      # for j, y in enumerate(f2):
      # if np.allclose(x[:3, :3], y[:3, :3]):
      # ic(x, y)
      assert match is not None
   # ic('pass')

if __name__ == '__main__':
   main()
