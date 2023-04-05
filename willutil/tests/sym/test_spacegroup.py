import numpy as np
import willutil as wu

def main():
   test_two_iface()
   test_spacegroup_frames_P213()
   test_spacegroup_frames_P23()
   test_spacegroup_frames_I213()
   test_spacegroup_frames_P4132()
   test_spacegroup_frames_I4132()
   test_spacegroup_frames_P432()
   test_spacegroup_frames_F432()
   test_spacegroup_frames_I432()

def helper_test_spacegroup_frames(sg):
   f1 = wu.sym.sgframes(sg, [1, 1, 1])
   f2 = wu.sym.frames(sg, cells=1, ontop=None)
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

def test_two_iface():
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
