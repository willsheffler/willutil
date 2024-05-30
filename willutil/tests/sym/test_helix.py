import numpy as np
import willutil as wu
import pytest

def main():

   test_helix_scaling()
   assert 0

   h = wu.sym.Helix(turns=11, nfold=1, turnsB=1, phase=0.5)
   wu.showme(h, radius=3.8, spacing=1.3, coils=4)
   assert 0

   test_helix_9_1_1_r100_s40_p50_t2_d80_c7()
   test_helix_7_1_1_r80_s30_p20_t1_c7()
   test_helix_scaling()
   test_helix_params()
   test_helix_upper_neighbors()

def test_helix_upper_neighbors():
   h = wu.sym.Helix(turns=9, nfold=1, turnsB=1, phase=0)
   wu.showme(h)
   wu.showme(h, closest=9)
   wu.showme(h, closest=5, closest_upper_only=True)

def test_helix_params():
   h = wu.sym.Helix(turns=9, nfold=1, turnsB=1, phase=0.001)
   h = wu.sym.Helix(turns=9, nfold=1, turnsB=1, phase=0.999)
   with pytest.raises(ValueError):
      h = wu.sym.Helix(turns=9, nfold=1, turnsB=1, phase=-0.001)
   with pytest.raises(ValueError):
      h = wu.sym.Helix(turns=9, nfold=1, turnsB=1, phase=1.001)

def test_helix_scaling():
   pytest.importorskip('willutil_cpp')
   h = wu.sym.Helix(turns=9, nfold=1, turnsB=1, phase=0)

   np.random.seed(7)
   xyz = wu.tests.point_cloud(100, std=30, outliers=0)

   hframes = h.frames(xtalrad=9e8, closest=0, radius=100, spacing=40, coils=4)
   rb = wu.RigidBodyFollowers(coords=xyz, frames=hframes, symtype='H')
   origorig = rb.origins()
   origori = rb.orientations()

   scale = [1, 1, 2]
   rb.scale = scale
   assert np.allclose(rb.origins(), origorig * scale)
   assert np.allclose(rb.orientations(), origori)

   scale = [1.4, 1.4, 1]
   rb.scale = scale
   assert np.allclose(rb.origins(), origorig * scale)
   assert np.allclose(rb.orientations(), origori)

   scale = [1.4, 1.4, 0.5]
   rb.scale = scale
   assert np.allclose(rb.origins(), origorig * scale)
   assert np.allclose(rb.orientations(), origori)

   # for i in range(10):
   #    # rb.scale = [1, 1, 1 + i / 10]
   #    rb.scale = [1 + i / 10, 1 + i / 10, 1 - i / 20]
   #    wu.showme(rb)
   #    ic(rb.origins())

   h = wu.sym.Helix(turns=9, nfold=1, turnsB=1, phase=0.5)
   hframes = h.frames(xtalrad=9e8, closest=0, radius=90, spacing=40, coils=4)
   rb = wu.RigidBodyFollowers(coords=xyz, frames=hframes, symtype='H')
   assert not np.allclose(rb.orientations(), origori)

def test_helix_9_1_1_r100_s40_p50_t2_d80_c7():
   h = wu.sym.Helix(turns=9, nfold=1, turnsB=1, phase=0.5)
   hframes = h.frames(xtalrad=80, closest=7, radius=100, spacing=40, coils=2)
   foo = np.array([[[5.46948158e-01, -8.37166478e-01, 0.00000000e+00, 5.46948158e+01],
                    [8.37166478e-01, 5.46948158e-01, 0.00000000e+00, 8.37166478e+01],
                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, -3.55555556e+01],
                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
                   [[5.46948158e-01, 8.37166478e-01, 0.00000000e+00, 5.46948158e+01],
                    [-8.37166478e-01, 5.46948158e-01, 0.00000000e+00, -8.37166478e+01],
                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 3.55555556e+01],
                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
                   [[2.45485487e-01, -9.69400266e-01, 0.00000000e+00, 2.45485487e+01],
                    [9.69400266e-01, 2.45485487e-01, 0.00000000e+00, 9.69400266e+01],
                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, -7.55555556e+01],
                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
                   [[-8.25793455e-02, 9.96584493e-01, 0.00000000e+00, -8.25793455e+00],
                    [-9.96584493e-01, -8.25793455e-02, 0.00000000e+00, -9.96584493e+01],
                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 3.11111111e+01],
                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
                   [[-8.25793455e-02, -9.96584493e-01, 0.00000000e+00, -8.25793455e+00],
                    [9.96584493e-01, -8.25793455e-02, 0.00000000e+00, 9.96584493e+01],
                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, -3.11111111e+01],
                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
                   [[-6.77281572e-01, 7.35723911e-01, 0.00000000e+00, -6.77281572e+01],
                    [-7.35723911e-01, -6.77281572e-01, 0.00000000e+00, -7.35723911e+01],
                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 2.66666667e+01],
                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
                   [[-6.77281572e-01, -7.35723911e-01, 0.00000000e+00, -6.77281572e+01],
                    [7.35723911e-01, -6.77281572e-01, 0.00000000e+00, 7.35723911e+01],
                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 5.77777778e+01],
                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]])
   assert np.allclose(foo, hframes)

def test_helix_7_1_1_r80_s30_p20_t1_c7():
   h = wu.sym.Helix(turns=9, nfold=1, turnsB=1, phase=0.2)
   hframes = h.frames(closest=9, radius=80, spacing=30, coils=1)
   foo = np.array([[[1., 0., 0., 80.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]],
                   [[0.99068595, -0.13616665, 0., 79.25487568], [0.13616665, 0.99068595, 0., 10.89333193],
                    [0., 0., 1., -30.], [0., 0., 0., 1.]],
                   [[0.99068595, 0.13616665, 0., 79.25487568], [-0.13616665, 0.99068595, 0., -10.89333193],
                    [0., 0., 1., 30.], [0., 0., 0., 1.]],
                   [[0.77571129, -0.63108794, 0., 62.05690326], [0.63108794, 0.77571129, 0., 50.48703555],
                    [0., 0., 1., 3.33333333], [0., 0., 0., 1.]],
                   [[0.77571129, 0.63108794, 0., 62.05690326], [-0.63108794, 0.77571129, 0., -50.48703555],
                    [0., 0., 1., -3.33333333], [0., 0., 0., 1.]],
                   [[0.8544194, 0.51958395, 0., 68.35355236], [-0.51958395, 0.8544194, 0., -41.566716],
                    [0., 0., 1., -33.33333333], [0., 0., 0., 1.]],
                   [[0.8544194, -0.51958395, 0., 68.35355236], [0.51958395, 0.8544194, 0., 41.566716],
                    [0., 0., 1., 33.33333333], [0., 0., 0., 1.]],
                   [[0.68255314, -0.73083596, 0., 54.60425146], [0.73083596, 0.68255314, 0., 58.46687714],
                    [0., 0., 1., -26.66666667], [0., 0., 0., 1.]],
                   [[0.68255314, 0.73083596, 0., 54.60425146], [-0.73083596, 0.68255314, 0., -58.46687714],
                    [0., 0., 1., 26.66666667], [0., 0., 0., 1.]]])
   assert np.allclose(foo, hframes)

if __name__ == '__main__':
   main()
