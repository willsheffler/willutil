import pytest, numpy as np, itertools as it, functools as ft

import willutil as wu
from willutil.homog.thgeom import *
from willutil.homog.hgeom import *

# pytest.skip(allow_module_level=True)

def main():

   test_th_axis_angle_cen_hel_vec()

   test_th_vec()
   test_th_rog()
   # assert 0, 'DONE'
   test_torch_rmsfit()
   test_th_misc()
   test_axisangcenhel_roundtrip()
   test_th_intersect_planes()
   test_th_axis_angle_cen_hel()
   test_th_rot_single()
   test_th_rot_56789()
   test_th_axis_angle_hel()
   test_th_axis_angle()

   test_torch_grad()
   test_th_axis_angle_cen_rand()
   test_torch_rmsfit_grad()

   ic('test_thgeom.py DONE')

def test_th_vec():
   torch = pytest.importorskip('torch')
   v = thrandvec(10)
   # ic(v)
   v2 = thvec(v)
   assert v is v2
   p = thrandpoint(10)
   v3 = thvec(p)
   assert torch.allclose(p[..., :3], v3[..., :3])
   assert torch.allclose(v3[..., 3], torch.tensor(0.0))

   v4 = thvec(p[..., :3])
   assert torch.allclose(p[..., :3], v4[..., :3])
   assert torch.allclose(v4[..., 3], torch.tensor(0.0))

def test_th_rog():
   torch = pytest.importorskip('torch')
   points = thrandpoint(10)
   rg = throg(points)
   rgx = throg(points, aboutaxis=[1, 0, 0])
   assert rg >= rgx
   points[..., 0] = 0
   rg = throg(points)
   assert np.allclose(rg, rgx)

def test_axisangcenhel_roundtrip():
   torch = pytest.importorskip('torch')

   axis0 = torch.tensor([1., 0, 0, 0], requires_grad=True)
   axis = thnormalized(axis0)
   ang = torch.tensor([2.0], requires_grad=True)
   cen = torch.tensor([1, 2, 3, 1.], requires_grad=True)
   hel = torch.tensor([1.], requires_grad=True)
   x = throt(axis, ang, cen, hel)
   axis2, ang2, cen2, hel2 = thaxis_angle_cen_hel(x)

   ang2.backward()
   assert np.allclose(axis0.grad.detach(), [0, 0, 0, 0])
   assert np.allclose(ang.grad.detach(), [1])
   assert np.allclose(cen.grad.detach(), [0, 0, 0, 0])
   assert np.allclose(hel.grad.detach(), [0])

   axis0 = torch.tensor([1., 0, 0, 0], requires_grad=True)
   axis = thnormalized(axis0)
   ang = torch.tensor([2.0], requires_grad=True)
   cen = torch.tensor([1, 2, 3, 1.], requires_grad=True)
   hel = torch.tensor([1.], requires_grad=True)
   x = throt(axis, ang, cen, hel)
   axis2, ang2, cen2, hel2 = thaxis_angle_cen_hel(x)
   axis2[0].backward()
   assert np.allclose(axis0.grad.detach(), [0, 0, 0, 0])
   assert np.allclose(ang.grad.detach(), [0])
   assert np.allclose(cen.grad.detach(), [0, 0, 0, 0])
   assert np.allclose(hel.grad.detach(), [0])

   axis0 = torch.tensor([1., 0, 0, 0], requires_grad=True)
   axis = thnormalized(axis0)
   ang = torch.tensor([2.0], requires_grad=True)
   cen = torch.tensor([1, 2, 3, 1.], requires_grad=True)
   hel = torch.tensor([1.], requires_grad=True)
   x = throt(axis, ang, cen, hel)
   axis2, ang2, cen2, hel2 = thaxis_angle_cen_hel(x)
   axis2[1].backward()
   assert np.allclose(axis0.grad.detach(), [0, 1, 0, 0])
   assert np.allclose(ang.grad.detach(), [0])
   assert np.allclose(cen.grad.detach(), [0, 0, 0, 0])
   assert np.allclose(hel.grad.detach(), [0])

   axis0 = torch.tensor([1., 0, 0, 0], requires_grad=True)
   axis = thnormalized(axis0)
   ang = torch.tensor([2.0], requires_grad=True)
   cen = torch.tensor([1, 2, 3, 1.], requires_grad=True)
   hel = torch.tensor([1.], requires_grad=True)
   x = throt(axis, ang, cen, hel)
   axis2, ang2, cen2, hel2 = thaxis_angle_cen_hel(x)
   axis2[2].backward()
   assert np.allclose(axis0.grad.detach(), [0, 0, 1, 0])
   assert np.allclose(ang.grad.detach(), [0])
   assert np.allclose(cen.grad.detach(), [0, 0, 0, 0])
   assert np.allclose(hel.grad.detach(), [0])

   axis0 = torch.tensor([1., 0, 0, 0], requires_grad=True)
   axis = thnormalized(axis0)
   ang = torch.tensor([2.0], requires_grad=True)
   cen = torch.tensor([1, 2, 3, 1.], requires_grad=True)
   hel = torch.tensor([1.], requires_grad=True)
   x = throt(axis, ang, cen, hel)
   axis2, ang2, cen2, hel2 = thaxis_angle_cen_hel(x)
   cen2[1].backward()
   # ic(axis0.grad)
   assert np.allclose(axis0.grad.detach(), [0, -1, 0, 0], atol=1e-4)
   assert np.allclose(ang.grad.detach(), [0], atol=1e-4)
   assert np.allclose(cen.grad.detach(), [0, 1, 0, 0], atol=1e-4)
   assert np.allclose(hel.grad.detach(), [0], atol=1e-4)

   axis0 = torch.tensor([1., 0, 0, 0], requires_grad=True)
   axis = thnormalized(axis0)
   ang = torch.tensor([2.0], requires_grad=True)
   cen = torch.tensor([1, 2, 3, 1.], requires_grad=True)
   hel = torch.tensor([1.], requires_grad=True)
   x = throt(axis, ang, cen, hel)
   axis2, ang2, cen2, hel2 = thaxis_angle_cen_hel(x)
   hel2.backward()
   assert np.allclose(axis0.grad.detach(), [0, 0, 0, 0], atol=1e-4)
   assert np.allclose(ang.grad.detach(), [0], atol=1e-4)
   assert np.allclose(cen.grad.detach(), [0, 0, 0, 0], atol=1e-4)
   assert np.allclose(hel.grad.detach(), [1], atol=1e-4)

   # assert 0

def test_th_axis_angle_cen_hel_vec():
   xforms = hrand(100)
   xgeom = wu.homog.axis_angle_cen_hel_of(xforms)
   for i, (x, a, an, c, h) in enumerate(zip(xforms, *xgeom)):
      a2, an2, c2, h2 = wu.homog.axis_angle_cen_hel_of(x)
      assert np.allclose(a, a2)
      assert np.allclose(an, an2)
      assert np.allclose(c, c2)
      assert np.allclose(h, h2)

def test_th_rot_56789():
   torch = pytest.importorskip('torch')
   torch.autograd.set_detect_anomaly(True)

   shape = (5, 6, 7, 8, 9)
   axis0 = wu.hnormalized(np.random.randn(*shape, 3))
   ang0 = np.random.random(shape) * (np.pi - 0.1) + 0.1
   cen0 = np.random.randn(*shape, 4) * 100.0
   cen0[..., 3] = 1.0
   axis0 = torch.tensor(axis0, requires_grad=True)
   ang0 = torch.tensor(ang0, requires_grad=True)
   cen0 = torch.tensor(cen0, requires_grad=True)
   hel = torch.randn(*shape)

   rot = t_rot(axis0, ang0, shape=(4, 4))
   rot2 = wu.homog.rot(axis0.detach(), ang0.detach(), shape=(4, 4))
   assert np.allclose(rot.detach(), rot2, atol=1e-5)

   rot = throt(axis0, ang0, cen0, hel=None)
   rot2 = wu.homog.hrot(axis0.detach(), ang0.detach(), cen0.detach())
   assert np.allclose(rot.detach(), rot2, atol=1e-5)

   s = torch.sum(rot)
   s.backward()
   assert axis0.grad is not None
   assert axis0.grad.shape == (5, 6, 7, 8, 9, 4)

def test_th_rot_single():
   torch = pytest.importorskip('torch')

   axis0 = wu.hnormalized(np.random.randn(3))
   ang0 = np.random.random() * (np.pi - 0.1) + 0.1
   cen0 = np.random.randn(4) * 100.0
   cen0[3] = 1.0
   axis0 = torch.tensor(axis0, requires_grad=True)
   ang0 = torch.tensor(ang0, requires_grad=True)
   cen0 = torch.tensor(cen0, requires_grad=True)

   rot = t_rot(axis0, ang0, shape=(4, 4))
   rot2 = wu.homog.rot(axis0.detach(), ang0.detach(), shape=(4, 4))
   assert np.allclose(rot.detach(), rot2, atol=1e-5)

   rot = throt(axis0, ang0, cen0, hel=None)
   rot2 = wu.homog.hrot(axis0.detach(), ang0.detach(), cen0.detach())
   assert np.allclose(rot.detach(), rot2, atol=1e-5)

   s = torch.sum(rot)
   s.backward()
   assert axis0.grad is not None

def test_th_axis_angle_cen_rand():
   torch = pytest.importorskip('torch')
   torch.autograd.set_detect_anomaly(True)

   shape = (5, 6, 7, 8, 9)
   if not torch.cuda.is_available():
      shape = shape[3:]

   axis0 = wu.hnormalized(np.random.randn(*shape, 3))
   ang0 = np.random.random(shape) * (np.pi - 0.1) + 0.1
   cen0 = np.random.randn(*shape, 4) * 100.0
   cen0[..., 3] = 1.0
   axis0 = torch.tensor(axis0, requires_grad=True)
   ang0 = torch.tensor(ang0, requires_grad=True)
   cen0 = torch.tensor(cen0, requires_grad=True)
   hel0 = torch.randn(*shape)

   # ic(axis0.shape)
   # ic(ang0.shape)
   # ic(cen0.shape)
   # ic(hel0.shape)

   # rot = t_rot(axis0, ang0, dim=4)
   rot = throt(axis0, ang0, cen0, hel0)
   # ic(rot.shape)
   # assert 0

   axis, ang, cen = axis_ang_cen_of(rot.detach().numpy())
   hel = hel0.detach().numpy()
   assert np.allclose(axis0.detach(), axis, rtol=1e-5)
   assert np.allclose(ang0.detach(), ang, rtol=1e-5)
   assert np.allclose(np.linalg.norm(axis, axis=-1), 1.0)
   cenhat = (rot.detach().numpy() @ cen[..., None]).squeeze()
   cenhel = cen + hel[..., None] * axis
   assert np.allclose(cenhel, cenhat, rtol=1e-4, atol=1e-4)

   # ic(rot.shape)
   axis2, ang2, cen2, hel2 = thaxis_angle_cen_hel(rot)
   assert np.allclose(axis2.detach(), axis)
   assert np.allclose(ang2.detach(), ang)
   assert np.allclose(cen2.detach(), cen)
   assert np.allclose(hel2.detach(), hel0)

def test_th_intersect_planes():
   torch = pytest.importorskip('torch')
   torch.autograd.set_detect_anomaly(True)

   p1 = torch.tensor([0., 0, 0, 1], requires_grad=True)
   n1 = torch.tensor([1., 0, 0, 0], requires_grad=True)
   p2 = torch.tensor([0., 0, 0, 1], requires_grad=True)
   n2 = torch.tensor([0., 1, 0, 0], requires_grad=True)
   isct, norm, status = thintersect_planes(p1, n1, p2, n2)
   assert status == 0
   assert isct[2] == 0
   assert np.allclose(abs(norm[:3].detach()), (0, 0, 1))
   isct[0].backward()
   assert np.allclose(p1.grad.detach(), [1, 0, 0, 0])
   assert np.allclose(n1.grad.detach(), [0, 0, 0, 0])
   assert np.allclose(p2.grad.detach(), [0, 0, 0, 0])
   assert np.allclose(n2.grad.detach(), [0, 0, 0, 0])

   p1 = torch.tensor([0., 0, 0, 1], requires_grad=True)
   n1 = torch.tensor([1., 0, 0, 0], requires_grad=True)
   p2 = torch.tensor([0., 0, 0, 1], requires_grad=True)
   n2 = torch.tensor([0., 0, 1, 0], requires_grad=True)
   isct, norm, status = thintersect_planes(p1, n1, p2, n2)
   assert status == 0
   assert isct[1] == 0
   assert np.allclose(abs(norm[:3].detach()), (0, 1, 0))

   p1 = torch.tensor([0., 0, 0, 1], requires_grad=True)
   n1 = torch.tensor([0., 1, 0, 0], requires_grad=True)
   p2 = torch.tensor([0., 0, 0, 1], requires_grad=True)
   n2 = torch.tensor([0., 0, 1, 0], requires_grad=True)
   isct, norm, status = thintersect_planes(p1, n1, p2, n2)
   assert status == 0
   assert isct[0] == 0
   assert np.allclose(abs(norm[:3].detach()), (1, 0, 0))

   p1 = torch.tensor([7., 0, 0, 1], requires_grad=True)
   n1 = torch.tensor([1., 0, 0, 0], requires_grad=True)
   p2 = torch.tensor([0., 9, 0, 1], requires_grad=True)
   n2 = torch.tensor([0., 1, 0, 0], requires_grad=True)
   isct, norm, status = thintersect_planes(p1, n1, p2, n2)
   assert status == 0
   assert np.allclose(isct[:3].detach(), [7, 9, 0])
   assert np.allclose(abs(norm.detach()), [0, 0, 1, 0])

   p1 = torch.tensor([0., 0, 0, 1], requires_grad=True)
   n1 = torch.tensor([1., 1, 0, 0], requires_grad=True)
   p2 = torch.tensor([0., 0, 0, 1], requires_grad=True)
   n2 = torch.tensor([0., 1, 1, 0], requires_grad=True)
   isct, norm, status = thintersect_planes(p1, n1, p2, n2)
   assert status == 0
   assert np.allclose(abs(norm.detach()), hnormalized([1, 1, 1, 0]))

   p1 = np.array([[0.39263901, 0.57934885, -0.7693232, 1.], [-0.80966465, -0.18557869, 0.55677976, 0.]]).T
   p2 = np.array([[0.14790894, -1.333329, 0.45396509, 1.], [-0.92436319, -0.0221499, 0.38087016, 0.]]).T
   isct2, sts = intersect_planes(p1, p2)
   isct2, norm2 = isct2.T
   p1 = torch.tensor([0.39263901, 0.57934885, -0.7693232, 1.])
   n1 = torch.tensor([-0.80966465, -0.18557869, 0.55677976, 0.])
   p2 = torch.tensor([0.14790894, -1.333329, 0.45396509, 1.])
   n2 = torch.tensor([-0.92436319, -0.0221499, 0.38087016, 0.])
   isct, norm, sts = thintersect_planes(p1, n1, p2, n2)
   assert sts == 0
   assert torch.all(thray_in_plane(p1, n1, isct, norm))
   assert torch.all(thray_in_plane(p2, n2, isct, norm))
   assert np.allclose(isct.detach(), isct2)
   assert np.allclose(norm.detach(), norm2)

   p1 = torch.tensor([2., 0, 0, 1], requires_grad=True)
   n1 = torch.tensor([1., 0, 0, 0], requires_grad=True)
   p2 = torch.tensor([0., 0, 0, 1], requires_grad=True)
   n2 = torch.tensor([0., 0, 1, 0], requires_grad=True)
   isct, norm, status = thintersect_planes(p1, n1, p2, n2)

   assert status == 0
   assert abs(thdot(n1, norm)) < 0.0001
   assert abs(thdot(n2, norm)) < 0.0001
   assert torch.all(thpoint_in_plane(p1, n1, isct))
   assert torch.all(thpoint_in_plane(p2, n2, isct))
   assert torch.all(thray_in_plane(p1, n1, isct, norm))
   assert torch.all(thray_in_plane(p2, n2, isct, norm))

def test_th_axis_angle_cen_hel():
   torch = pytest.importorskip('torch')
   torch.autograd.set_detect_anomaly(True)

   axis0 = torch.tensor(hnormalized(np.array([1., 1, 1, 0])), requires_grad=True)
   ang0 = torch.tensor(0.9398483, requires_grad=True)
   cen0 = torch.tensor([1., 2, 3, 1], requires_grad=True)
   h0 = torch.tensor(2.443, requires_grad=True)
   x = throt(axis0, ang0, cen0, h0)
   axis, ang, cen, hel = thaxis_angle_cen_hel(x)
   ax2, an2, cen2 = axis_ang_cen_of(x.detach().numpy())
   assert np.allclose(ax2, axis.detach())
   assert np.allclose(an2, ang.detach())

   assert np.allclose(cen2, cen.detach())
   hel.backward()
   assert np.allclose(ang0.grad, 0, atol=1e-5)
   hg = h0.detach().numpy() * np.sqrt(3) / 3
   assert np.allclose(axis0.grad, [hg, hg, hg, 0])

def test_torch_grad():
   torch = pytest.importorskip('torch')
   x = torch.tensor([2, 3, 4], dtype=torch.float, requires_grad=True)
   s = torch.sum(x)
   s.backward()
   assert np.allclose(x.grad.detach().numpy(), [1., 1., 1.])

def test_torch_quat():
   torch = pytest.importorskip('torch')
   torch.autograd.set_detect_anomaly(True)

   for v in (1., -1.):
      q0 = torch.tensor([v, 0., 0., 0.], requires_grad=True)
      q = thquat_to_upper_half(q0)
      assert np.allclose(q.detach(), [1, 0, 0, 0])
      s = torch.sum(q)
      s.backward()
      assert q0.is_leaf
      assert np.allclose(q0.grad.detach(), [0, v, v, v])

def test_torch_rmsfit(trials=10):
   torch = pytest.importorskip('torch')
   torch.autograd.set_detect_anomaly(True)

   for _ in range(trials):
      p = thrandpoint(10, std=10)
      q = thrandpoint(10, std=10)
      # ic(p)
      rms0 = thrms(p, q)
      rms, qhat, xpqhat = thrmsfit(p, q)
      assert rms0 > rms
      # ic(float(rms0), float(rms))
      assert np.allclose(thrms(qhat, q), rms)
      for i in range(10):
         qhat2 = thxform(thrand_xform_small(1, 0.01, 0.001), qhat)
         rms2 = thrms(q, qhat2)

         if rms2 < rms - 0.001:
            print(float(rms), float(rms2))
         assert rms2 >= rms - 0.001

# @pytest.mark.skip
def test_torch_rmsfit_grad():
   torch = pytest.importorskip('torch')
   if not torch.cuda.is_available():
      wu.tests.force_pytest_skip('CUDA not availble')
   # torch.autograd.set_detect_anomaly(True)
   # assert 0
   ntrials = 1
   npts = 50
   shift = 100
   nstep = 50
   for std in (0.01, 0.1, 1, 10, 100):
      for i in range(ntrials):
         xpq = rand_xform()
         points1 = hrandpoint(npts, std=10)
         points2 = hxform(xpq, points1) + rand_vec(npts) * std
         points2[:, 0] += shift
         points1[:, 3] = 1
         points2[:, 3] = 1
         # ic(points1)
         # ic(points2)
         assert points2.shape == (npts, 4)

         for i in range(nstep):
            p = torch.tensor(points1, requires_grad=True)
            q = torch.tensor(points2, requires_grad=True)
            p2, q2 = thpoint(p), thpoint(q)
            assert p2.shape == (npts, 4)

            rms, qhat, xpqhat = thrmsfit(p2, q2)
            rms2 = thrms(qhat, q)
            assert torch.allclose(qhat, thxformpts(xpqhat, p), atol=0.0001)
            # ic(rms, rms2)
            assert torch.allclose(rms, rms2, atol=0.0001)

            rms.backward()
            assert np.allclose(p.grad[:, 3].detach(), 0)
            assert np.allclose(q.grad[:, 3].detach(), 0)
            points1 = points1 - p.grad.detach().numpy() * 10 * float(rms)
            points2 = points2 - q.grad.detach().numpy() * 10 * float(rms)

            # if not i % 10:
            #     ic(std, i, float(rms))
            # ic(torch.norm(p.grad), torch.norm(q.grad))

         assert rms < 1e-3

def test_th_axis_angle():
   torch = pytest.importorskip('torch')
   torch.autograd.set_detect_anomaly(True)

   axis0 = torch.tensor([10., 10., 10., 0], requires_grad=True)
   ang0 = torch.tensor(0.4, requires_grad=True)
   x = throt(axis0, ang0)
   assert x.shape == (4, 4)
   assert x[1, 3] == 0.
   assert x[3, 1] == 0.
   assert x[3, 3] == 1.
   x[0, 0].backward()
   assert np.allclose(axis0.grad.detach(), [0.0035, -0.0018, -0.0018, 0], atol=0.002)
   assert np.allclose(ang0.grad.detach(), -0.2596, atol=0.001)

   axis0 = torch.tensor(hnormalized(np.array([1., 1, 1, 0])), requires_grad=True)
   ang0 = torch.tensor(0.8483, requires_grad=True)
   x = throt(axis0, ang0)
   ax, an, h = thaxis_angle_hel(x)
   assert np.allclose(an.detach(), ang0.detach())
   assert np.allclose(h.detach(), 0)
   ax2, an2, h2 = axis_angle_hel_of(x.detach())
   assert torch.allclose(torch.linalg.norm(ax, axis=-1), torch.ones_like(ax))
   assert np.allclose(ax2, ax.detach())
   assert np.allclose(an2, an.detach())
   assert np.allclose(h2, h.detach())
   an.backward()
   assert np.allclose(ang0.grad, 1)
   assert np.allclose(axis0.grad, [0, 0, 0, 0])

def test_th_axis_angle_hel():
   torch = pytest.importorskip('torch')
   torch.autograd.set_detect_anomaly(True)

   axis0 = torch.tensor(hnormalized(np.array([1., 1, 1, 0])), requires_grad=True)
   ang0 = torch.tensor(0.9398483, requires_grad=True)
   h0 = torch.tensor(2.443, requires_grad=True)
   x = throt(axis0, ang0, hel=h0)
   ax, an, h = thaxis_angle_hel(x)
   ax2, an2, h2 = axis_angle_hel_of(x.detach())
   assert np.allclose(ax2, ax.detach())
   assert np.allclose(an2, an.detach())
   assert np.allclose(h2, h.detach())
   h.backward()
   assert np.allclose(ang0.grad, 0)
   hg = h0.detach().numpy() * np.sqrt(3) / 3
   assert np.allclose(axis0.grad, [hg, hg, hg, 0])

def test_th_misc():
   torch = pytest.importorskip('torch')

   r = torch.randn(2, 4, 5, 3)
   p = thpoint(r)
   assert np.allclose(p[..., :3], r)
   assert np.allclose(p[..., 3], 1)
   p = thrandpoint(11)
   assert p.shape == (11, 4)
   assert np.allclose(p[:, 3], 1)

#################################################

if __name__ == '__main__':
   main()

# def test_th_axis_angle_cen_rand():
#    shape = (5, 6, 7, 8, 9)
#    axis0 = hnormalized(np.random.randn(*shape, 3))
#    ang0 = np.random.random(shape) * (np.pi - 0.1) + 0.1
#    cen0 = np.random.randn(*shape, 4) * 100.0
#    cen0[..., 3] = 1.0
#    axis0 = torch.tensor(axis0, requires_grad=True)
#    ang0 = torch.tensor(ang0, requires_grad=True)
#    cen0 = torch.tensor(cen0, requires_grad=True)
#    hel = torch.randn(*shape)

#    ic(axis0.shape)
#    ic(ang0.shape)
#    ic(cen0.shape)
#    ic(hel.shape)

#    rot = throt(axis0, ang0, cen0, hel)
#    ic(rot.shape)
#    assert 0

#    axis, ang, cen = axis_ang_cen_of(rot)

#    assert np.allclose(axis0, axis, rtol=1e-5)
#    assert np.allclose(ang0, ang, rtol=1e-5)
#    #  check rotation doesn't move cen
#    cenhat = (rot @ cen[..., None]).squeeze()
#    assert np.allclose(cen + hel, cenhat, rtol=1e-4, atol=1e-4)
#    assert np.allclose(np.linalg.norm(axis, axis=-1), 1.0)

# def test_th_intersect_planes():

#    p1 = torch.tensor([0., 0, 0, 1], requires_grad=True)
#    n1 = torch.tensor([1., 0, 0, 0], requires_grad=True)
#    p2 = torch.tensor([0., 0, 0, 1], requires_grad=True)
#    n2 = torch.tensor([0., 1, 0, 0], requires_grad=True)
#    isct, norm, status = thintersect_planes(p1, n1, p2, n2)
#    assert status == 0
#    assert isct[2] == 0
#    assert np.allclose(abs(norm[:3].detach()), (0, 0, 1))

#    p1 = torch.tensor([0., 0, 0, 1], requires_grad=True)
#    n1 = torch.tensor([1., 0, 0, 0], requires_grad=True)
#    p2 = torch.tensor([0., 0, 0, 1], requires_grad=True)
#    n2 = torch.tensor([0., 0, 1, 0], requires_grad=True)
#    isct, norm, status = thintersect_planes(p1, n1, p2, n2)
#    assert status == 0
#    assert isct[1] == 0
#    assert np.allclose(abs(norm[:3].detach()), (0, 1, 0))

#    p1 = torch.tensor([0., 0, 0, 1], requires_grad=True)
#    n1 = torch.tensor([0., 1, 0, 0], requires_grad=True)
#    p2 = torch.tensor([0., 0, 0, 1], requires_grad=True)
#    n2 = torch.tensor([0., 0, 1, 0], requires_grad=True)
#    isct, norm, status = thintersect_planes(p1, n1, p2, n2)
#    assert status == 0
#    assert isct[0] == 0
#    assert np.allclose(abs(norm[:3].detach()), (1, 0, 0))

#    p1 = torch.tensor([7., 0, 0, 1], requires_grad=True)
#    n1 = torch.tensor([1., 0, 0, 0], requires_grad=True)
#    p2 = torch.tensor([0., 9, 0, 1], requires_grad=True)
#    n2 = torch.tensor([0., 1, 0, 0], requires_grad=True)
#    isct, norm, status = thintersect_planes(p1, n1, p2, n2)
#    assert status == 0
#    assert np.allclose(isct[:3].detach(), [7, 9, 0])
#    assert np.allclose(abs(norm.detach()), [0, 0, 1, 0])

#    p1 = torch.tensor([0., 0, 0, 1], requires_grad=True)
#    n1 = torch.tensor([1., 1, 0, 0], requires_grad=True)
#    p2 = torch.tensor([0., 0, 0, 1], requires_grad=True)
#    n2 = torch.tensor([0., 1, 1, 0], requires_grad=True)
#    isct, norm, status = thintersect_planes(p1, n1, p2, n2)
#    assert status == 0
#    assert np.allclose(abs(norm.detach()), hnormalized([1, 1, 1, 0]))

#    p1 = np.array([[0.39263901, 0.57934885, -0.7693232, 1.],
#                   [-0.80966465, -0.18557869, 0.55677976, 0.]]).T
#    p2 = np.array([[0.14790894, -1.333329, 0.45396509, 1.],
#                   [-0.92436319, -0.0221499, 0.38087016, 0.]]).T
#    isct2, sts = intersect_planes(p1, p2)
#    isct2, norm2 = isct2.T
#    p1 = torch.tensor([0.39263901, 0.57934885, -0.7693232, 1.])
#    n1 = torch.tensor([-0.80966465, -0.18557869, 0.55677976, 0.])
#    p2 = torch.tensor([0.14790894, -1.333329, 0.45396509, 1.])
#    n2 = torch.tensor([-0.92436319, -0.0221499, 0.38087016, 0.])
#    isct, norm, sts = thintersect_planes(p1, n1, p2, n2)
#    assert sts == 0
#    assert torch.all(thray_in_plane(p1, n1, isct, norm))
#    assert torch.all(thray_in_plane(p2, n2, isct, norm))
#    assert np.allclose(isct.detach(), isct2)
#    assert np.allclose(norm.detach(), norm2)

#    p1 = torch.tensor([2., 0, 0, 1], requires_grad=True)
#    n1 = torch.tensor([1., 0, 0, 0], requires_grad=True)
#    p2 = torch.tensor([0., 0, 0, 1], requires_grad=True)
#    n2 = torch.tensor([0., 0, 1, 0], requires_grad=True)
#    isct, norm, status = thintersect_planes(p1, n1, p2, n2)

#    assert status == 0
#    assert abs(thdot(n1, norm)) < 0.0001
#    assert abs(thdot(n2, norm)) < 0.0001
#    assert torch.all(thpoint_in_plane(p1, n1, isct))
#    assert torch.all(thpoint_in_plane(p2, n2, isct))
#    assert torch.all(thray_in_plane(p1, n1, isct, norm))
#    assert torch.all(thray_in_plane(p2, n2, isct, norm))

# def test_th_axis_angle_cen_hel():
#    torch = pytest.importorskip('torch')
#    torch.autograd.set_detect_anomaly(True)

#    axis0 = torch.tensor(hnormalized(np.array([1., 1, 1, 0])), requires_grad=True)
#    ang0 = torch.tensor(0.9398483, requires_grad=True)
#    cen0 = torch.tensor([1., 2, 3, 1], requires_grad=True)
#    h0 = torch.tensor(2.443, requires_grad=True)
#    x = throt(axis0, ang0, cen0, h0)
#    axis, ang, cen, hel = thaxis_angle_cen_hel(x)
#    ax2, an2, cen2 = axis_ang_cen_of(x.detach().numpy())
#    assert np.allclose(ax2, axis.detach())
#    assert np.allclose(an2, ang.detach())

#    assert np.allclose(cen2, cen.detach())
#    hel.backward()
#    assert np.allclose(ang0.grad, 0, atol=1e-5)
#    hg = h0.detach().numpy() * np.sqrt(3) / 3
#    assert np.allclose(axis0.grad, [hg, hg, hg, 0])

#    assert 0

# def test_torch_grad():
#    x = torch.tensor([2, 3, 4], dtype=torch.float, requires_grad=True)
#    s = torch.sum(x)
#    s.backward()
#    assert np.allclose(x.grad.detach().numpy(), [1., 1., 1.])

# def test_torch_quat():
#    torch = pytest.importorskip('torch')
#    torch.autograd.set_detect_anomaly(True)

#    for v in (1., -1.):
#       q0 = torch.tensor([v, 0., 0., 0.], requires_grad=True)
#       q = thquat_to_upper_half(q0)
#       assert np.allclose(q.detach(), [1, 0, 0, 0])
#       s = torch.sum(q)
#       s.backward()
#       assert q0.is_leaf
#       assert np.allclose(q0.grad.detach(), [0, v, v, v])

# def test_torch_rmsfit():
#    torch = pytest.importorskip('torch')
#    torch.autograd.set_detect_anomaly(True)

#    ntrials = 100
#    n = 10
#    std = 1
#    shift = 100
#    for std in (0.001, 0.01, 0.1, 1, 10):
#       for i in range(ntrials):
#          # xform = throt([0, 0, 1], 120, degrees=True)
#          xform = torch.tensor(rand_xform(), requires_grad=True)
#          p = thrandpoint(10, std=10)
#          q = thxform(xform, p)
#          delta = thrandvec(10, std=std)
#          q = q + delta
#          q = q + shift
#          q[:, 3] = 1
#          # ic(delta)
#          # ic(q - p)

#          # ic(p.dtype, q.dtype)

#          rms, qhat, xpq = thrmsfit(p, q)
#          # ic(xpq)

#          assert np.allclose(qhat.detach().numpy(), thxform(xpq, p).detach().numpy())
#          assert np.allclose(rms.detach().numpy(), thrms(qhat, q).detach().numpy())
#          assert rms < std * 3
#          # print(rms)
#          rms.backward()
