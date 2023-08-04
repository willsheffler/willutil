import willutil as wu
from willutil.cpp.rms import qcp_rms_double, qcp_rms_align_double, qcp_rms_regions_f4i4
import numpy as np

def main():
   test_qcp_regions_simple_1seg()
   test_qcp_regions_simple_2seg()
   # test_qcp_regions_simple_Nseg()

   test_qcp(niter=10)
   test_qcp_align(niter=10)

   print('test_qcp PASS', flush=True)

def test_qcp_regions_simple_1seg():
   N = 100
   pts1 = wu.hrandpoint(N + 10).astype(np.float32)
   pts2 = wu.hrandpoint(N).astype(np.float32)
   # pts1[:100] = pts1[:100] * 0.01 + pts2 * 0.99
   offsets = np.arange(10).reshape(10, 1)
   # offsets = np.tile(offsets, (170_000, 1))
   # ic(offsets.shape)
   # with wu.Timer():
   rms = qcp_rms_regions_f4i4(pts1, pts2, [N], offsets)
   # ic(rms)
   for i in range(10):
      rmsref = qcp_rms_double(pts1[i:N + i], pts2)
      assert rms.shape == (len(offsets), )
      assert np.allclose(rms[i], rmsref, atol=1e-4)

def test_qcp_regions_simple_2seg():
   N = 40
   pts1 = wu.hrandpoint(N).astype(np.float32)
   pts2 = wu.hrandpoint(N).astype(np.float32)
   rmsref = qcp_rms_double(pts1, pts2)
   # pts1[:, :3] -= pts1[:, :3].mean(axis=0).reshape(1, 3)
   # pts2[:, :3] -= pts2[:, :3].mean(axis=0).reshape(1, 3)
   for i in range(5, 35):
      sizes = [i, N - i]
      offsets = np.array([[0, i]], dtype='i4')
      rms = qcp_rms_regions_f4i4(pts1, pts2, sizes, offsets)
      assert rms.shape == (len(offsets), )
      assert np.allclose(rms, rmsref, atol=1e-4)

def test_qcp_regions():
   pts100 = wu.hrandpoint(100).astype(np.float32)
   pts40 = wu.hrandpoint(40).astype(np.float32)
   sizes = [19, 21]
   # sizes = [5, 7, 11, 13]
   offsets = np.array([[30, 60], [31, 60], [30, 61]], dtype='i4')
   # offsets = np.array([[10, 20, 30, 40]], dtype='i4')
   rms = qcp_rms_regions_f4i4(pts100, pts40, sizes, offsets)
   assert rms.shape == (len(offsets), )

def test_qcp_align(niter=20, npts=50):

   for i in range(niter):
      pts1 = wu.hrandpoint(npts)
      pts2 = wu.hrandpoint(npts)
      pts1copy, pts2copy = pts1.copy(), pts2.copy()
      rms, fit, x = wu.hrmsfit(pts1, pts2)
      rms2, R, T = qcp_rms_align_double(pts1, pts2)
      assert np.allclose(rms, rms2)
      assert np.allclose(x[:3, :3], R)
      assert np.allclose(x[:3, 3], T)
      assert np.allclose(pts1, pts1copy)
      assert np.allclose(pts2, pts2copy)

def test_qcp(niter=100, npts=50):
   for i in range(niter):
      pts1 = wu.hrandpoint(npts)
      pts2 = wu.hrandpoint(npts)
      rms, fit, x = wu.hrmsfit(pts1, pts2)
      rms2 = qcp_rms_double(pts1, pts2)
      assert np.allclose(rms, rms2)

if __name__ == '__main__':
   main()
