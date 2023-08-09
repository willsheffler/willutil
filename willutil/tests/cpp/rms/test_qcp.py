import willutil as wu
from willutil.cpp.rms import qcp_rms_double, qcp_rms_align_double, qcp_rms_regions_f4i4
import numpy as np

def main():
   test_qcp_regions_junct_simple()
   # assert 0

   test_qcp_regions()
   test_qcp_regions_junct()

   test_qcp_regions_simple_1seg()
   test_qcp_regions_simple_2seg()
   test_qcp_regions_simple_Nseg()

   test_qcp(niter=10)
   test_qcp_align(niter=10)

   test_qcp_regions_perf()

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

def _random_int_partition(n, r):
   import random
   p = list()
   while sum(p) < n:
      p.append(random.randint(0, r))
   p[-1] = n - sum(p[:-1])
   return p

def _random_offsets(n, l, sizes):
   return np.stack([np.random.randint(0, l - s + 1, n) for s in sizes], axis=-1)

def test_qcp_regions_simple_Nseg():
   N = 40
   pts1 = wu.hrandpoint(N).astype(np.float32)
   pts2 = wu.hrandpoint(N).astype(np.float32)
   rmsref = qcp_rms_double(pts1, pts2)
   # pts1[:, :3] -= pts1[:, :3].mean(axis=0).reshape(1, 3)
   # pts2[:, :3] -= pts2[:, :3].mean(axis=0).reshape(1, 3)
   for i in range(100):
      sizes = _random_int_partition(N, 10)
      offsets = np.cumsum([0] + sizes[:-1]).reshape(1, len(sizes))
      rms = qcp_rms_regions_f4i4(pts1, pts2, sizes, offsets)
      assert rms.shape == (len(offsets), )
      assert np.allclose(rms, rmsref, atol=1e-4)

def compute_rms_offsets_brute(pts1, pts2, sizes, offsets, junct=0):
   rms = np.empty(len(offsets))
   offsets2 = np.cumsum([0] + list(sizes[:-1]))
   crd2 = list()
   for s, o in zip(sizes, offsets2):
      if junct == 0 or 2 * junct >= s:
         crd2.append(pts2[o:o + s, :3])
      else:
         crd2.append(pts2[o:o + junct, :3])
         crd2.append(pts2[o + s - junct:o + s, :3])
   crd2 = np.concatenate(crd2)
   for i, ofst in enumerate(offsets):
      crd1 = list()
      for s, o in zip(sizes, ofst):
         if junct == 0 or 2 * junct >= s:
            crd1.append(pts1[o:o + s, :3])
         else:
            crd1.append(pts1[o:o + junct, :3])
            crd1.append(pts1[o + s - junct:o + s, :3])

      crd1 = np.concatenate(crd1)
      rms[i] = qcp_rms_double(crd1, crd2)
   return rms

def test_qcp_regions_perf():
   t = wu.Timer()
   for i in range(100):
      np.random.seed(0)
      pts1 = wu.hrandpoint(100).astype(np.float32)
      pts2 = wu.hrandpoint(50).astype(np.float32)
      sizes = _random_int_partition(50, 40)
      offsets = _random_offsets(1000, len(pts1), sizes)
      t.checkpoint('setup')
      rms = qcp_rms_regions_f4i4(pts1, pts2, sizes, offsets)
      t.checkpoint('qcp_rms_regions_f4i4')
   t.report()

def helper_test_qcp_regions(noffset=1, junct=0):
   pts1 = wu.hrandpoint(100).astype(np.float32)
   pts2 = wu.hrandpoint(50).astype(np.float32)
   sizes = _random_int_partition(50, 40)
   offsets = _random_offsets(noffset, len(pts1), sizes)
   rmsref = compute_rms_offsets_brute(pts1, pts2, sizes, offsets, junct=junct)
   rms = qcp_rms_regions_f4i4(pts1, pts2, sizes, offsets, junct=junct)
   assert np.allclose(rms, rmsref)

def test_qcp_regions():
   for i in range(1000):
      helper_test_qcp_regions(noffset=10, junct=0)

def test_qcp_regions_junct_simple():
   pts1 = wu.hrandpoint(9).astype(np.float32)[:, :3]
   pts2 = wu.hrandpoint(9).astype(np.float32)[:, :3]

   # pts1 -= pts1[:4].mean(axis=0).reshape(-1, 3)
   # pts2 -= pts2[:4].mean(axis=0).reshape(-1, 3)

   sizes = [9]
   offsets = [[0]]
   rmsref = compute_rms_offsets_brute(pts1, pts2, sizes, offsets, junct=4)
   rms = qcp_rms_regions_f4i4(pts1, pts2, sizes, offsets, junct=4)
   # ic(rmsref)
   # ic(rms)
   assert np.allclose(rms, rmsref)

def test_qcp_regions_junct():
   for j in range(2, 10):
      for i in range(10):
         helper_test_qcp_regions(noffset=10, junct=j)

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
