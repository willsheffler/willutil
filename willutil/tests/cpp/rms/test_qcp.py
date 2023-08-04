import willutil as wu
from willutil.cpp.rms import qcp_rms_float
import numpy as np

def main():
   test_qcp()

def test_qcp():
   t = wu.Timer()
   npts = 100
   for i in range(1):
      pts1 = wu.hrandpoint(npts).astype(np.float32)
      pts2 = wu.hrandpoint(npts).astype(np.float32)

      ic(pts1[:, :3].T @ pts2[:, :3])

      # pts2 = wu.hxform(wu.hrand(), pts1)
      t.checkpoint('setup')
      rms, fit, x = wu.hrmsfit(pts1, pts2)
      t.checkpoint('hrmsfit')
      rms2 = qcp_rms_float(pts1, pts2)
      t.checkpoint('qcp')
      assert np.allclose(rms, rms2)
   t.report()

if __name__ == '__main__':
   main()
