import pytest
import tempfile
import numpy as np
import willutil as wu

def main():
   test_mmcif_read()
   test_mmcif_write()

@pytest.mark.xfail()
def test_mmcif_read():

   fname = wu.tests.test_data_path('pdb/3v86.cif.gz')
   pdb = wu.pdb.readcif(fname)
   assert pdb.cryst1 == 'CRYST1   35.470   35.470   40.160  90.00  90.00 120.00 P 3 2 1'
   assert np.allclose(
      pdb.ca(),
      np.array([[19.366, -17.079, 0.93], [16.667, -14.389, 1.241], [13.913, -16.736, 2.387], [16.384, -18.364, 4.778],
                [17.119, -15.046, 6.502], [13.444, -14.25, 6.07], [12.007, -16.828, 8.449], [15.213, -16.489, 10.487],
                [14.593, -12.801, 11.149], [11.128, -14.013, 12.109], [12.559, -16.11, 14.943],
                [15.172, -13.814, 16.455], [12.376, -11.253, 16.255], [10.064, -13.345, 18.425],
                [12.846, -14.625, 20.678], [13.711, -10.981, 21.31], [10.156, -9.88, 22.081], [10.246, -12.636, 24.691],
                [13.566, -11.568, 26.199], [12.35, -7.977, 26.473], [9.312, -9.335, 28.299], [10.885, -11.751, 30.771],
                [13.89, -9.558, 31.53], [11.574, -6.614, 32.072], [9.847, -8.624, 34.81], [13.283, -9.337, 36.3],
                [13.945, -5.672, 36.947]]))

@pytest.mark.xfail()
def test_mmcif_write():
   fname = wu.tests.test_data_path('pdb/3v86.cif.gz')
   pdb = wu.pdb.readcif(fname)

   with tempfile.TemporaryDirectory() as td:
      wu.pdb.dumpcif(f'{td}/test.cif', pdb)

if __name__ == '__main__':
   main()
