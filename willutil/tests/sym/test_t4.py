import pytest
import willutil as wu

@pytest.mark.skip
def test_t4():
   assert 0
   fname = wu.tests.test_data_path('pdb/t4_ifaces.pdb')
   pdb = wu.pdb.readpdb(fname)
   # cr
   # ic(pdb.)
   assert 0

   t4 = np.load('/home/sheffler/T4.npy')
   t4[:, :3, :3] = np.swapaxes(t4[:, :3, :3], 1, 2)
   assert np.allclose(t4[:, 3, :3], 0)
   assert np.allclose(t4[:, 3, 3], 1)
   ic(np.linalg.det(-t4[0, :3, :3]))
   ic(t4[0, :3, :3])
   ic(wu.hvalid(t4))
   # assert 0
   t4 = wu.hxform(wu.hinv(t4[0]), t4)
   ic(t4.shape)
   # ic(t4[0])
   # wu.showme(t4)

def main():
   test_t4()

if __name__ == '__main__':
   main()
