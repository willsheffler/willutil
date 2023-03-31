import pytest
import willutil as wu
import numpy as np

def main():
   test_add_bb_o(
      wu.tests.fixtures.ncac(),
      wu.tests.fixtures.ncaco(),
   )
   test_dssp(
      wu.tests.fixtures.ncac(),
      wu.tests.fixtures.ncaco(),
   )
   ic('TEST_DSSP MAIN DONE')

def test_add_bb_o(ncac, ncaco):
   test = wu.chem.add_bb_o_guess(ncac)
   # ic((wu.hnorm(test[:-1, 3] - ncaco[:-1, 3])))

   assert 0.15 > np.mean(wu.hnorm(test[:-1] - ncaco[:-1]))

def test_dssp(ncac, ncaco):
   pytest.importorskip('mdtraj')
   ss = wu.chem.dssp(ncaco)
   assert len(ss) == len(ncaco)

   ncaco_hat = wu.chem.add_bb_o_guess(ncac)
   s2 = wu.chem.dssp(ncaco_hat)

   assert ss == 'LLLLLLEEEEEEEELLLLEEEEEEEELLHHHHHHHHHHHHHHLLLLEEEEEELLLLEEEEEELLLLLLLL'
   assert s2 == 'LLLLLLEEEEEEEELLLLEEEEEEEELLHHHHHHHHHHHHHHHLLLEEEEEELLLLEEEEEELLLLLLLL'

   assert np.sum(np.array(list(ss)) == np.array(list(s2))) == len(ss) - 1

if __name__ == '__main__':
   main()