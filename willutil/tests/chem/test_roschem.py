import pytest

import willutil.chem.roschem
import willutil as wu

def test_extract_rosetta_chem_data():
   pytest.importorskip('pyrosetta')
   aas = ['LYS']
   aas = []
   dat = wu.chem.roschem.extract_rosetta_chem_data(store=True, aas=aas)
   for aa in aas:
      assert aa in dat
      assert set(dat[aa].keys()) == {'resinfo', 'resatominfo'}
   assert dat['LYS']['resinfo']['natoms'] == 22
   assert dat['LYS']['resinfo']['nheavyatoms'] == 9
   assert dat['LYS']['resinfo']['nheavyatoms'] == 9
   assert dat['LYS']['resinfo']['mainchain_atoms'] == [1, 2, 3]
   assert dat['LYS']['resinfo']['is_sidechain_amine'] == True
   assert dat['LYS']['resinfo']['chi_atoms'] == [[1, 2, 5, 6], [2, 5, 6, 7], [5, 6, 7, 8],
                                                 [6, 7, 8, 9]]

if __name__ == '__main__':
   test_extract_rosetta_chem_data()
