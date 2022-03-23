import pytest

import willutil.chem.roschem
import willutil as wu

def _test_extract_rosetta_chem_data():
   pytest.importorskip('pyrosetta')
   aas = ['LYS']
   # aas = []
   dat = wu.chem.roschem.extract_rosetta_chem_data(store=False, aas=aas)
   assert len(dat) == 1
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

def test_existing_rosetta_chem_data():
   dat = wu.chem.roschem.get_rosetta_chem_data()

   aas = set([
      '0AZ', 'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'CYZ', 'D0AZ', 'DALA', 'DARG', 'DASN', 'DASP',
      'DCYS', 'DCYZ', 'DGLN', 'DGLU', 'DHIS', 'DHIS_D', 'DHYP', 'DILE', 'DLEU', 'DLYS', 'DMET',
      'DPHE', 'DPRO', 'DSER', 'DTHR', 'DTRP', 'DTYR', 'DVAL', 'GLN', 'GLU', 'GLY', 'HIS', 'HIS_D',
      'HYP', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'VRT'
   ])
   assert len(dat) == len(aas)
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
   test_existing_rosetta_chem_data()
