import willutil as wu

def test_rosetta_chem_data():
   dat = wu.chem.rosetta_chem_data()
   for aa in wu.chem.aa3:
      assert aa in dat
      assert set(dat[aa].keys()) == {'resatominfo', 'resinfo'}
   assert dat['LYS']['resinfo']['natoms'] == 22
   assert dat['LYS']['resinfo']['nheavyatoms'] == 9
   assert dat['LYS']['resinfo']['nheavyatoms'] == 9
   assert dat['LYS']['resinfo']['mainchain_atoms'] == [1, 2, 3]
   assert dat['LYS']['resinfo']['is_sidechain_amine'] == True
   assert dat['LYS']['resinfo']['chi_atoms'] == [[1, 2, 5, 6], [2, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9]]

if __name__ == '__main__':
   test_rosetta_chem_data()
