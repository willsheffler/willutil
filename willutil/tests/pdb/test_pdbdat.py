from willutil.pdb.pdbdat import PdbData
from willutil.pdb.pdbdat import _change_seq_ss_to_ids
from willutil import Timer
import numpy as np
import xarray as xr
import pytest

def test_pairdat_subset_by_aa(respairdat10):

   rpd = respairdat10
   # xr.set_options(display_max_rows=999)
   # print(rpd.data)
   assert len(rpd.data.pdbid) + 1 == len(rpd.data.pdb_res_offsets)
   assert len(rpd.data.pdbid) + 1 == len(rpd.data.pdb_pair_offsets)
   # print('pdb', len(rpd.pdb))
   # print('pdb_res_offsetfs', len(rpd.pdb_res_offsets))
   # print('pdb_res_offsetfs', rpd.pdb_res_offsets[-1], len(rpd.r_pdbid))
   # print('pdb_pair_offsetfs', len(rpd.pdb_pair_offsets))
   # print('pdb_pair_offsetfs', rpd.pdb_pair_offsets[-1], len(rpd.p_pdbid))
   # assert 0
   # for aas in ["ILV", "W", "MPQRTF"]:
   for aas in ["ILV"]:
      rp, keepers = rpd.subset_by_aa(aas, return_keepers=True)
      rp.sanitycheck()
      assert rp.keys() == rpd.keys()
      assert np.all(np.isin(rp.id2aa[rp.aaid], tuple(aas)))
      notaa = rpd.id2aa[rpd.aaid[~keepers]]
      assert not np.any(np.isin(notaa, tuple(aas)))

def test_pairdat_subset_by_ss(respairdat10):
   rpd = respairdat10
   for ss in ["H", "E", "L", "HE", "HL", "EL", "HEL"]:
      rp, keepers = rpd.subset_by_ss(ss, return_keepers=True)
      rp.sanitycheck()
      assert np.all(np.isin(rp.id2ss[rp.ssid], tuple(ss)))
      notss = rpd.id2ss[rpd.ssid[~keepers]]
      assert not np.any(np.isin(notss, tuple(ss)))
      assert rp.keys() == rpd.keys()
   with pytest.raises(ValueError):
      rpd.subset_by_ss('Q', return_keepers=True)

def test_pairdatextra_subset_by_aa(respairdat10_plus_xmap_rots):
   test_pairdat_subset_by_ss(respairdat10_plus_xmap_rots)

def test_pairdatextra_subset_by_ss(respairdat10_plus_xmap_rots):
   test_pairdat_subset_by_ss(respairdat10_plus_xmap_rots)

def test_tiny_subset_by_aa_pdb_removal_2pdb():
   rp = xr.Dataset(dict(
      pdb=(['pdbid'], ['a', 'b']),
      nres=(['pdbid'], [1, 2]),
      r_pdbid=(['resid'], [0, 1, 1]),
      seq=(['resid'], ['A', 'C', 'D']),
      ss=(['resid'], ['E', 'H', 'L']),
      resno=(['resid'], [0, 0, 1]),
      p_pdbid=(['pairid'], [1]),
      p_resi=(['pairid'], [1]),
      p_resj=(['pairid'], [2]),
   ), attrs=dict(
      pdb_res_offsets=[0, 1, 3],
      pdb_pair_offsets=[0, 0, 1],
   ))
   rp = PdbData(rp)
   rp.sanitycheck()
   _change_seq_ss_to_ids(rp)
   with pytest.raises(ValueError):
      rp.subset_by_aa('C')
   rpC = rp.subset_by_aa('CD')
   # print(rpC)
   rpC.sanitycheck()
   assert len(rpC.nres) == 1
   assert np.all(rpC.r_pdbid == [0])
   assert np.all(rpC.aaid == [1, 2])
   assert np.all(rpC.pdb_res_offsets == [0, 2])
   assert np.all(rpC.pdb_pair_offsets == [0, 1])

def test_tiny_subset_by_aa_pdb_removal_3pdb():
   rp = xr.Dataset(dict(
      pdb=(['pdbid'], ['a', 'b', 'c']),
      nres=(['pdbid'], [1, 2, 3]),
      r_pdbid=(['resid'], [0, 1, 1, 2, 2, 2]),
      resno=(['resid'], [0, 0, 1, 0, 1, 2]),
      seq=(['resid'], ['A', 'C', 'D', 'E', 'F', 'G']),
      ss=(['resid'], ['E', 'H', 'L', 'E', 'H', 'L']),
      p_pdbid=(['pairid'], [1, 2, 2, 2]),
      p_resi=(['pairid'], [1, 3, 3, 4]),
      p_resj=(['pairid'], [2, 4, 5, 5]),
   ), attrs=dict(
      pdb_res_offsets=[0, 1, 3, 6],
      pdb_pair_offsets=[0, 0, 1, 4],
   ))
   rp = PdbData(rp)
   rp.sanitycheck()
   _change_seq_ss_to_ids(rp)
   with pytest.raises(ValueError):
      rp.subset_by_aa('C')

   rpC = rp.subset_by_aa('EFG')
   rpC.sanitycheck()
   assert len(rpC.nres) == 1
   assert np.all(rpC.r_pdbid == [0])
   assert np.all(rpC.aaid == [3, 4, 5])
   assert np.all(rpC.p_resi == [0, 0, 1])
   assert np.all(rpC.p_resj == [1, 2, 2])
   assert np.all(rpC.pdb_res_offsets == [0, 3])
   assert np.all(rpC.pdb_pair_offsets == [0, 3])

   rpC = rp.subset_by_aa('AEFG')
   # print(rpC)
   rpC.sanitycheck()
   assert len(rpC.nres) == 2
   assert np.all(rpC.r_pdbid == [0, 1, 1, 1])
   assert np.all(rpC.aaid == [0, 3, 4, 5])
   assert np.all(rpC.pdb_res_offsets == [0, 1, 4])
   assert np.all(rpC.pdb_pair_offsets == [0, 0, 3])

def main():
   from willutil.tests.testdata import fixtures
   rp = fixtures.respairdat10()
   rpextra = fixtures.respairdat10_plus_xmap_rots()

   with Timer() as t:
      test_pairdat_subset_by_aa(rp)
      # test_pairdat_subset_by_ss(rp)
      # test_pairdatextra_subset_by_aa(rpextra)
      # test_pairdatextra_subset_by_ss(rpextra)
      # test_tiny_subset_by_aa_pdb_removal_2pdb()
      # test_tiny_subset_by_aa_pdb_removal_3pdb()

#
if __name__ == '__main__':
   main()
