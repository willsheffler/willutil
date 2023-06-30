import pytest
import willutil as wu
from willutil.motif.motif_manager import apply_sym_template

torch = pytest.importorskip('torch')

def main():
   test_apply_sym_template()

def test_apply_sym_template():
   t = wu.Timer()
   import torch
   t.checkpoint('torch')

   # fname = '/home/sheffler/tmp/rf_diffusion_test_dat_rfmd_0017_MB18_trkmd.pickle'
   # fname = '/home/sheffler/tmp/rf_diffusion_test_dat_rfmd_0012_MB06_trkmd.pickle'
   # fname = '/home/sheffler/tmp/rf_diffusion_test_dat_rfmd_0002_MB30_trkmd.pickle'
   # fname = '/home/sheffler/tmp/rf_diffusion_test_dat_seed4_rfmd_0004_EB00_trkmd.pickle'
   fname = '/home/sheffler/tmp/rf_diffusion_test_dat_seed4_rfmd_0018_MB12_trkmd.pickle'
   Rs, Ts, xyzorig, symmids, symmsub, symmRs, metasymm = wu.load(fname)
   t.checkpoint('Rs')

   # tpltcrd = wu.readpdb('/home/sheffler/project/symmmotif_HE/input/poscontrol_tplt_1_renum.pdb').ncac()
   # wu.save(tpltcrd, '/home/sheffler/tmp/rf_diffusion_test_dat_tpltcrd.pickle')
   tpltcrd = wu.load('/home/sheffler/tmp/rf_diffusion_test_dat_tpltcrd.pickle')
   tpltcrd = torch.tensor(tpltcrd).to(Rs.device)
   tpltidx = torch.arange(len(tpltcrd)).to(Rs.device)
   t.checkpoint('tpltcrd')

   apply_sym_template(Rs, Ts, xyzorig, symmids, symmsub, symmRs, metasymm, tpltcrd, tpltidx)
   t.checkpoint('apply_sym_template')

   t.report()

if __name__ == '__main__':
   main()