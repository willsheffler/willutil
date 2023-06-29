import willutil as wu

def apply_sym_template(Rs, Ts, xyzorig, symmids, symmsub, symmRs, metasymm, tpltcrd, tpltidx):
   import torch

   assert len(Rs) == len(Ts) == len(xyzorig) == 1
   Rs, Ts, xyzorig = Rs[0], Ts[0], xyzorig[0]
   nres, nsub = len(Rs) // len(symmsub), len(symmsub)
   assert torch.all(tpltidx < nres)

   # assert 0
   wu.showme(tpltcrd, name='template_in')
   wu.showme(wu.th_construct(Rs, Ts), name='RT_in')
   tmp = einsum('nij,naj->nai', Rs, xyzorig) + Ts.unsqueeze(-2)
   wu.showme(tmp.reshape(len(symmsub), -1, 3, 3), name='pose_in')

   #

   Ts[tpltidx] = tpltcrd[:, 1]
   Ts = torch.einsum('sij,rj->sri', symmRs[symmsub], Ts[:nres])
   ic(Ts.shape)
   Ts = Ts.reshape(-1, 3)

   #
   wu.showme(tpltcrd, name='template_out')
   wu.showme(wu.th_construct(Rs, Ts), name='RT_out')
   tmp = einsum('nij,naj->nai', Rs, xyzorig) + Ts.unsqueeze(-2)
   wu.showme(tmp.reshape(len(symmsub), -1, 3, 3), name='pose_out')

   return
