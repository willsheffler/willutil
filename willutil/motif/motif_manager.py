import numpy as np
import torch
import willutil as wu

class Motif:
   def __init__(self, pdbfile, region, pointsym=None, wholesym=None, index=None, device=None):
      self._region = region
      self.label, self.res, self.motifres = self._region
      self.pointsym, self.wholesym = pointsym, wholesym
      if self.pointsym == 'None': self.pointsym = None
      self.device = device
      self.index = index
      # self.mainres = torch.arange(*self.res)
      # self.motifres = torch.arange(*self.motifres)

      self._set_coords(pdbfile)

   def _set_coords(self, pdbfile):
      self.pdbfile = pdbfile

      # ic(self.pointsym)
      # ic(self.wholesym)
      self.xyz = wu.readpdb(self.pdbfile).ncac(splitchains=True)
      self.symaxis = None
      if self.pointsym is not None:
         nfold = int(self.pointsym[1:])
         if len(self.xyz) != nfold:
            raise ValueError(f'nfold mismatch on {pdbfile}, motif pointsym is {self.pointsym}, '
                             f'but pdb has only {len(self.xyz)} chains')
         self.symaxis = torch.tensor(wu.sym.axes(self.wholesym, self.pointsym)[:3], device=self.device, dtype=torch.float32)
         symelem = wu.sym.SymElem(nfold, self.symaxis.cpu().numpy())
         if len(self.xyz) > 1:
            self.xyz = wu.sym.aligncx(self.xyz, symelem)
      self.xyz = torch.tensor(self.xyz, device=self.device, dtype=torch.float32)
      # shift away from origin along symmaxis, needed for dumb COM alignment to work
      if self.symaxis is not None: self.xyz += 30.0 * self.symaxis

      # self.stubs = stub_rots(self.xyz)
      # assert torch.allclose(torch.linalg.det(self.stubs), torch.tensor(1.0, dtype=self.stubs.dtype))

      assert self.motifres[0] == 0
      assert self.motifres[1] == len(self.xyz[0])

   def crappy_com_fit(self, curxyz, symmsub, symmRs):
      # crappy fit com
      dev = curxyz.device
      curcom = curxyz.mean(axis=(0, 1))
      motif_com = self.xyz[0].mean(axis=(0, 1))
      motif_comsym = einsum('sij,j->si', symmRs, motif_com)
      symcomdist = torch.linalg.norm(curcom - motif_comsym, axis=-1)
      whichsub = torch.argmin(symcomdist)
      motif_crdsub = einsum('ij,raj->rai', symmRs[whichsub], self.xyz[0])
      motif_comsub = motif_crdsub.mean(axis=(0, 1))
      cxaxis = einsum('ij,j->i', symmRs[whichsub], self.symaxis)
      zero = torch.tensor([0.0, 0.0, 0.0], device=dev)

      dang = dihedral(motif_comsub, zero, cxaxis, curcom)
      rotcom = wu.t_rot(cxaxis, dang)
      motif_fit = einsum('ij,raj->rai', rotcom, motif_crdsub)
      motif_fit += proj(cxaxis, curcom - motif_comsub).to(dev)

      return motif_fit

   def apply_motif_xyz(self, xyz, symmsub, symmRs):
      ic(xyz.shape)
      assert 0
      xyz = xyz.clone()
      motifres = torch.arange(*self.res, device=xyzorig.device)
      xyz[motifres] = self.crappy_com_fit(xyz[motifres], symmsub, symmRs)
      return xyz

   def apply_motif_RT(self, R, T, xyzorig, symmsub, symmRs):
      motifres = torch.arange(*self.res, device=xyzorig.device)
      curxyz = einsum('rij,raj->rai', R[motifres], xyzorig[motifres]) + T[motifres, None]
      motif_fit = self.crappy_com_fit(curxyz, symmsub, symmRs)
      oldstubs = stub_rots(xyzorig[motifres])
      newstubs = stub_rots(motif_fit)

      # ic(curcom)
      # ic(motif_com)
      # ic(whichsub)
      # ic(cxaxis)
      # # ic(motif_comsym)
      # wu.dumppdb('motif_crdsub.pdb', motif_crdsub)
      # wu.dumppdb('motif_fit.pdb', motif_fit)
      # wu.dumppdb('curxyz.pdb', curxyz)
      # assert 0

      T[motifres] = motif_fit[:, 1]
      R[motifres] = newstubs @ torch.linalg.inv(oldstubs)

      return R, T

   def __str__(self):
      axs = ','.join([str(float(_)) for _ in self.symaxis])
      return f'Motif(pdbfile={self.pdbfile}, region={self._region}, symaxis={axs})'

   def printinfo(self, prefix=''):
      print(f'{prefix}Motif:')
      print(f'{prefix}    label    {self.label}', flush=True)
      print(f'{prefix}    index    {self.index}', flush=True)
      print(f'{prefix}    res      {self.res}', flush=True)
      print(f'{prefix}    motifres {self.motifres}', flush=True)
      print(f'{prefix}    device   {self.device}', flush=True)
      print(f'{prefix}    symmetry {self.pointsym}', flush=True)
      print(f'{prefix}    symaxis  {self.symaxis}', flush=True)
      print(f'{prefix}    xyz      {self.xyz.shape}', flush=True)
      # print(f'{prefix}    stubs    {self.stubs.shape}', flush=True)

   def dumppdb(self, fname):
      wu.dumppdb(fname, self.xyz)

class MotifManager:
   def __init__(self, conf, device=None):
      self.conf = conf
      self.device = device
      self._set_regions(conf.rfmotif.regions)
      self._sanity_check(self.conf)
      self.motifs = list()
      # print(conf.rfmotif.pdbs)
      # print(conf.rfmotif.symaxes)
      for i, pdb in enumerate(conf.rfmotif.pdbs):
         self.motifs.append(Motif(
            pdb,
            self.motifregions[i],
            conf.rfmotif.pointsyms[i],
            conf.inference.symmetry,
            device=self.device,
            index=i,
         ))

   def apply_motifs_xyz(self, xyz, symmsub, symmRs):
      for m in motifs:
         xyz = m.apply_motif_xyz(xyz, symmsub, symmRs)
      return xyz

   def apply_motifs_RT(self, R_in, T_in, xyzorig_in, symmsub, symmRs, debuglabel=None):
      assert len(self.motifs), 'apply_motifs_RT called with no motifs, maybe this is a mistake...'
      assert len(R_in) == len(T_in) == len(xyzorig_in) == 1
      R, T, xyzorig = R_in[0], T_in[0], xyzorig_in[0]

      # debuglabel = 'TEST'
      if debuglabel:
         xyzIN = einsum('nij,naj->nai', R, xyzorig) + T.unsqueeze(-2)
         wu.dumppdb(f'{debuglabel}_IN.pdb', xyzIN.reshape(len(symmsub), -1, 3, 3))

      for m in self.motifs:
         R, T = m.apply_motif_RT(R, T, xyzorig, symmsub, symmRs)

      nasym = len(R) // len(symmsub)
      R = torch.einsum('sij,rjk,slk->sril', symmRs[symmsub], R[:nasym], symmRs[symmsub]).reshape(-1, 3, 3)
      T = torch.einsum('sij,rj->sri', symmRs[symmsub], T[:nasym]).reshape(-1, 3)
      # R symmetry is broken !?!?! maybe doesn't matter because frank resym later??

      if debuglabel:
         xyzOUT = einsum('nij,naj->nai', R, xyzorig) + T.unsqueeze(-2)
         wu.dumppdb(f'{debuglabel}_OUT.pdb', xyzOUT.reshape(len(symmsub), -1, 3, 3))
         assert 0

      R_in[0] = R
      T_in[0] = T
      return R_in, T_in

   def __str__(self):
      s = 'MotifManager(\n'
      for m in self.motifs:
         s += '    ' + str(m) + '\n'
      s += ')'
      return s

   def __bool__(self):
      return bool(self.motifs)

   def _sanity_check(self, conf):
      assert len(self.motifregions) == len(conf.rfmotif.pdbs)
      assert len(conf.rfmotif.pointsyms) in (0, len(conf.rfmotif.pdbs))

   def _set_regions(self, regions):
      self.regions, self.motifregions = list(), list()
      if not regions: return
      assert len(regions) == 1
      startres = 0
      for i, c in enumerate(regions[0].split(',')):
         label = None
         if not str.isnumeric(c[0]):
            lb, ub = [int(_) for _ in c[1:].split('-')]
            label, n, tbeg, tend = c[0], ub - lb + 1, lb - 1, ub
         else:
            label, n, tbeg, tend = None, int(c), None, None
         region = (label, (startres, startres + n), (tbeg, tend))
         self.regions.append(region)
         if label: self.motifregions.append(region)
         startres += n

   def active(self):
      return bool(self.motifs)

   def printinfo(self):
      print('MotifManager motifs:')
      for m in self.motifs:
         m.printinfo(prefix='    ')

   def dumppdbs(self, prefix):
      if prefix.endswith('.pdb'): prefix = prefix[:-4]
      for m in self.motifs:
         m.dumppdb(f'{prefix}_{m.index}_{m.label}.pdb')

def normalized(x):
   return x / torch.linalg.norm(x, axis=-1)[..., None]

def stub_rots(xyz):
   # stub definition a little strange... easy to see visually on top of bb coordinatesnn,
   N, CA, C = xyz[..., 0, :], xyz[..., 1, :], xyz[..., 2, :]
   a = normalized(CA - N)
   b = normalized(torch.linalg.cross(a, C - CA))
   c = torch.linalg.cross(a, b)
   stubs = torch.stack([a, b, c], axis=-1)
   assert np.allclose(1, torch.linalg.det(stubs).cpu().detach())

   return stubs

def set_RT_from_coords(newxyz, idx, xyzorig, Rs, Ts):
   oldstubs = stub_rots(xyzorig[idx])
   newstubs = stub_rots(newxyz)
   Ts[idx] = newxyz[:, 1]
   Rs[idx] = newstubs @ torch.linalg.inv(oldstubs)

def dot(a, b):
   return torch.sum(a * b)

def dihedral(p1, p2, p3, p4):
   a = normalized(p2 - p1)
   b = normalized(p3 - p2)
   c = normalized(p4 - p3)
   x = torch.clip(dot(a, b) * dot(b, c) - dot(a, c), -1, 1)
   m = torch.linalg.cross(b, c)
   y = torch.clip(dot(a, m), -1, 1)
   return torch.arctan2(y, x)

def proj(u, v):
   return dot(u, v) / dot(u, u) * u

def apply_sym_template(Rs, Ts, xyzorig, symmids, symmsub, symmRs, metasymm, tpltcrd, tpltidx):

   assert len(Rs) == len(Ts) == len(xyzorig) == 1
   Rs, Ts, xyzorig = Rs[0], Ts[0], xyzorig[0]
   Lasu, nsub = len(Rs) // len(symmsub), len(symmsub)
   assert torch.all(tpltidx < Lasu)
   dev = Rs.device

   stubs = stub_rots(xyzorig)

   # N = 10
   # print(xyzorig.shape)
   # print(stubs.shape)
   # print(normalized(xyzorig[N, 0] - xyzorig[N, 1]))
   # print(stubs[N, 0])
   # print('--------------', flush=True)
   # wu.showme(wu.th_construct(stubs)[N:N + 1], name='RT_in')
   # wu.showme(xyzorig[N:N + 1].reshape(1, 1, 3, 3), name='pose_in')
   # assert 0

   curxyz = einsum('rij,raj->rai', Rs[tpltidx], xyzorig[tpltidx]) + Ts[tpltidx, None]
   curcom = curxyz.mean(axis=(0, 1))
   tpltcom = tpltcrd.mean(axis=(0, 1))
   tpltcomsym = einsum('sij,j->si', symmRs, tpltcom)
   whichsub = torch.argmin(torch.linalg.norm(curcom - tpltcomsym, axis=-1))
   tpltcrdsub = einsum('ij,raj->rai', symmRs[whichsub], tpltcrd)
   tpltcomsub = tpltcrdsub.mean(axis=(0, 1))

   cxaxis = normalized(torch.tensor([1.0, 1.0, 1.0], device=dev))
   cxaxis = einsum('ij,j->i', symmRs[whichsub], cxaxis)
   zero = torch.tensor([0.0, 0.0, 0.0], device=dev)

   dang = dihedral(tpltcomsub, zero, cxaxis, curcom)
   rotcom = wu.t_rot(cxaxis, dang)
   templatefitcom = einsum('ij,raj->rai', rotcom, tpltcrdsub)
   templatefitcom += proj(cxaxis, curcom - tpltcomsub).to(dev)

   wu.showme(curxyz, name='curxyz')
   wu.showme(templatefitcom, name='templatefitcom')

   wu.showme(tpltcrd, name='template_in')
   wu.showme(tpltcrdsub, name='template_in_sub')
   wu.showme(wu.th_construct(Rs @ stubs, Ts), name='RT_in')
   tmp = einsum('rij,raj->rai', Rs, xyzorig) + Ts.unsqueeze(-2)
   wu.showme(tmp.reshape(len(symmsub), -1, 3, 3), name='pose_in')

   #

   #

   set_RT_from_coords(templatefitcom, tpltidx, xyzorig, Rs, Ts)

   Rs = torch.einsum('sij,rjk,slk->sril', symmRs[symmsub], Rs[:Lasu], symmRs[symmsub]).reshape(-1, 3, 3)
   Ts = torch.einsum('sij,rj->sri', symmRs[symmsub], Ts[:Lasu]).reshape(-1, 3)

   # check symmetry of Rs ??!?!?!

   #
   wu.showme(templatefitcom, name='template_out')
   wu.showme(wu.th_construct(Rs @ stubs, Ts), name='RT_out')
   tmp = einsum('rij,raj->rai', Rs, xyzorig) + Ts.unsqueeze(-2)
   wu.showme(tmp.reshape(len(symmsub), -1, 3, 3), name='pose_out')

   return

def t_rot(axis, angle, shape=(3, 3), squeeze=True):

   # axis = torch.tensor(axis, dtype=dtype, requires_grad=requires_grad)
   # angle = angle * np.pi / 180.0 if degrees else angle
   # angle = torch.tensor(angle, dtype=dtype, requires_grad=requires_grad)

   if axis.ndim == 1: axis = axis[None, ]
   if angle.ndim == 0: angle = angle[None, ]
   # if angle.ndim == 0
   if axis.shape and angle.shape and not is_broadcastable(axis.shape[:-1], angle.shape):
      raise ValueError(f'axis/angle not compatible: {axis.shape} {angle.shape}')
   zero = torch.zeros(*angle.shape)
   axis = th_normalized(axis)
   a = torch.cos(angle / 2.0)
   tmp = axis * -torch.sin(angle / 2)[..., None]
   b, c, d = tmp[..., 0], tmp[..., 1], tmp[..., 2]
   aa, bb, cc, dd = a * a, b * b, c * c, d * d
   bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
   if shape == (3, 3):
      rot = torch.stack([
         torch.stack([aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)], axis=-1),
         torch.stack([2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)], axis=-1),
         torch.stack([2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc], axis=-1),
      ], axis=-2)
   elif shape == (4, 4):
      rot = torch.stack([
         torch.stack([aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac), zero], axis=-1),
         torch.stack([2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab), zero], axis=-1),
         torch.stack([2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc, zero], axis=-1),
         torch.stack([zero, zero, zero, zero + 1], axis=-1),
      ], axis=-2)
   else:
      raise ValueError(f't_rot shape must be (3,3) or (4,4), not {shape}')
   # ic('foo')
   # ic(axis.shape)
   # ic(angle.shape)
   # ic(rot.shape)
   if squeeze and rot.shape == (1, 3, 3): rot = rot.reshape(3, 3)
   if squeeze and rot.shape == (1, 4, 4): rot = rot.reshape(4, 4)
   return rot
