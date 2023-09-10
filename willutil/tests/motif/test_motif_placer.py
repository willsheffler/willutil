import random, pytest
import willutil as wu
from willutil.motif.motif_placer import *
import numpy as np
import torch
from opt_einsum import contract as einsum

def main():
   # a = torch.rand(10000) < 0.5
   # b = torch.rand(10000) < 0.5
   # c = a * b
   # a[a.clone()] = b[a]
   # assert torch.all(a == c)
   # assert 0, 'PASS'

   debug_symbridge_minfunc()
   assert 0
   test_rotation_point_match()
   debug_symbridge_rot_point_match()

   debug_polymotif()

   test_motif_occlusion()

   test_check_offsets_overlap_containment()
   test_motif_placer()
   test_motif_placer_minbeg_minend()
   perftest_motif_placer()

   print('test_motif_placer PASS', flush=True)

def rotations_point_match(beg, end, angle, nsamp):
   beg, end = wu.hpoint(beg), wu.hpoint(end)
   assert beg.shape == (4, ) and end.shape == (4, )
   if abs(angle) > 2 * np.pi: angle = np.radians(angle)
   cen = (beg + end) / 2
   dist = np.linalg.norm(end - beg)
   dirn = wu.hnormalized(end - beg)
   offset = np.tan(np.pi / 2 - angle / 2) * dist / 2
   # ic(cen)
   # ic(dirn)
   # ic(dist)
   # ic(offset)
   # perpref = np.array([0.34034305, 0.30650766, 0.888943, 0.])
   perpref = wu.hrandvec()
   axis0 = wu.hnormalized(wu.hprojperp(dirn, perpref))
   cen0 = wu.hrot(dirn, -np.pi / 2, cen) @ axis0
   cen0 = offset * cen0 + cen
   rots = wu.hrot(dirn, np.linspace(0, 2 * np.pi, nsamp), cen)
   axs = wu.hxform(rots, axis0)
   cen = wu.hxform(rots, cen0)
   matchrots = wu.hrot(axs, angle, cen)
   return matchrots, axs, cen

def test_rotation_point_match():
   for angle in range(60, 181, 10):
      beg, end = wu.hrandpoint(2)
      xptmatch, maxs, mcen = rotations_point_match(beg, end, angle, 10)
      assert np.allclose(end, wu.hxform(xptmatch, beg))

def _degbug_symbridge_make_test_coords(reg1, reg2, nfold, xsd=1, xyzsd=0):
   reg2 = wu.th_xform(wu.th_rot([0, 0, 1], torch.pi * 2 / nfold), reg2)
   x = torch.as_tensor(wu.hrandsmall(cart_sd=xsd, rot_sd=xsd / 15))
   reg1cen, reg2cen = reg1.mean(axis=(0, 1)), reg2.mean(axis=(0, 1))
   # reg1cen, reg2cen = torch.zeros(3), torch.zeros(3)
   xyznoise1 = torch.as_tensor(wu.hrandpoint(reg1.shape[:-1], std=xyzsd)[..., :3])
   xyznoise2 = torch.as_tensor(wu.hrandpoint(reg1.shape[:-1], std=xyzsd)[..., :3])
   reg1 = wu.th_xform(x, reg1 - reg1cen) + reg1cen + xyznoise1
   reg2 = wu.th_xform(x, reg2 - reg2cen) + reg2cen + xyznoise2
   return [reg1, reg2]

def cyclic_symbridge_rms(R0, T0, A0, C0, xyz, motif, nfold):
   motif0 = einsum('ij,raj->rai', R0, motif[0]) + T0
   motif1 = einsum('ij,raj->rai', R0, motif[1]) + T0
   xsym = wu.th_rot(A0, torch.pi * 2 / nfold, C0)
   motif1 = wu.th_xform(xsym, motif1)
   rms = (xyz[0] - motif0).square().mean()
   rms += (xyz[1] - motif1).square().mean()
   return torch.sqrt(rms)

def debug_symbridge_minfunc():
   pdb = wu.readpdb('/home/sheffler/project/dborigami/input/pep_abc_motif_min.pdb')
   cachains, _ = pdb.atomcoords(['N', 'CA', 'C'], splitchains=True)
   motif0 = torch.tensor(cachains[0], dtype=torch.float32)
   motif1 = torch.tensor(cachains[2], dtype=torch.float32)
   motif2 = torch.tensor(cachains[1], dtype=torch.float32)

   nfold = 3
   xyz1, xyz2 = _degbug_symbridge_make_test_coords(motif0, motif1, nfold=nfold, xsd=4, xyzsd=0)
   xyz2sym = wu.th_xform(wu.th_rot([0, 0, 1], -torch.pi * 2 / nfold), xyz2)

   wu.showme(torch.cat([xyz1, xyz2, xyz2sym]), name='ref')
   wu.showme(torch.cat([motif0, motif1]), name='motif')
   assert 0

   def Q2R(Q):
      Qs = torch.cat((torch.ones((1), device=Q.device), Q), dim=-1)
      Qs = normQ(Qs)
      return Qs2Rs(Qs[None, :]).squeeze(0)

   with torch.enable_grad():
      T0 = torch.zeros(3, device=xyz1.device).requires_grad_(True)
      Q0 = torch.zeros(3, device=xyz1.device).requires_grad_(True)
      A0 = torch.tensor([0.0, 0.0, 1.0], device=xyz1.device).requires_grad_(True)
      C0 = torch.zeros(3, device=xyz1.device).requires_grad_(True)

   lbfgs = torch.optim.LBFGS([T0, Q0, A0, C0], history_size=10, max_iter=4, line_search_fn="strong_wolfe")

   def closure():
      lbfgs.zero_grad()
      loss = cyclic_symbridge_rms(Q2R(Q0), T0, A0, C0, [xyz1, xyz2], [motif0, motif1], nfold)
      # ic(A0, C0)
      loss.backward()
      return loss

   # wu.showme(torch.cat([xyz1, xyz2, xyz2sym]), name='ref')
   # wu.showme(torch.cat([motif0, motif1]), name='motif')

   for i in range(40):
      loss = lbfgs.step(closure)
      ic(loss)
      motif0b = einsum('ij,raj->rai', Q2R(Q0), motif0) + T0
      motif1b = einsum('ij,raj->rai', Q2R(Q0), motif1) + T0
      # wu.showme(torch.cat([motif0b, motif1b]), name=f'minmotif{i}')

def debug_symbridge_rot_point_match():
   pdb = wu.readpdb('/home/sheffler/project/dborigami/input/pep_abc_motif_min.pdb')
   cachains, _ = pdb.atomcoords(['N', 'CA', 'C'], splitchains=True)
   ic(len(cachains))
   ic(cachains[0].shape)
   ic(cachains[1].shape)
   ic(cachains[2].shape)
   # wu.showme(cachains[0])
   # wu.showme(cachains[1])
   # wu.showme(cachains[2])

   SYMANG = 70
   # for SYMANG in range(90, 181, 10):
   reg1 = cachains[0][:9]
   reg2 = cachains[2][:9]
   target = wu.hxform(wu.hrot([0, 0, 1], SYMANG), reg2)
   target[..., 0] += 20
   target[..., 1] += 10
   target[..., 2] += 30

   targetcom = target.mean(axis=(0, 1))
   reg2com = reg2.mean(axis=(0, 1))
   xptmatch, maxs, mcen = rotations_point_match(reg2com, targetcom, SYMANG, 100)
   assert np.allclose(targetcom, wu.hxform(xptmatch, reg2com))
   wu.showme(mcen + maxs)
   wu.showme(mcen)
   _debug_helper_show_close_xforms(reg1, reg2, xptmatch, targetcom, thresh=1)
   return

   CENSD = 30
   nsamp = 1_000_000
   # AXISSD = 0.3
   # randaxis = np.random.normal(size=(nsamp, 3)) * AXISSD
   # randaxis[:, 2] = 1
   randaxis = np.random.normal(size=(nsamp, 3))
   randaxis = wu.hnormalized(randaxis)
   totgt = (target.mean(axis=(0, 1)) - reg2.mean(axis=(0, 1)))
   randaxis = wu.hnormalized(wu.hprojperp(totgt, randaxis))
   randcen = wu.hpoint(np.random.normal(size=(nsamp, 3)) * CENSD)
   ic(randcen.shape)
   randcen[:, :3] += (target.mean(axis=(0, 1)) + reg2.mean(axis=(0, 1))) / 2
   # ic(randcen.shape, randaxis.shape)
   xformsrand = wu.hrot(randaxis, SYMANG, randcen)
   # ic(xformsrand.shape)
   _debug_helper_show_close_xforms(reg1, reg2, xformsrand, targetcom, randaxis, randcen)

def _debug_helper_show_close_xforms(reg1, reg2, xformsrand, targetcom, randaxis=None, randcen=None, thresh=1, showall=False):

   xreg2 = wu.hxform(xformsrand, reg2)

   coms = xreg2.mean(axis=(1, 2))

   # ic(targetcom)
   # ic(coms[:3])
   dist = np.linalg.norm(targetcom[:3] - coms, axis=-1)
   # ic(dist)
   isclose = dist < thresh
   if showall: isclose[:] = True
   # ic(isclose.shape, isclose.sum())
   if randaxis is None or randcen is None:
      axs, ang, cen = wu.haxis_ang_cen_of(xformsrand[isclose])
   else:
      axs = randaxis[isclose]
      cen = randcen[isclose]

   wu.showme(reg1, name='reg1')
   from pymol import cmd
   cmd.set('suspend_updates', 'on')
   for samp, axs, cen in zip(xreg2[isclose], axs, cen):
      # axvis = np.stack([cen, cen + axs, cen + 2 * axs])[None, :, :3]
      # axsamp = np.concatenate([samp, axvis], axis=0)
      axsamp = samp
      wu.showme(axsamp, name='samp')
   cmd.hide('car')
   cmd.set('suspend_updates', 'off')

def debug_polymotif():
   fnames, coords = wu.load('/home/sheffler/project/multimotif/input/lanth_polymotif_2h_176_10x10.pickle')
   coords = [torch.as_tensor(c) for c in coords]
   # wu.showme(coords[0][0])
   dist = [[None, None], [None, None]]
   for i, j in [(0, 0), (0, 1), (1, 1)]:
      dist[i][j] = torch.cdist(coords[i][:, :, 1], coords[j][:, :, 1])
      assert torch.allclose(torch.cdist(coords[i][113, :, 1], coords[j][113, :, 1]), dist[i][j][113])

def debug_polymotif_read():
   flist = open('/home/sheffler/project/multimotif/input/lanth_polymotif_list_1.txt').read().split()
   # ic(xyz)
   from collections import defaultdict
   counter = defaultdict(lambda: 0)
   fnames = list()
   coords = [list(), list()]
   for f in flist:
      pdb = wu.readpdb(f)
      xyz, mask = pdb.atomcoords(['N', 'CA', 'C'], splitchains=True, removeempty=True)
      sizes = tuple(sorted([len(_) for _ in xyz]))
      counter[sizes] += 1
      if sizes == (10, 10):
         fnames.append(f)
         coords[0].append(xyz[0])
         coords[1].append(xyz[1])
      # group into shapes
      # read into fnames, coordstack, diststack
      # create new drms func
      # xyz, mask = wu.readpdb(f).atomcoords(['N', 'CA', 'C'], splitchains=True, nomask=True, removeempty=True)
   ic(counter)

   coords = [np.stack(c) for c in coords]
   wu.save((fnames, coords), 'lanth_polymotif_2h_176_10x10.pickle')

   assert 0

def randslice(n):
   r = random.randrange(n), random.randrange(n)
   return slice(min(r), max(r))

def test_motif_occlusion():
   xyz = torch.as_tensor(wu.tests.load_test_data('xyz/1pgx_ncac.npy'), device='cpu')
   xyz = xyz[:30]

   nres, nasym = len(xyz), None
   cbreaks = get_symm_cbreaks(nres, nasym, cbreaks=[])
   sizes = [7, 13]
   # motif, motifpos = make_test_motif(xyz, sizes, rnoise=0.1, nasym=nasym, cbreaks=cbreaks)
   # print(motifpos)
   # xyz[0:3, :, 0] += 100000
   # xyz[6:9, :, 0] += 100000
   # contacts = (1000.0 > torch.cdist(xyz[:, 1], xyz[:, 1])).to(torch.float32)

   # placement = place_motif_dme_fast(xyz, motif, nasym=nasym, cbreaks=cbreaks)

   contacts = (10.0 > torch.cdist(xyz[:, 1], xyz[:, 1])).to(torch.float32)
   # contacts[:] = 1
   # torch.manual_seed(0)
   # contacts = torch.rand((len(xyz), len(xyz))) < 0.5
   # ic(contacts.to(int))

   occ = compute_offset_occlusion_tensor(contacts, sizes)
   # ic(occ.shape)
   # ic(occ.to(int))

   occref, offsets = compute_offset_occlusion_brute(contacts, sizes, cbreaks=cbreaks)
   occtst = occ[tuple(offsets.T)]
   # ic(offsets)
   # ic(occref.to(int))
   # ic(occtst.to(int))
   assert np.allclose(occref, occtst)

   # showme_motif_placements(xyz, motif, placement.offset[:10])

   # assert 0, 'FAST_MOTIF_OCC???'

def perftest_motif_placer():
   t = wu.Timer()
   N = 140
   xyz = torch.tensor(wu.hrandpoint(3 * N)[:, :3].reshape(N, 3, 3), dtype=torch.float32, device='cuda')

   nres, nasym = len(xyz), None
   cbreaks = get_symm_cbreaks(nres, nasym, cbreaks=[])
   junct = 0
   sizes = [15, 15, 15, 15]
   # ic(sum(sizes))
   # ic(cbreaks)

   motif, motifpos = make_test_motif(xyz, sizes, rnoise=0.1, nasym=nasym, cbreaks=cbreaks)
   # ic(motifpos)
   t.checkpoint('make_test_motif')

   _ = place_motif_dme_fast(xyz[:40], motif, nasym=nasym, cbreaks=cbreaks, junct=junct, nolapcheck=10, nrmsalign=10)
   t.checkpoint('fastdme_init')
   result = place_motif_dme_fast(
      xyz,
      motif,
      nasym=nasym,
      cbreaks=cbreaks,
      junct=junct,
      return_alldme=True,
      nolapcheck=1000,
      nrmsalign=1,
      motif_occlusion_weight=0.1,
      motif_occlusion_dist=10,
   )
   t.checkpoint('fastdme')
   # t.report()
   assert np.allclose([m[0] for m in motifpos], result.offset[0])

   runtime = sum(t.checkpoints['fastdme'])
   ncalc = result.alldme.nelement()
   print(f'fastdme ncalc: {ncalc:,}, rate: {ncalc/runtime/1e9:7.5}G', flush=True)

   # ic(result.rms[:4])
   # ic(result.drms[:4])
   # ic(motifpos, result.offset[0])

   # wu.viz.scatter(result.rms, result.drms)

def test_check_offsets_overlap_containment():
   AC, T, F, COOC = np.allclose, True, False, check_offsets_overlap_containment

   assert AC([T, F, F], COOC([0, 1, 2], sizes=[10], nres=30, nasym=10))
   assert AC([T, T, F], COOC([0, 10, 11], sizes=[10], nres=40, nasym=20))
   assert AC([F, T, T], COOC([0, 10, 11], sizes=[10], nres=40, minbeg=3))
   assert AC([T, T, F], COOC([0, 10, 28], sizes=[10], nres=40, minend=3))
   assert AC([T, T, F, T], COOC([0, 10, 11, 20], sizes=[10], nres=40, nasym=20))

def test_motif_placer(showme=False):
   xyz = torch.as_tensor(wu.tests.load_test_data('xyz/1pgx_ncac.npy'), device='cpu')
   nres, nasym = len(xyz), None
   cbreaks = get_symm_cbreaks(nres, nasym, cbreaks=[15, 38])
   junct = 10
   sizes = [13, 14]
   # ic(sum(sizes))
   # ic(cbreaks)

   motif, motifpos = make_test_motif(xyz, sizes, rnoise=0.1, nasym=nasym, cbreaks=cbreaks)

   # fastdme = place_motif_dme_fast(xyz, motif, nasym=nasym, cbreaks=cbreaks, junct=junct)
   fastdme = place_motif_dme_fast(xyz, motif, nasym=nasym, cbreaks=cbreaks, junct=junct, return_alldme=True)

   doffset, dme, alldo, alldme = place_motif_dme_brute(xyz, motif, nasym=nasym, cbreaks=cbreaks)

   x = fastdme.alldme[tuple(alldo.T)]
   if all([junct * 2 >= s for s in sizes]):
      assert torch.allclose(x, alldme)
   # else:
   # wu.viz.scatter(alldme, x)
   # roffset, rms, allro, allrms = place_motif_rms_brute(xyz, motif, nasym=nasym)
   # print(rms)
   # ok = torch.logical_and(allrms < 10, alldme < 10)
   # wu.viz.scatter(allrms[ok], alldme[ok])

   assert np.allclose([m[0] for m in motifpos], fastdme.offset[0])

   if showme:
      showme_motif_placements(xyz, motif, doffset)
      assert 0
   # t.report()
   # wu.showme(motif)
   # wu.showme(xyz)

def showme_motif_placements(xyz, motif, offsets):
   wu.showme(xyz, name='ref')
   for i, ofst in enumerate(offsets):
      scrd = torch.cat([xyz[o:o + len(m), 1] for o, m in zip(ofst, motif)])
      mcrd = wu.th_point(torch.cat(motif)[:, 1])
      rms, _, x = wu.th_rmsfit(mcrd, wu.th_point(scrd))
      wu.showme(wu.th_xform(x, torch.cat(motif)), name=f'rms{rms}')

def test_motif_placer_minbeg_minend(showme=False):
   for minbeg in [0, 3]:
      for minend in [0, 4]:
         pdb = wu.pdb.readpdb(wu.test_data_path('pdb/1pgx.pdb1.gz'))
         xyz = torch.tensor(pdb.ncac(), device='cpu')
         nres, nasym = len(xyz), None
         cbreaks = get_symm_cbreaks(nres, nasym, cbreaks=[20, 40, 60])
         junct = 10
         sizes = [12, 13]
         motif, motifpos = make_test_motif(xyz, sizes, rnoise=0.1, nasym=nasym, cbreaks=cbreaks, minbeg=minbeg, minend=minend)
         fastdme = place_motif_dme_fast(xyz, motif, nasym=nasym, cbreaks=cbreaks, junct=junct, return_alldme=True, minbeg=minbeg, minend=minend)
         doffset, dme, alldo, alldme = place_motif_dme_brute(xyz, motif, nasym=nasym, cbreaks=cbreaks, minbeg=minbeg, minend=minend)
         x = fastdme.alldme[tuple(alldo.T - minbeg)]
         if all([junct * 2 >= s for s in sizes]):
            assert torch.allclose(x, alldme)
         assert np.allclose([m[0] for m in motifpos], fastdme.offset[0])

def normQ(Q):
   """normalize a quaternions
    """
   return Q / torch.linalg.norm(Q, keepdim=True, dim=-1)

# ============================================================
def avgQ(Qs):
   """average a set of quaternions
    input dims:
    Qs - (B,N,R,4)
    averages across 'N' dimension
    """
   def areClose(q1, q2):
      return ((q1 * q2).sum(dim=-1) >= 0.0)

   N = Qs.shape[1]
   Qsum = Qs[:, 0] / N

   for i in range(1, N):
      mask = areClose(Qs[:, 0], Qs[:, i])
      Qsum[mask] += Qs[:, i][mask] / N
      Qsum[~mask] -= Qs[:, i][~mask] / N

   return normQ(Qsum)

def Rs2Qs(Rs):
   Qs = torch.zeros((*Rs.shape[:-2], 4), device=Rs.device)

   Qs[..., 0] = 1.0 + Rs[..., 0, 0] + Rs[..., 1, 1] + Rs[..., 2, 2]
   Qs[..., 1] = 1.0 + Rs[..., 0, 0] - Rs[..., 1, 1] - Rs[..., 2, 2]
   Qs[..., 2] = 1.0 - Rs[..., 0, 0] + Rs[..., 1, 1] - Rs[..., 2, 2]
   Qs[..., 3] = 1.0 - Rs[..., 0, 0] - Rs[..., 1, 1] + Rs[..., 2, 2]
   Qs[Qs < 0.0] = 0.0
   Qs = torch.sqrt(Qs) / 2.0
   Qs[..., 1] *= torch.sign(Rs[..., 2, 1] - Rs[..., 1, 2])
   Qs[..., 2] *= torch.sign(Rs[..., 0, 2] - Rs[..., 2, 0])
   Qs[..., 3] *= torch.sign(Rs[..., 1, 0] - Rs[..., 0, 1])

   return Qs

def Qs2Rs(Qs):
   Rs = torch.zeros((*Qs.shape[:-1], 3, 3), device=Qs.device)

   Rs[..., 0, 0] = Qs[..., 0] * Qs[..., 0] + Qs[..., 1] * Qs[..., 1] - Qs[..., 2] * Qs[..., 2] - Qs[..., 3] * Qs[..., 3]
   Rs[..., 0, 1] = 2 * Qs[..., 1] * Qs[..., 2] - 2 * Qs[..., 0] * Qs[..., 3]
   Rs[..., 0, 2] = 2 * Qs[..., 1] * Qs[..., 3] + 2 * Qs[..., 0] * Qs[..., 2]
   Rs[..., 1, 0] = 2 * Qs[..., 1] * Qs[..., 2] + 2 * Qs[..., 0] * Qs[..., 3]
   Rs[..., 1, 1] = Qs[..., 0] * Qs[..., 0] - Qs[..., 1] * Qs[..., 1] + Qs[..., 2] * Qs[..., 2] - Qs[..., 3] * Qs[..., 3]
   Rs[..., 1, 2] = 2 * Qs[..., 2] * Qs[..., 3] - 2 * Qs[..., 0] * Qs[..., 1]
   Rs[..., 2, 0] = 2 * Qs[..., 1] * Qs[..., 3] - 2 * Qs[..., 0] * Qs[..., 2]
   Rs[..., 2, 1] = 2 * Qs[..., 2] * Qs[..., 3] + 2 * Qs[..., 0] * Qs[..., 1]
   Rs[..., 2, 2] = Qs[..., 0] * Qs[..., 0] - Qs[..., 1] * Qs[..., 1] - Qs[..., 2] * Qs[..., 2] + Qs[..., 3] * Qs[..., 3]

   return Rs

if __name__ == '__main__':
   main()
