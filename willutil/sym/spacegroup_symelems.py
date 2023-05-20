import itertools as it
from opt_einsum import contract as einsum
from collections import defaultdict
import numpy as np
import willutil as wu
from willutil.sym.spacegroup_data import *
from willutil.sym.spacegroup_util import *
from willutil.sym.SymElem import *
from willutil.homog.hgeom import h_point_line_dist, hvec, hpoint, line_line_closest_points_pa, hnorm, angle

def _inunit(p):
   x, y, z, w = p.T
   ok = ((-0.001 < x) * (x < 0.999) * (-0.001 < y) * (y < 0.999) * (-0.001 < z) * (z < 0.999))
   return ok

def _flipaxs(a):
   if np.sum(a[:3] * [1, 1.1, 1.2]) < 0:
      a[:3] *= -1
   return a

def _compute_symelems(spacegroup, unitframes=None, lattice=None, torch_device=None, aslist=False):
   if torch_device:
      try:
         import torch
      except ImportError:
         torch_device = False
   if unitframes is None:
      unitframes = wu.sym.sgframes(spacegroup, cellgeom='unit')

   if lattice is None:
      lattice = lattice_vectors(spacegroup, cellgeom='nonsingular')
      # a, b, c = 1, 1.23456789, 9.87654321
      # if sg_lattice[spacegroup] == 'CUBIC': lattice = np.diag([a, a, a])
      # elif sg_lattice[spacegroup] == 'TETRAGONAL': lattice = np.diag([a, a, b])
      # elif sg_lattice[spacegroup] == 'ORTHORHOMBIC': lattice = np.diag([a, b, c])
      # else: assert 0

   # if len(unitframes) < 4: ncell = 2
   # if len(unitframes) < 8: ncell = 2
   # unitframes = unitframes.astype(np.float32)
   f2cel = latticeframes(unitframes, lattice, cells=2)
   f4cel = latticeframes(unitframes, lattice, cells=4)

   # for f in f4cel:
   # if np.allclose(f[:3, :3], np.eye(3)) and f[0, 3] == 0 and f[2, 3] == 0:
   # print(f)

   # relframes = einsum('aij,bjk->abik', f2cel, wu.)
   f2geom = wu.homog.axis_angle_cen_hel_of(f2cel)
   axs, ang, cen, hel = f2geom
   axs, cen, hel = axs[:, :3], cen[:, :3], hel[:, None]
   flip = np.sum(axs * [3, 2, 1], axis=1) <= 0
   axs[flip] = -axs[flip]
   hel[flip] = -hel[flip]
   # flip = np.stack([flip, flip, flip], axis=1)
   # ic(flip.shape)
   # axs = np.where(flip, axs, -axs)
   # hel = np.where(flip[:, 0], hel, -hel)
   # ic(axs.shape)
   # ic(hel.shape)
   tag0 = np.concatenate([axs, cen, hel], axis=1).round(10)
   symelems = defaultdict(list)

   for nfold in [2, 3, 4, 6, -2, -3, -4, -6]:
      screw, nfold = nfold < 0, abs(nfold)
      nfang = 2 * np.pi / nfold

      # idx = np.isclose(ang, nfang, atol=1e-6)
      if screw:
         idx = np.logical_and(np.isclose(ang, nfang, atol=1e-6), ~np.isclose(0, hel[:, 0]))
      else:
         idx = np.logical_and(np.isclose(ang, nfang, atol=1e-6), np.isclose(0, hel[:, 0]))
      if np.sum(idx) == 0: continue

      # ic(tag0[idx])

      nftag = tag0[idx]
      nftag = nftag[np.lexsort(-nftag.T, axis=0)]
      nftag = np.unique(nftag, axis=0)

      nftag = nftag[np.argsort(-nftag[:, 0], kind='stable')]
      nftag = nftag[np.argsort(-nftag[:, 1], kind='stable')]
      nftag = nftag[np.argsort(-nftag[:, 2], kind='stable')]
      nftag = nftag[np.argsort(-nftag[:, 5], kind='stable')]
      nftag = nftag[np.argsort(-nftag[:, 4], kind='stable')]
      nftag = nftag[np.argsort(-nftag[:, 3], kind='stable')]

      d = np.sum(nftag[:, 3:6]**2, axis=1).round(6)
      nftag = nftag[np.argsort(d, kind='stable')]

      if torch_device:
         f5cell_torch = torch.tensor(f4cel, device=torch_device, dtype=torch.float32)

      # remove symmetric dups
      keep = nftag[:1]
      for itag, tag in enumerate(nftag[1:]):

         if torch_device:
            symtags = _make_symtags_torch(tag, f5cell_torch, torch_device, t)
            seenit = torch.all(torch.isclose(torch.tensor(keep[None], dtype=torch.float32, device=torch_device), symtags[:, None], atol=0.001), axis=2)
            if torch.any(seenit): continue
            picktag = _pick_symelemtags(symtags.to('cpu').numpy(), symelems)
         else:
            symtags = _make_symtags(tag, f4cel)
            seenit = np.all(np.isclose(keep[None], symtags[:, None], atol=0.001), axis=2)
            if np.any(seenit): continue
            picktag = _pick_symelemtags(symtags, symelems)

         # picktag = None
         if picktag is None:
            keep = np.concatenate([keep, tag[None]])
         else:
            keep = np.concatenate([keep, picktag[None]])

      # ic((keep * 1000).astype('i'))
      for tag in keep:
         try:
            se = SymElem(nfold, tag[:3], tag[3:6], hel=tag[6])
            seenit = symelems[se.label].copy()
            if screw and se.label[:2] in symelems:
               seenit += symelems[se.label[:2]]
            if not any([_symelem_is_same(se, se2, f4cel) for se2 in seenit]):
               symelems[se.label].append(se)
         except ScrewError:
            continue

      # ic(symelems)
   symelems = _symelem_remove_ambiguous_syms(symelems)

   # move back to unitcell postions, identical for cubic groups
   # 'nonsingular' lattice used to avoid cases where symelems only appear
   # in a particular lattice configuration
   newelems = defaultdict(list)
   for psym, elems in symelems.items():
      for e in elems:
         e2 = e.tounit(lattice)
         newelems[psym].append(e2)
         assert e2.tolattice(lattice) == e
   symelems = newelems

   if aslist:
      symelems = list(itertools.chain(*symelems.values()))
   return symelems

def _find_compound_symelems(sym, se=None, frames=None):
   if se is None: se = wu.sym.symelems(sym, asdict=False, screws=False)
   se = [e for e in se if e.iscyclic]
   if frames is None: frames = wu.sym.sgframes(sym, cells=3)
   isects = defaultdict(set)
   for e1, e2 in it.product(se, se):
      # if e1.id == e2.id: continue
      axis, cen = e1.axis, e1.cen
      symcen = einsum('fij,j->fi', frames, e2.cen)
      symcen = symcen
      symaxis = einsum('fij,j->fi', frames, e2.axis)
      taxis, tcen = [np.tile(x, (len(symcen), 1)) for x in (axis, cen)]
      p, q = line_line_closest_points_pa(tcen, taxis, symcen, symaxis)
      d = hnorm(p - q)
      p = (p + q) / 2
      ok = _inunit(p)
      ok = np.logical_and(ok, d < 0.001)
      if np.sum(ok) == 0: continue
      axis2 = symaxis[ok][0]
      cen = p[ok][0]
      axis = einsum('fij,j->fi', frames, axis)
      axis2 = einsum('fij,j->fi', frames, axis2)
      cen = einsum('fij,j->fi', frames, cen)
      axis = axis[_inunit(cen)]
      axis2 = axis2[_inunit(cen)]
      cen = cen[_inunit(cen)]
      # ic(cen)
      pick = np.argmin(hnorm(cen - [0.003, 0.002, 0.001, 1]))
      axis = _flipaxs(axis[pick])
      axis2 = _flipaxs(axis2[pick])
      nf1, nf2 = e1.nfold, e2.nfold
      # D np.pi/2
      ang = angle(axis, axis2)
      if e2.nfold > e1.nfold:
         nf1, nf2 = e2.nfold, e1.nfold
         axis, axis2 = axis2, axis
      if np.isclose(ang, np.pi / 2):
         psym = f'D{nf1}'
      elif (nf1, nf2) == (3, 2) and np.isclose(ang, 0.9553166181245092):
         psym = 'T'
      elif any([
         (nf1, nf2) == (3, 2) and np.isclose(ang, 0.6154797086703874),
         (nf1, nf2) == (4, 3) and np.isclose(ang, 0.9553166181245092),
         (nf1, nf2) == (4, 2) and np.isclose(ang, 0.7853981633974484),
      ]):
         psym = 'O'
      # elif (nf1, nf2) in [(2, 2), (3, 3)]:
      # continue
      else:
         # print('SKIP', nf1, nf2, ang, flush=True)
         continue
      cen = cen[pick]
      t = tuple([f'{psym}{nf1}{nf2}', *cen[:3].round(9), *axis[:3].round(9), *axis2[:3].round(9)])
      isects[psym].add(t)

   for psym in isects:
      isects[psym] = list(sorted(isects[psym]))

   # ic(isects['D2'])
   # remove redundant centers
   for psym in isects:
      isects[psym] = list({t[1:4]: t for t in isects[psym]}.values())

   # ic(isects['D2'])
   # assert 0
   compound = defaultdict(list)
   for psym in isects:
      compound[psym] = [SymElem(t[0], t[4:7], t[1:4], t[7:10]) for t in isects[psym]]
   newd2 = list()
   for ed2 in compound['D2']:
      if not np.any([np.allclose(ed2.cen, eto.cen) for eto in it.chain(compound['T'], compound['O'], compound['D4'])]):
         newd2.append(ed2)
   compound['D2'] = newd2
   newd4 = list()
   for ed4 in compound['D4']:
      if not np.any([np.allclose(ed4.cen, eo.cen) for eo in compound['O']]):
         newd4.append(ed4)
   compound['D4'] = newd4
   compound = {k: v for k, v in compound.items() if len(v)}

   # newcompound = dict()
   # for psym, elems in compound.items():
   #    newcompound[psym] = [_to_central_symelem(frames, e, [0.6, 0.6, 0.6]) for e in elems]
   # compound = newcompound

   return compound

# def _to_central_symelem(frames, elem, cen):
#    ic(elem)
#    ic(elem.matching_frames(frames))
#    ic(elem.matching_frames(wu.sym.sgframes('I432', cells=2)))
#    # wu.showme(elem, scale=10)
#    # wu.showme(elem.operators, scale=10)
#    # wu.showme(frames, scale=10)
#    if not elem.isdihedral: return elem
#    ic(len(frames))
#    for i, f in enumerate(frames):
#       cen = einsum('ij,j->i', f, elem.cen)
#       # if not _inunit(cen): continue
#       elem2 = SymElem(
#          elem.nfold,
#          einsum('ij,j->i', f, elem.axis),
#          cen,
#          einsum('ij,j->i', f, elem.axis2),
#       )

#       try:
#          m = np.max(elem2.matching_frames(frames))
#       except AssertionError:
#          pass
#       if 48 * 8 > m:
#          ic(elem)
#          assert 0
#    assert 0
#    return

#    # ic(elem)
#    cens = einsum('fij,j->fi', frames, elem.cen)
#    ipick = np.argmin(hnorm(cens - cen))
#    frame = frames[ipick]
#    elem = SymElem(
#       elem.nfold,
#       einsum('ij,j->i', frame, elem.axis),
#       einsum('ij,j->i', frame, elem.cen),
#       einsum('ij,j->i', frame, elem.axis2),
#    )
#    return elem

def _symelem_is_same(elem, elem2, frames):
   assert elem.iscyclic or elem.isscrew
   assert elem2.iscyclic or elem2.isscrew
   assert elem.label[:2] == elem2.label[:2]
   axis = einsum('fij,j->fi', frames, elem2.axis)
   axsame = np.all(np.isclose(axis, elem.axis), axis=1)
   axsameneg = np.all(np.isclose(-axis, elem.axis), axis=1)
   axok = np.logical_or(axsame, axsameneg)
   if not np.any(axok): return False
   frames = frames[axok]
   axis = axis[axok]
   cen = einsum('fij,j->fi', frames, elem2.cen)
   censame = np.all(np.isclose(cen, elem.cen), axis=1)
   # ic(censame.shape)
   if any(censame): return True
   # if any(censame):
   # ic(elem.axis, axis[censame])
   # ic(elem.cen, cen[censame])
   # ic(elem)
   # ic(elem2)
   # assert not any(censame)  # should have been filtered out already

   d = h_point_line_dist(elem.cen, cen, axis)
   return np.min(d) < 0.001

'''
ic| sym: 'I23'
    symelems: defaultdict(<class 'list'>,
                          {'C2': [SymElem(2, axis=[0.0, 0.0, 1.0], cen=[-0.0, -0.0, 0.0]),
                                  SymElem(2, axis=[0.0, 0.0, 1.0], cen=[0.5, -0.0, 0.0]),
                                  SymElem(2, axis=[0.0, 1.0, 0.0], cen=[0.5, 0.0, -0.0]),
                                  SymElem(2, axis=[0.0, 0.0, 1.0], cen=[0.5, 0.5, 0.0])],
                           'C21': [SymElem(2, axis=[0.0, 0.0, 1.0], cen=[0.25, 0.25, 0.0], hel=0.5)],
                           'C3': [SymElem(3, axis=[1.0, 1.0, 1.0], cen=[0.0, 0.0, 0.0])]})
ic| sym: 'P213'
    symelems: defaultdict(<class 'list'>,
                          {'C21': [SymElem(2, axis=[0.0, 0.0, 1.0], cen=[0.25, -0.0, 0.0], hel=0.5),
                                   SymElem(2, axis=[0.0, 1.0, 0.0], cen=[0.5, 0.0, 0.25], hel=0.5)],
                           'C3': [SymElem(3, axis=[1.0, 1.0, 1.0], cen=[0.0, 0.0, 0.0]),
                                  SymElem(3, axis=[1.0, -1.0, -1.0], cen=[0.0, 0.0, 0.5])]})
ic| sym: 'I213'
    symelems: defaultdict(<class 'list'>,
                          {'C2': [SymElem(2, axis=[0.0, 1.0, 0.0], cen=[0.25, 0.0, -0.0]),
                                  SymElem(2, axis=[0.0, 1.0, 0.0], cen=[0.75, 0.0, -0.0])],
                           'C21': [SymElem(2, axis=[0.0, 0.0, 1.0], cen=[0.25, -0.0, 0.0], hel=0.5),
                                   SymElem(2, axis=[0.0, 0.0, -1.0], cen=[0.75, 0.0, 0.0], hel=0.5)],
                           'C3': [SymElem(3, axis=[1.0, 1.0, 1.0], cen=[0.0, 0.0, 0.0])]})
ic| sym: 'P432'
    symelems: defaultdict(<class 'list'>,
                          {'C2': [SymElem(2, axis=[0.0, 1.0, 1.0], cen=[0.0, 0.0, 0.0]),
                                  SymElem(2, axis=[0.0, 0.0, 1.0], cen=[0.5, -0.0, 0.0]),
                                  SymElem(2, axis=[0.0, 1.0, 1.0], cen=[0.5, 0.0, -0.0])],
                           'C3': [SymElem(3, axis=[1.0, 1.0, 1.0], cen=[0.0, 0.0, 0.0])],
                           'C4': [SymElem(4, axis=[-0.0, -0.0, 1.0], cen=[0.0, 0.0, 0.0]),
                                  SymElem(4, axis=[0.0, 0.0, 1.0], cen=[0.5, 0.5, 0.0])]})
ic| 'PASS test_spacegroup_symelems'

'''

def _make_symtags(tag, frames):
   concat = np.concatenate
   tax, tcen, thel = hvec(tag[:3]), hpoint(tag[3:6]), np.tile(tag[6], [len(frames), 1])

   # if is 21, 42, 63 screw, allow reverse axis with same helical shift
   c1 = (frames @ tax)[:, :3]
   c2 = (frames @ tcen)[:, :3]
   if np.any(np.isclose(thel, [0.5, np.sqrt(2) / 2])):
      # if is 21, 42, 63 screw, allow reverse axis with same helical shift
      symtags = concat([
         concat([c1, c2, +thel], axis=1),
         concat([-c1, c2, -thel], axis=1),
         concat([-c1, c2, +thel], axis=1),
      ])
   else:
      symtags = concat([
         concat([c1, c2, +thel], axis=1),
         concat([-c1, c2, -thel], axis=1),
      ])

   return symtags

def _make_symtags_torch(tag, frames, torch_device, t):
   import torch
   tag = torch.tensor(tag, device=torch_device).to(torch.float32)
   # frames = torch.tensor(frames, device=torch_device).to(torch.float32)
   concat = torch.cat
   tax, tcen, thel = wu.th_vec(tag[:3]), wu.th_point(tag[3:6]), torch.tile(tag[6], [len(frames), 1])

   # concat = np.concatenate
   # tax, tcen, thel = hvec(tag[:3]), hpoint(tag[3:6]), np.tile(tag[6], [len(frames), 1])

   const1 = torch.tensor(0.5, device=torch_device).to(torch.float32)
   const2 = torch.tensor(np.sqrt(2.) / 2., device=torch_device).to(torch.float32)
   c1 = (frames @ tax)[:, :3]
   c2 = (frames @ tcen)[:, :3]
   if torch.any(torch.isclose(thel, const1)) or torch.any(torch.isclose(thel, const2)):
      # if is 21, 42, 63 screw, allow reverse axis with same helical shift
      symtags = concat([
         concat([c1, c2, +thel], axis=1),
         concat([-c1, c2, -thel], axis=1),
         concat([-c1, c2, +thel], axis=1),
      ])
   else:
      symtags = concat([
         concat([c1, c2, +thel], axis=1),
         concat([-c1, c2, -thel], axis=1),
      ])
   return symtags

def _pick_symelemtags(symtags, symelems):

   # assert 0, 'this is incorrect somehow'

   # for i in [0, 1, 2]:
   #    symtags = symtags[np.argsort(-symtags[:, i], kind='stable')]
   # for i in [6, 5, 4, 3]:
   #    symtags = symtags[symtags[:, i] > -0.0001]
   #    symtags = symtags[symtags[:, i] < +0.9999]
   # for i in [5, 4, 3]:
   #    symtags = symtags[np.argsort(symtags[:, i], kind='stable')]
   # if len(symtags) == 0: return None
   # # ic(symtags)

   cen = [se.cen[:3] for psym in symelems for se in symelems[psym]]
   if cen and len(symtags):
      w = np.where(np.all(np.isclose(symtags[:, None, 3:6], np.stack(cen)[None]), axis=2))[0]
      if len(w) > 0:
         return symtags[w[0]]
   return None

   # d = np.sum(symtags[:, 3:6]**2, axis=1).round(6)
   # symtags = symtags[np.argsort(d, kind='stable')]
   # # ic(symtags[0])
   # return symtags[0]

def _symelem_remove_ambiguous_syms(symelems):
   symelems = symelems.copy()
   for sym1, sym2 in [('C2', 'C4'), ('C3', 'C6')]:
      if sym2 in symelems:
         newc2 = list()
         for s2 in symelems[sym1]:
            for s in symelems[sym2]:
               if np.allclose(s.axis, s2.axis):
                  if h_point_line_dist(s.cen, s2.cen, s.axis) < 0.001:
                     break
                  # if np.allclose(s.cen, s2.cen): break

            else:
               newc2.append(s2)
         symelems[sym1] = newc2
   return symelems
