from willutil import Bunch
from willutil.homog.hgeom import *
from willutil.sym.symframes import *
from willutil.sym.unbounded import *
from willutil.sym.asufit import *
from willutil.viz import showme

def frames(
   sym,
   axis=None,
   axis0=None,
   bbsym=None,
   asym_of=None,
   asym_index=0,
   sortframes=True,
   com=None,
   symops=None,
   spacing=None,
):
   '''generate symmetrical coordinate frames
    axis aligns Cx or bbaxis or axis0 to this
    bbsym removes redundant building block frames, e.g. TET with c3 bbs has 4 frames 
    asym_of removes redundant frames wrt a point group, e.g. turn TET into C3 and get asym unit of that C3
    '''

   if spacing is not None or symops is not None:
      return frames_unbounded(sym=sym, axis=axis, axis0=axis0, symops=symops, spacing=spacing, com=com)

   if sym is None or sym.upper() == 'C1':
      return np.eye(4).reshape(1, 4, 4)
   sym = map_sym_abbreviation(sym)
   sym = sym.lower()
   f = sym_frames[sym.lower()].copy()

   if asym_of:
      assert asym_of.startswith('c')
      dupaxis = axes(sym, asym_of)
      dupnfold = int(asym_of[1:])
      arbitrary_step_vector = hnormalized([100, 10, 1, 0])
      arbitrary_delta = np.array([0.0001, 0.0002, 0.0003, 0])
      symx = hrot(dupaxis, 2 * np.pi * asym_index / dupnfold)
      arbitrary_step_vector = hxform(symx, arbitrary_step_vector)
      arbitrary_delta = hxform(symx, arbitrary_delta)
      angs = np.arange(dupnfold) / dupnfold * 2 * np.pi
      dups = hrot(dupaxis, angs)  #.reshape(-1, 4, 4)
      f2 = dups[None, :] @ f[:, None]
      x = hdot(f2, dupaxis + arbitrary_delta)
      tgtdir = hcross(arbitrary_step_vector, dupaxis)
      dot = hdot(tgtdir, x)
      order = np.argsort(-dot, axis=-1)
      assert np.sum(order[:, 0] == 0) == len(f) / dupnfold
      f = f[order[:, 0] == 0]

   if bbsym:
      assert asym_of is None or bbsym == asym_of
      if not bbsym.lower().startswith('c'):
         raise ValueError(f'bad bblock sym {bbsym}')
      bbnfold = int(bbsym[1:])
      # bbaxes = axes(sym, bbnfold, all=True)
      bbaxes = symaxes_all[sym][bbnfold].copy()
      partial_ok = asym_of is not None
      f = remove_if_same_axis(f, bbaxes, partial_ok=partial_ok)

   if axis is not None:
      if axis0 is not None: startax = axis0
      elif sym.startswith('c'): startax = Uz
      elif bbsym: startax = axes(sym, bbnfold)
      elif asym_of: startax = axes(sym, asym_of)
      else: raise ValueError(f'dont know what to align to axis={axis}')
      # print(startax)
      # print(axis)
      # showme(f @ htrans(10 * f[0, :, 2]), name='a')
      f = halign(startax, axis) @ f
      # showme(f @ htrans(10 * f[0, :, 2]), name='b')
      # assert 0

   if sortframes:
      csym = bbsym or asym_of
      if csym:
         if axis is None: axis = axes(sym, csym)
         # order = np.argsort(-hdot(axis, hdot(f, axes(sym, csym))))
         # com = None
         ref = axes(sym, csym) if com is None else com
         order = np.argsort(-hdot(axis, hxform(f, ref)))
         # print(f.shape, order)
         f = f[order]
         # print(order)
         # assert 0

   return f

def map_sym_abbreviation(sym):
   if sym == 'I': return 'icos'
   if sym == 'O': return 'oct'
   if sym == 'T': return 'tet'
   return sym

def symaxis_angle(sym, nf1, nf2):
   return wu.hangle(axes(sym, nf1), axes(sym, nf2))

def symaxis_radbias(sym, nf1, nf2):
   return 1 / np.arctan(wu.hangle(axes(sym, nf1), axes(sym, nf2)))

def min_symaxis_angle(sym):
   symaxes = axes(sym)
   minaxsang = 9e9
   for i, iax in symaxes.items():
      for j, jax in symaxes.items():
         if i != j:
            minaxsang = min(minaxsang, line_angle(iax, jax))
            # print(i, j, line_angle_degrees(iax, jax))
   return minaxsang

def axes(sym, nfold=None, all=False):
   if nfold is None:
      return symaxes[sym].copy()
   elif isinstance(nfold, str):
      assert nfold.lower().startswith('c')
      nfold = int(nfold[1:])
   sym = sym.lower()
   if all:
      return symaxes_all[sym][nfold].copy()
   return symaxes[sym][nfold].copy()

def remove_if_same_axis(frames, bbaxes, onesided=True, partial_ok=False):
   assert onesided
   axes = hxform(frames, bbaxes[0])
   dots = hdot(bbaxes, axes, outerprod=True)

   uniq = list()
   whichaxis = list()
   for i, dot in enumerate(dots):
      w = np.where(np.logical_and(0.99999 < np.abs(dot), np.abs(dot) < 1.00001))[0]
      assert len(w) == 1
      w = w[0]
      if not np.any(np.isclose(dots[:i, w], dot[w], atol=0.00001)):
         whichaxis.append(w)
         uniq.append(i)
   whichaxis = np.array(whichaxis)
   # should be same num of bblocks on axis, (1 or 2)
   whichpartial = list()
   for i in range(len(bbaxes)):
      n = np.sum(whichaxis == i)
      # print(i, n)
      if not partial_ok:
         assert n == np.sum(whichaxis == 0)
      if n == 2:
         a, b = np.where(whichaxis == i)[0]
         assert np.allclose(axes[uniq[a]], -axes[uniq[b]], atol=1e-6)
      elif n != 1:
         if not partial_ok:
            assert 0

   uniq = np.array(uniq)
   return frames[uniq]

_ambiguous_axes = Bunch(tet=[], oct=[(2, 4)], icos=[], d2=[], _strict=True)

def ambiguous_axes(sym):
   return _ambiguous_axes[sym]

_ = -1

tetrahedral_axes = {
   2: hnormalized([1, 0, 0]),
   3: hnormalized([1, 1, 1]),
   '3b': hnormalized([1, 1, _])  # other c3
}
octahedral_axes = {2: hnormalized([1, 1, 0]), 3: hnormalized([1, 1, 1]), 4: hnormalized([1, 0, 0])}
icosahedral_axes = {
   2: hnormalized([1, 0, 0]),
   3: hnormalized([0.934172, 0.000000, 0.356822]),
   5: hnormalized([0.850651, 0.525731, 0.000000])
}

tetrahedral_axes_all = {
   2: hnormalized([
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
      # [_, 0, 0],
      # [0, _, 0],
      # [0, 0, _],
   ]),
   3: hnormalized([
      [1, 1, 1],
      [1, _, _],
      [_, _, 1],
      [_, 1, _],
      # [_, _, _],
      # [_, 1, 1],
      # [1, 1, _],
      # [1, _, 1],
   ]),
   '3b': hnormalized([
      [_, 1, 1],
      [1, _, 1],
      [1, 1, _],
      [_, _, -1],
   ]),
}
octahedral_axes_all = {
   2:
   hnormalized([
      [1, 1, 0],
      [0, 1, 1],
      [1, 0, 1],
      [_, 1, 0],
      [0, _, 1],
      [_, 0, 1],
      # [1, _, 0],
      # [0, 1, _],
      # [1, 0, _],
      # [_, _, 0],
      # [0, _, _],
      # [_, 0, _],
   ]),
   3:
   hnormalized([
      [1, 1, 1],
      [_, 1, 1],
      [1, _, 1],
      [1, 1, _],
      # [_, 1, _],
      # [_, _, 1],
      # [1, _, _],
      # [_, _, _],
   ]),
   4:
   hnormalized([
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
      # [_, 0, 0],
      # [0, _, 0],
      # [0, 0, _],
   ]),
}

def _icosahedral_axes_all():
   a2 = icosahedral_frames @ icosahedral_axes[2]
   a3 = icosahedral_frames @ icosahedral_axes[3]
   a5 = icosahedral_frames @ icosahedral_axes[5]

   # six decimals enough to account for numerical errors
   a2 = a2[np.unique(np.around(a2, decimals=6), axis=0, return_index=True)[1]]
   a3 = a3[np.unique(np.around(a3, decimals=6), axis=0, return_index=True)[1]]
   a5 = a5[np.unique(np.around(a5, decimals=6), axis=0, return_index=True)[1]]

   a2 = np.stack([a for i, a in enumerate(a2) if np.all(np.sum(a * a2[:i], axis=-1) > -0.999)])
   a3 = np.stack([a for i, a in enumerate(a3) if np.all(np.sum(a * a3[:i], axis=-1) > -0.999)])
   a5 = np.stack([a for i, a in enumerate(a5) if np.all(np.sum(a * a5[:i], axis=-1) > -0.999)])

   assert len(a2) == 15  # 30
   assert len(a3) == 10  # 20
   assert len(a5) == 6  #12
   icosahedral_axes_all = {
      2: hnormalized(a2),
      3: hnormalized(a3),
      5: hnormalized(a5),
   }
   return icosahedral_axes_all

icosahedral_axes_all = _icosahedral_axes_all()

def _d_axes(nfold):
   return {2: hnormalized([1, 0, 0]), nfold: hnormalized([0, 0, 1])}

def _d_frames(nfold):
   cx = hrot([0, 0, 1], np.pi * 2 / nfold)
   c2 = hrot([1, 0, 0], np.pi)
   frames = list()
   for ix in range(nfold):
      rot2 = hrot([0, 0, 1], np.pi * 2 * ix / nfold)
      for i2 in range(2):
         rot1 = [np.eye(4), c2][i2]
         frames.append(rot1 @ rot2)
   return np.array(frames)

def _d_axes_all(nfold):
   ang = 2 * np.pi / nfold
   frames = _d_frames(nfold)
   a2A = frames @ [1, 0, 0, 0]
   anA = frames @ [0, 0, 1, 0]
   if nfold % 2 == 0:
      a2A = np.concatenate([a2A, frames @ [np.cos(ang / 2), np.sin(ang / 2), 0, 0]])

   # six decimals enough to account for numerical errors
   a2B = a2A[np.unique(np.around(a2A, decimals=6), axis=0, return_index=True)[1]]
   anB = anA[np.unique(np.around(anA, decimals=6), axis=0, return_index=True)[1]]
   a2B = np.flip(a2B, axis=0)

   a2 = np.stack([a for i, a in enumerate(a2B) if np.all(np.sum(a * a2B[:i], axis=-1) > -0.999)])
   an = np.stack([a for i, a in enumerate(anB) if np.all(np.sum(a * anB[:i], axis=-1) > -0.999)])

   # if nfold == 4:
   #     print(np.around(a2A, decimals=3))
   #     print()
   #     print(np.around(a2B, decimals=3))
   #     print()
   #     print(np.around(a2, decimals=3))
   #     print()

   assert len(an) == 1, f'nfold {nfold}'
   assert len(a2) == nfold, f'nfold {nfold}'

   axes_all = {
      2: hnormalized(a2),
      nfold: hnormalized(an),
   }
   return axes_all

symaxes = dict(
   tet=tetrahedral_axes,
   oct=octahedral_axes,
   icos=icosahedral_axes,
)

symaxes_all = dict(
   tet=tetrahedral_axes_all,
   oct=octahedral_axes_all,
   icos=icosahedral_axes_all,
)

tetrahedral_angles = {(i, j): angle(
   tetrahedral_axes[i],
   tetrahedral_axes[j],
) for i, j in [
   (2, 3),
   (3, '3b'),
]}
octahedral_angles = {(i, j): angle(
   octahedral_axes[i],
   octahedral_axes[j],
) for i, j in [
   (2, 3),
   (2, 4),
   (3, 4),
]}
icosahedral_angles = {(i, j): angle(
   icosahedral_axes[i],
   icosahedral_axes[j],
) for i, j in [
   (2, 3),
   (2, 5),
   (3, 5),
]}
nfold_axis_angles = dict(
   tet=tetrahedral_angles,
   oct=octahedral_angles,
   icos=icosahedral_angles,
)
sym_point_angles = dict(tet={
   2: [np.pi],
   3: [np.pi * 2 / 3]
}, oct={
   2: [np.pi],
   3: [np.pi * 2 / 3],
   4: [np.pi / 2]
}, icos={
   2: [np.pi],
   3: [np.pi * 2 / 3],
   5: [np.pi * 2 / 5, np.pi * 4 / 5]
}, d3={
   2: [np.pi],
   3: [np.pi * 2 / 3],
})

sym_frames = dict(
   tet=tetrahedral_frames,
   oct=octahedral_frames,
   icos=icosahedral_frames,
)
minsymang = dict(
   tet=angle(tetrahedral_axes[2], tetrahedral_axes[3]) / 2,
   oct=angle(octahedral_axes[2], octahedral_axes[3]) / 2,
   icos=angle(icosahedral_axes[2], icosahedral_axes[3]) / 2,
   d2=np.pi / 4,
)
for icyc in range(3, 33):
   sym = 'd%i' % icyc
   symaxes[sym] = _d_axes(icyc)
   sym_frames[sym] = _d_frames(icyc)
   ceil = int(np.ceil(icyc / 2))
   sym_point_angles[sym] = {
      2: [np.pi],
      icyc: [np.pi * 2 * j / icyc for j in range(1, ceil)],
   }
   minsymang[sym] = np.pi / icyc / 2
   symaxes_all[sym] = _d_axes_all(icyc)
   _ambiguous_axes[sym] = list() if icyc % 2 else [(2, icyc)]

sym_frames['d2'] = np.stack([
   np.eye(4),
   hrot([1, 0, 0], np.pi),
   hrot([0, 1, 0], np.pi),
   hrot([0, 0, 1], np.pi),
])

symaxes['d2'] = {
   '2a': np.array([1, 0, 0, 0]),
   '2b': np.array([0, 1, 0, 0]),
   '2c': np.array([0, 0, 1, 0]),
}
symaxes_all['d2'] = {
   2: np.array([
      np.array([1, 0, 0, 0]),
      np.array([0, 1, 0, 0]),
      np.array([0, 0, 1, 0]),
   ])
}

sym_point_angles['d2'] = {2: [np.pi]}

def sym_nfold_map(nfold):
   if isinstance(nfold, str):
      return int(nfold[:-1])
   return nfold

def get_syminfo(sym):
   sym = sym.lower()
   try:
      ambig = ambiguous_axes(sym)
      nfoldmap = {k: sym_nfold_map(k) for k in symaxes[sym]}
      assert sym_frames[sym].shape[-2:] == (4, 4)
      return Bunch(
         frames=sym_frames[sym],
         axes=symaxes[sym],
         axesall=symaxes_all[sym],
         point_angles=sym_point_angles[sym],
         ambiguous_axes=ambig,
         nfoldmap=nfoldmap,
      )

   except KeyError as e:
      # raise ValueError(f'sim.py: dont know symmetry "{sym}"')
      print(f'sym.py: dont know symmetry "{sym}"')
      raise e

_sym_permute_axes_choices = dict(
   d2=np.array([
      np.eye(4),  #           x y z
      hrot([1, 0, 0], 90),  # x z y
      hrot([0, 0, 1], 90),  # y z x
      hrot([1, 0, 0], 90) @ hrot([0, 0, 1], 90),  # y x z
      hrot([0, 1, 0], 90),  # z y x
      hrot([1, 0, 0], 90) @ hrot([0, 1, 0], 90),  # z y x        
   ]),
   d3=np.array([
      np.eye(4),
      hrot([0, 0, 1], 180),
   ]),
)

def sym_permute_axes_choices(sym):
   if sym in _sym_permute_axes_choices:
      return _sym_permute_axes_choices[sym]
   else:
      return np.eye(4).reshape(1, 4, 4)

for icyc in range(2, 33):
   sym = 'c%i' % icyc
   symaxes[sym] = {icyc: np.array([0, 0, 1, 0])}
   angles = 2 * np.pi * np.arange(icyc) / icyc
   # print(angles * 180 / np.pi)
   sym_frames[sym] = hrot([0, 0, 1, 0], angles)
   sym_point_angles[sym] = {
      icyc: [angles],
   }
   minsymang[sym] = np.pi / icyc / 2
   symaxes_all[sym] = symaxes[sym]
   _ambiguous_axes[sym] = list()

def symunit_bounds(cagesym, cycsym):
   flb, fub, fnum = 1, -1, 2
   if cagesym == 'tet' and cycsym == 'c3':
      fub, fnum = None, 1
   return flb, fub, fnum
