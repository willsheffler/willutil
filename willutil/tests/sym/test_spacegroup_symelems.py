import numpy as np
import itertools as it
import willutil as wu
from willutil.sym.SymElem import SymElem
from willutil.sym.spacegroup_symelems import _compute_symelems
from numpy import array

def main():

   test_compound_elems()
   assert 0

   test_symelems_P23()

   test_screw_elem()

   test_symelems_I4132()

   # test_symelems_P4132()
   test_symelems_P4232()
   # test_symelems_P4332()

   test_symelems_I213()
   test_symelems_P213()

   test_symelems_F23()
   test_symelems_I23()

   test_symelems_P432()
   test_symelems_I432()
   test_symelems_F432()

   test_symelems_F4132()

   ic('PASS test_spacegroup_symelems')

def inunit(p):
   x, y, z, w = p.T
   ok = ((-0.001 < x) * (x < 0.999) * (-0.001 < y) * (y < 0.999) * (-0.001 < z) * (z < 0.999))
   return ok

def flipaxs(a):
   if np.sum(a[:3] * [1, 1.1, 1.2]) < 0:
      a[:3] *= -1
   return a

def test_compound_elems():
   # sym = 'P23'
   # sym = 'I23'
   sym = 'I4132'
   se = wu.sym.symelems(sym, asdict=True, screws=False)
   frames = wu.sym.sgframes(sym, cells=3)
   ic(se)
   # showsymelems(sym, se)
   isects = defaultdict(set)
   for e1, e2 in it.product(it.chain(*se.values()), it.chain(*se.values())):
      # if e1.id == e2.id: continue
      # e2 = se['C3'][0]
      axis, cen = e1.axis, e1.cen
      symcen = einsum('fij,j->fi', frames, e2.cen)
      symcen = symcen
      symaxis = einsum('fij,j->fi', frames, e2.axis)
      taxis, tcen = [np.tile(x, (len(symcen), 1)) for x in (axis, cen)]
      p, q = wu.hlinesisect(tcen, taxis, symcen, symaxis)
      d = wu.hnorm(p - q)
      p = (p + q) / 2
      ok = inunit(p)
      ok = np.logical_and(ok, d < 0.001)
      if np.sum(ok) == 0: continue
      axis2 = symaxis[ok][0]
      cen = p[ok][0]
      axis = einsum('fij,j->fi', frames, axis)
      axis2 = einsum('fij,j->fi', frames, axis2)
      cen = einsum('fij,j->fi', frames, cen)
      axis = axis[inunit(cen)]
      axis2 = axis2[inunit(cen)]
      cen = cen[inunit(cen)]
      pick = np.argmin(wu.hnorm(cen - [0.003, 0.002, 0.001, 1]))
      axis = flipaxs(axis[pick])
      axis2 = flipaxs(axis2[pick])
      if not np.isclose(wu.hangle(axis, axis2), np.pi / 2):
         ic('SKIPPING T or O')
         continue
      cen = cen[pick]
      nfold = e1.nfold
      if e2.nfold > e1.nfold:
         nfold = e2.nfold
         axis, axis2 = axis2, axis
      t = tuple([nfold, *cen[:3].round(9), *axis[:3].round(9), *axis2[:3].round(9)])
      isects[f'D{nfold}'].add(t)
   # ic(isects)

   # ic(isects['D2'])
   # remove redundant D2 centers
   isects['D2'] = list({t[1:4]: t for t in isects['D2']}.values())
   # ic(isects['D2'])
   # assert 0
   delems = defaultdict(list)
   for psym in isects:
      delems[psym] = [SymElem(t[0], t[4:7], t[1:4], t[7:10]) for t in isects[psym]]
      for e in delems[psym]:
         print(e)
   # delems['D3'] = list()
   # showsymelems(sym, se, scan=10)
   showsymelems(sym, delems)

   # p = einsum('fij,j->fi', frames, [0.25, 0.125, 0.0, 1])
   # wu.showme(p * 10)
   # p = einsum('fij,j->fi', frames, [0.5, 0.25, 0.375, 1])
   # wu.showme(p * 10)
   # p = einsum('fij,j->fi', frames, [0.125, 0.125, 0.125, 1])
   # wu.showme(p * 10)
   # p = einsum('fij,j->fi', frames, [0.375, 0.375, 0.375, 1])
   # wu.showme(p * 10)

   # showsymelems(sym, se)
   assert 0

def test_screw_elem():
   a = SymElem(4, axis=[0.0, 1.0, 0.0], cen=[0.5, 0.0, 0.25], hel=0.75)
   b = SymElem(4, axis=[0.0, -1.0, 0.0], cen=[0.5, 0.0, 0.25], hel=0.25)
   assert a == b
   assert np.allclose(b.axis, [0, -1, 0, 0])
   assert np.allclose(b.cen, [0.5, 0, 0.25, 1])
   assert b.hel == 0.25

def test_symelems_I4132(showme=False, **kw):
   sym = 'I4132'
   symelems = _compute_symelems(sym, wu.sym.sgframes(sym, cellgeom='unit'))
   ic(sym, symelems)
   if showme: showsymelems(sym, symelems, allframes=True, **kw)

   assert symelems == {
      'C2': [
         SymElem(2, axis=[0.0, 1.0, 0.0], cen=[0.25, 0.0, -0.0]),
         SymElem(2, axis=[-0.0, 1.0, -1.0], cen=[0.125, 0.0, 0.25]),
         SymElem(2, axis=[-0.0, 1.0, -1.0], cen=[0.375, 0.875, -0.125]),
      ],
      'C21': [
         SymElem(2, axis=[0.0, 0.0, 1.0], cen=[0.25, -0.0, 0.0], hel=0.5),
         SymElem(2, axis=[0.0, 1.0, 0.0], cen=[0.5, 0.0, 0.25], hel=0.5),
      ],
      'C3': [
         SymElem(3, axis=[1.0, 1.0, 1.0], cen=[0.0, 0.0, 0.0]),
      ],
      'C41': [
         SymElem(4, axis=[0.0, 0.0, -1.0], cen=[0.25, 0.0, 0.0], hel=0.25),
         SymElem(4, axis=[0.0, -1.0, 0.0], cen=[0.5, 0.0, 0.25], hel=0.25),
      ],
   }

def test_symelems_P4232(showme=False, **kw):
   sym = 'P4232'
   symelems = _compute_symelems(sym, wu.sym.sgframes(sym, cellgeom='unit'))
   ic(sym, symelems)
   if showme: showsymelems(sym, symelems, allframes=True, **kw)
   assert symelems == {
      'C2': [
         SymElem(2, axis=[0.0, 0.0, 1.0], cen=[-0.0, -0.0, 0.0]),
         SymElem(2, axis=[0.0, 0.0, 1.0], cen=[0.5, -0.0, 0.0]),
         SymElem(2, axis=[0.0, 1.0, 0.0], cen=[0.5, 0.0, -0.0]),
         SymElem(2, axis=[-0.0, 1.0, -1.0], cen=[0.25, 0.0, 0.5]),
         SymElem(2, axis=[-0.0, 1.0, -1.0], cen=[0.75, 0.0, 0.5]),
      ],
      'C3': [SymElem(3, axis=[1.0, 1.0, 1.0], cen=[0.0, 0.0, 0.0])],
      'C42': [
         SymElem(4, axis=[0.0, 0.0, 1.0], cen=[0.5, 0.0, 0.0], hel=-0.5),
         SymElem(4, axis=[0.0, 1.0, 0.0], cen=[0.5, 0.0, 0.0], hel=0.5),
      ],
   }

def test_symelems_I432(showme=False, **kw):
   sym = 'I432'
   symelems = _compute_symelems(sym, wu.sym.sgframes(sym, cellgeom='unit'))
   ic(sym, symelems)
   if showme: showsymelems(sym, symelems, allframes=True, **kw)
   assert symelems == {
      'C2': [
         SymElem(2, axis=[0.0, 1.0, 1.0], cen=[0.0, 0.0, 0.0]),
         SymElem(2, axis=[0.0, 0.0, 1.0], cen=[0.5, -0.0, 0.0]),
         SymElem(2, axis=[-0.0, 1.0, -1.0], cen=[0.25, 0.0, 0.5]),
      ],
      'C21': [SymElem(2, axis=[0.0, 0.0, 1.0], cen=[0.25, 0.25, 0.0], hel=0.5)],
      'C3': [SymElem(3, axis=[1.0, 1.0, 1.0], cen=[0.0, 0.0, 0.0])],
      'C4': [SymElem(4, axis=[-0.0, -0.0, 1.0], cen=[0.0, 0.0, 0.0])],
      'C42': [SymElem(4, axis=[0.0, 0.0, 1.0], cen=[0.5, 0.0, 0.0], hel=-0.5)],
   }

def test_symelems_F432(showme=False, **kw):
   sym = 'F432'
   symelems = _compute_symelems(sym, wu.sym.sgframes(sym, cellgeom='unit'))
   ic(sym, symelems)
   if showme: showsymelems(sym, symelems, allframes=True, **kw)
   assert symelems == {
      'C2': [
         SymElem(2, axis=[0.0, 1.0, 1.0], cen=[0.0, 0.0, 0.0]),
         SymElem(2, axis=[0.0, 0.0, 1.0], cen=[0.25, 0.25, 0.0]),
         SymElem(2, axis=[0.0, 1.0, 1.0], cen=[0.5, 0.0, -0.0]),
      ],
      'C21': [
         SymElem(2, axis=[0.0, 0.0, 1.0], cen=[0.25, -0.0, 0.0], hel=0.5),
         SymElem(2, axis=[1.0, 0.0, 1.0], cen=[1.125, 0.25, 0.875], hel=0.3535533906),
      ],
      'C3': [SymElem(3, axis=[1.0, 1.0, 1.0], cen=[0.0, 0.0, 0.0])],
      'C4': [SymElem(4, axis=[0.0, 0.0, 1.0], cen=[0.0, 0.0, 0.0])],
      'C42': [SymElem(4, axis=[0.0, 0.0, 1.0], cen=[0.25, 0.25, 0.0], hel=-0.5)],
   }

def test_symelems_F4132(showme=False, **kw):
   sym = 'F4132'
   symelems = _compute_symelems(sym, wu.sym.sgframes(sym, cellgeom='unit'))
   ic(sym, symelems)
   if showme: showsymelems(sym, symelems, allframes=True, **kw)
   assert symelems == {
      'C2': [
         SymElem(2, axis=[0.0, 0.0, 1.0], cen=[0.0, 0.0, 0.0]),
         SymElem(2, axis=[0.0, 1.0, -1.0], cen=[0.125, 0.0, 0.25]),
      ],
      'C21': [
         SymElem(2, axis=[0.0, 1.0, 1.0], cen=[0.125, 0.0, 0.0], hel=0.3535533906),
         SymElem(2, axis=[0.0, 0.0, 1.0], cen=[0.25, 0.0, 0.0], hel=0.5),
         SymElem(2, axis=[0.0, 1.0, 0.0], cen=[0.25, 0.0, 0.0], hel=0.5),
         SymElem(2, axis=[0.0, 1.0, 1.0], cen=[0.625, 0.0, 0.0], hel=0.3535533906),
      ],
      'C3': [
         SymElem(3, axis=[1.0, 1.0, 1.0], cen=[0.0, 0.0, 0.0]),
      ],
      'C41': [
         SymElem(4, axis=[0.0, 0.0, 1.0], cen=[0.25, 0.0, 0.0], hel=0.25),
         SymElem(4, axis=[0.0, 1.0, 0.0], cen=[0.25, 0.0, 0.0], hel=0.25),
      ],
   }

def test_symelems_P23(showme=False, **kw):
   sym = 'P23'
   symelems = _compute_symelems(sym, wu.sym.sgframes(sym, cellgeom='unit'))
   ic(sym, symelems)
   if showme: showsymelems(sym, symelems, colorbyelem=True, **kw)
   assert symelems == {
      'C2': [
         SymElem(2, axis=[0.0, 0.0, 1.0], cen=[-0.0, -0.0, 0.0]),
         SymElem(2, axis=[0.0, 0.0, 1.0], cen=[0.5, -0.0, 0.0]),
         SymElem(2, axis=[0.0, 1.0, 0.0], cen=[0.5, 0.0, -0.0]),
         SymElem(2, axis=[0.0, 0.0, 1.0], cen=[0.5, 0.5, 0.0]),
      ],
      'C3': [SymElem(3, axis=[1.0, 1.0, 1.0], cen=[0.0, 0.0, 0.0])],
   }

def test_symelems_F23(showme=False, **kw):
   sym = 'F23'
   symelems = _compute_symelems(sym, wu.sym.sgframes(sym, cellgeom='unit'))
   ic(sym, symelems)
   if showme: showsymelems(sym, symelems, colorbyelem=False, **kw)
   assert symelems == {
      'C2': [
         SymElem(2, axis=[0.0, 0.0, 1.0], cen=[-0.0, -0.0, 0.0]),
         SymElem(2, axis=[0.0, 0.0, 1.0], cen=[0.25, 0.25, 0.0]),
      ],
      'C21': [
         SymElem(2, axis=[0.0, 0.0, 1.0], cen=[0.25, -0.0, 0.0], hel=0.5),
         SymElem(2, axis=[0.0, 1.0, 0.0], cen=[0.25, 0.0, -0.0], hel=0.5),
      ],
      'C3': [
         SymElem(3, axis=[1.0, 1.0, 1.0], cen=[0.0, 0.0, 0.0]),
      ]
   }

def test_symelems_I23(showme=False, **kw):
   sym = 'I23'
   symelems = _compute_symelems(sym, wu.sym.sgframes(sym, cellgeom='unit'))
   ic(sym, symelems)
   if showme: showsymelems(sym, symelems, colorbyelem=False, **kw)
   assert symelems == {
      'C2': [
         SymElem(2, axis=[0.0, 0.0, 1.0], cen=[-0.0, -0.0, 0.0]),
         SymElem(2, axis=[0.0, 0.0, 1.0], cen=[0.5, -0.0, 0.0]),
      ],
      'C21': [
         SymElem(2, axis=[0.0, 0.0, 1.0], cen=[0.25, 0.25, 0.0], hel=0.5),
      ],
      'C3': [
         SymElem(3, axis=[1.0, 1.0, 1.0], cen=[0.0, 0.0, 0.0]),
      ]
   }

def test_symelems_P432(showme=False, **kw):
   sym = 'P432'
   symelems = _compute_symelems(sym, wu.sym.sgframes(sym, cellgeom='unit'))
   ic(sym, symelems)
   if showme: showsymelems(sym, symelems, **kw)
   symelems == {
      'C2': [
         SymElem(2, axis=[0.0, 1.0, 1.0], cen=[0.0, 0.0, 0.0]),
         SymElem(2, axis=[0.0, 0.0, 1.0], cen=[0.5, -0.0, 0.0]),
         SymElem(2, axis=[0.0, 1.0, 1.0], cen=[0.5, 0.0, -0.0]),
      ],
      'C3': [
         SymElem(3, axis=[1.0, 1.0, 1.0], cen=[0.0, 0.0, 0.0]),
      ],
      'C4': [
         SymElem(4, axis=[-0.0, -0.0, 1.0], cen=[0.0, 0.0, 0.0]),
         SymElem(4, axis=[0.0, 0.0, 1.0], cen=[0.5, 0.5, 0.0]),
      ],
   }

def test_symelems_I213(showme=False, **kw):
   sym = 'I213'
   symelems = _compute_symelems(sym, wu.sym.sgframes(sym, cellgeom='unit'))
   ic(sym, symelems)
   if showme: showsymelems(sym, symelems, **kw)
   assert symelems == {
      'C2': [
         SymElem(2, axis=[0.0, 1.0, 0.0], cen=[0.25, 0.0, -0.0]),
      ],
      'C21': [
         SymElem(2, axis=[0.0, 0.0, 1.0], cen=[0.25, -0.0, 0.0], hel=0.5),
         SymElem(2, axis=[0.0, 1.0, 0.0], cen=[0.5, 0.0, 0.25], hel=0.5),
      ],
      'C3': [
         SymElem(3, axis=[1.0, 1.0, 1.0], cen=[0.0, 0.0, 0.0]),
      ]
   }

def test_symelems_P213(showme=False, **kw):
   sym = 'P213'
   symelems = _compute_symelems(sym, wu.sym.sgframes(sym, cellgeom='unit'))
   ic(sym, symelems)
   if showme: showsymelems(sym, symelems, **kw)
   assert symelems == {
      'C21': [
         SymElem(2, axis=[0.0, 0.0, 1.0], cen=[0.25, -0.0, 0.0], hel=0.5),
         SymElem(2, axis=[0.0, 1.0, 0.0], cen=[0.5, 0.0, 0.25], hel=0.5),
      ],
      'C3': [
         SymElem(3, axis=[1.0, 1.0, 1.0], cen=[0.0, 0.0, 0.0]),
      ]
   }

def showsymelems(
   sym,
   symelems,
   allframes=True,
   colorbyelem=False,
   cells=3,
   bounds=[-0.1, 1.1],
   scale=10,
   offset=0.2,
   scan=0,
):
   f = np.eye(4).reshape(1, 4, 4)
   if allframes: f = wu.sym.sgframes(sym, cells=cells, cellgeom=[10])
   args = wu.Bunch(xyzlen=[0.3, 0.4, 1.0])

   ii = 0
   for i, c in enumerate(symelems):
      for j, s in enumerate(symelems[c]):
         if colorbyelem: args.colors = [[(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)][ii]]
         f2 = f
         if scan:
            f2 = f[:, None] @ wu.htrans(s.axis[None] * np.linspace(0, scale * np.sqrt(3), scan)[:, None])[None]
            ic(f2.shape)
            f2 = f2.reshape(-1, 4, 4)
            ic(f2.shape)
            # assert 0
         wu.showme(
            f2 @ wu.htrans(s.cen * scale + offset * wu.hvec([0.1, 0.2, 0.3])) @ wu.halign([0, 0, 1], s.axis),
            name=s.label,
            bounds=[b * 10 for b in bounds],
            **args,
         )
         ii += 1
   from willutil.viz.pymol_viz import showcube
   showcube(0, 10)

if __name__ == '__main__':
   main()
