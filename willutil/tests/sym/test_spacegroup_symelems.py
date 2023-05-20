import itertools
import numpy as np
import willutil as wu
from willutil.sym.SymElem import SymElem
from willutil.sym.spacegroup_symelems import _compute_symelems, _find_compound_symelems
from numpy import array

def main():

   test_symelems_P3()

   assert 0

   test_compound_elems_P213()
   test_compound_elems_I23()
   test_compound_elems_F23()
   test_compound_elems_P23()
   test_compound_elems_P4132()
   test_compound_elems_P432()
   test_compound_elems_I432()
   test_compound_elems_F432()
   test_compound_elems_I4132()
   test_compound_elems_F4132()

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

def test_symelems_P3():

   sym = 'P3'
   frames = wu.sym.sgframes(sym, cellgeom='nonsingular', cells=3)

   ic(frames.shape)
   elems = _compute_symelems(sym)
   ic(elems)
   assert elems == {
      'C3': [
         SymElem(3, axis=[0, 0, 1], cen=[0, 0, 0.0], label='C3'),
         SymElem(3, axis=[0, 0, 1], cen=[2 / 3, 1 / 3, 0.0], label='C3'),
         SymElem(3, axis=[0, 0, 1], cen=[1 / 3, -1 / 3, 0.0], label='C3'),
      ]
   }

   # wu.showme(frames @ wu.htrans([0.01, 0.015, 0.02]), scale=10)
   # wu.showme(elems, scale=10)

   elems = list(itertools.chain(*elems.values()))
   celems = _find_compound_symelems(sym, elems)
   assert celems == {}

def test_compound_elems_P4132(showme=False):
   sym = 'P4132'
   elems = wu.sym.symelems(sym, asdict=True)
   celems = _find_compound_symelems(sym)

   # for k, v in celems.items():
   #    print(k)
   #    for x in v:
   #       print(x, flush=True)

   if showme: showsymelems(sym, elems)
   if showme: showsymelems(sym, celems)

   # print(repr(celems), flush=True)
   assert celems == {'D3': [SymElem(3, axis=[1, 1, 1], axis2=[0.0, -1.0, 1.0], cen=[0.375, 0.375, 0.375], label='D3')]}

def test_compound_elems_P432(showme=False):
   sym = 'P432'
   elems = wu.sym.symelems(sym, asdict=True)
   celems = _find_compound_symelems(sym)

   # for k, v in celems.items():
   #    print(k)
   #    for x in v:
   #       print(x, flush=True)

   if showme: showsymelems(sym, elems)
   if showme: showsymelems(sym, celems)

   # print(repr(celems), flush=True)
   assert celems == {
      'O': [
         SymElem('O43', axis=[0, 0, 1], axis2=[1.0, 1.0, 1.0], cen=[0.0, 0.0, 0.0], label='O'),
         SymElem('O43', axis=[0, 0, 1], axis2=[1.0, 1.0, 1.0], cen=[0.5, 0.5, 0.5], label='O'),
      ],
      'D4': [
         SymElem(4, axis=[1, 0, 0], axis2=[0.0, 1.0, 1.0], cen=[0.5, 0.0, 0.0], label='D4'),
         SymElem(4, axis=[0, 0, 1], axis2=[1.0, 0.0, 0.0], cen=[0.5, 0.5, 0.0], label='D4'),
      ],
   }

def test_compound_elems_I432(showme=False):
   sym = 'I432'
   elems = wu.sym.symelems(sym, asdict=True)
   celems = _find_compound_symelems(sym)

   # for k, v in celems.items():
   #    print(k)
   #    for x in v:
   #       print(x, flush=True)

   if showme: showsymelems(sym, elems)
   if showme: showsymelems(sym, celems)

   # print(repr(celems), flush=True)
   assert celems == {
      'O': [SymElem('O43', axis=[0, 0, 1], axis2=[1.0, 1.0, 1.0], cen=[0.0, 0.0, 0.0], label='O')],
      'D2': [SymElem(2, axis=[1, 0, 1], axis2=[0.0, 1.0, 0.0], cen=[0.5, 0.25, 0.0], label='D2')],
      'D4': [SymElem(4, axis=[1, 0, 0], axis2=[0.0, 0.0, 1.0], cen=[0.5, 0.0, 0.0], label='D4')],
      'D3': [SymElem(3, axis=[1, 1, 1], axis2=[0.0, -1.0, 1.0], cen=[0.25, 0.25, 0.25], label='D3')],
   }

def test_compound_elems_F4132(showme=False):
   sym = 'F4132'
   elems = wu.sym.symelems(sym, asdict=True)
   celems = _find_compound_symelems(sym)

   # for k, v in celems.items():
   #    print(k)
   #    for x in v:
   #       print(x, flush=True)

   if showme: showsymelems(sym, elems)
   if showme: showsymelems(sym, celems)

   # print(repr(celems), flush=True)
   assert celems == {'T': [SymElem('T32', axis=[1, 1, 1], axis2=[0.0, 0.0, 1.0], cen=[0.0, 0.0, 0.0], label='T')], 'D3': [SymElem(3, axis=[1, 1, 1], axis2=[0.0, -1.0, 1.0], cen=[0.125, 0.125, 0.125], label='D3')]}

def test_compound_elems_F432(showme=False):
   sym = 'F432'
   elems = wu.sym.symelems(sym, asdict=True)
   celems = _find_compound_symelems(sym)

   # for k, v in celems.items():
   #    print(k)
   #    for x in v:
   #       print(x, flush=True)

   if showme: showsymelems(sym, elems)
   if showme: showsymelems(sym, celems)
   # print(repr(celems), flush=True)
   assert celems == {
      'D2': [
         SymElem(2, axis=[1, 1, 0], axis2=[0.0, 0.0, 1.0], cen=[0.25, 0.25, 0.0], label='D2'),
      ],
      'O': [
         SymElem('O43', axis=[0, 0, 1], axis2=[1.0, 1.0, 1.0], cen=[0.0, 0.0, 0.0], label='O'),
         SymElem('O43', axis=[1, 1, 1], axis2=[1.0, 0.0, 1.0], cen=[0.5, 0.0, 0.0], label='O'),
      ],
      'T': [
         SymElem('T32', axis=[1, 1, 1], axis2=[0.0, 0.0, 1.0], cen=[0.25, 0.25, 0.25], label='T'),
      ],
   }

def test_compound_elems_I4132(showme=False):
   sym = 'I4132'
   elems = wu.sym.symelems(sym)
   celems = _find_compound_symelems(sym)

   # for k, v in celems.items():
   #    print(k)
   #    for x in v:
   #       print(x, flush=True)

   # if showme: showsymelems(sym, elems)
   if showme: showsymelems(sym, celems)
   # print(repr(celems), flush=True)
   assert celems == {
      'D2': [
         SymElem(2, axis=[1, 0, 1], axis2=[0.0, 1.0, 0.0], cen=[0.25, 0.125, 0.0], label='D2'),
         SymElem(2, axis=[0, 0, 1], axis2=[-1.0, 1.0, 0.0], cen=[0.5, 0.25, 0.375], label='D2'),
      ],
      'D3': [
         SymElem(3, axis=[1, 1, 1], axis2=[0.0, -1.0, 1.0], cen=[0.125, 0.125, 0.125], label='D3'),
         SymElem(3, axis=[1, 1, 1], axis2=[0.0, -1.0, 1.0], cen=[0.375, 0.375, 0.375], label='D3'),
      ],
   }

def test_compound_elems_P23(showme=False):
   sym = 'P23'
   elems = wu.sym.symelems(sym)
   celems = _find_compound_symelems(sym)

   # for k, v in celems.items():
   #    print(k)
   #    for x in v:
   #       print(x, flush=True)

   # if showme: showsymelems(sym, elems)
   if showme: showsymelems(sym, celems)

   assert celems == {
      'D2': [
         SymElem(2, axis=[1, 0, 0], axis2=[0.0, 1.0, 0.0], cen=[0.5, 0.0, 0.0]),
         SymElem(2, axis=[1, 0, 0], axis2=[0.0, 0.0, 1.0], cen=[0.5, 0.5, 0.0]),
      ],
      'T': [
         SymElem('T32', axis=[1, 1, 1], axis2=[0.0, 0.0, 1.0], cen=[0.0, 0.0, 0.0]),
         SymElem('T32', axis=[1, 1, 1], axis2=[0.0, 0.0, 1.0], cen=[0.5, 0.5, 0.5]),
      ],
   }

def test_compound_elems_P213(showme=False):
   sym = 'P213'
   elems = wu.sym.symelems(sym)
   celems = _find_compound_symelems(sym)

   # for k, v in celems.items():
   #    print(k)
   #    for x in v:
   #       print(x, flush=True)

   # if showme: showsymelems(sym, elems)
   if showme: showsymelems(sym, celems)

   assert celems == {}

def test_compound_elems_I23(showme=False):
   sym = 'I23'
   elems = wu.sym.symelems(sym, asdict=True)
   celems = _find_compound_symelems(sym)

   # for k, v in celems.items():
   #    print(k)
   #    for x in v:
   #       print(x, flush=True)

   if showme: showsymelems(sym, elems)
   if showme: showsymelems(sym, celems)

   print(repr(celems))
   assert celems == {
      'D2': [SymElem(2, axis=[1, 0, 0], axis2=[0.0, 0.0, 1.0], cen=[0.5, 0.0, 0.0], label='D2')],
      'T': [SymElem('T32', axis=[1, 1, 1], axis2=[0.0, 0.0, 1.0], cen=[0.0, 0.0, 0.0], label='T')],
   }

def test_compound_elems_F23(showme=False):
   sym = 'F23'
   elems = wu.sym.symelems(sym, asdict=True)
   celems = _find_compound_symelems(sym)

   # for k, v in celems.items():
   #    print(k)
   #    for x in v:
   #       print(x, flush=True)

   if showme: showsymelems(sym, elems)
   if showme: showsymelems(sym, celems)

   print(repr(celems))
   assert celems == {
      'T': [
         SymElem('T32', axis=[1, 1, 1], axis2=[0.0, 0.0, 1.0], cen=[0.0, 0.0, 0.0], label='T'),
         SymElem('T32', axis=[1, 1, 1], axis2=[0.0, 0.0, 1.0], cen=[0.25, 0.25, 0.25], label='T'),
      ]
   }

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
   # ic(sym, symelems)
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
   # ic(sym, symelems)
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
   # ic(sym, symelems)
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
   # ic(sym, symelems)
   if showme: showsymelems(sym, symelems, allframes=True, **kw)
   assert symelems == {
      'C2': [
         SymElem(2, axis=[0.0, 1.0, 1.0], cen=[0.0, 0.0, 0.0]),
         SymElem(2, axis=[0.0, 0.0, 1.0], cen=[0.25, 0.25, 0.0]),
         SymElem(2, axis=[0.0, 1.0, 1.0], cen=[0.5, 0.0, 0.0]),
      ],
      'C21': [
         SymElem(2, axis=[0.0, 0.0, 1.0], cen=[0.25, 0.0, 0.0], hel=0.5),
         SymElem(2, axis=[1.0, 0.0, 1.0], cen=[0.25, 0.25, 0.0], hel=0.3535533906),
      ],
      'C3': [SymElem(3, axis=[1.0, 1.0, 1.0], cen=[0.0, 0.0, 0.0])],
      'C4': [SymElem(4, axis=[0.0, 0.0, 1.0], cen=[0.0, 0.0, 0.0])],
      'C42': [SymElem(4, axis=[0.0, 0.0, 1.0], cen=[0.25, 0.25, 0.0], hel=-0.5)],
   }

def test_symelems_F4132(showme=False, **kw):
   sym = 'F4132'
   symelems = _compute_symelems(sym, wu.sym.sgframes(sym, cellgeom='unit'))
   # ic(sym, symelems)
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
   # ic(sym, symelems)
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
   # ic(sym, symelems)
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
   # ic(sym, symelems)
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
   # ic(sym, symelems)
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
   # ic(sym, symelems)
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
   # ic(sym, symelems)
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

if __name__ == '__main__':
   main()
