import numpy as np
import willutil as wu
from willutil.sym.SymElem import SymElem
from willutil.sym.spacegroup_symelems import _compute_symelems, _find_compound_symelems
from numpy import array

def main():

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

def test_compound_elems_P4132(showme=False):
   sym = 'P4132'
   elems = wu.sym.symelems(sym, asdict=True)
   celems = _find_compound_symelems(sym)

   for k, v in celems.items():
      print(k)
      for x in v:
         print(x, flush=True)

   if showme: showsymelems(sym, elems)
   if showme: showsymelems(sym, celems)

   print(repr(celems), flush=True)
   assert celems == {'D3': [SymElem(3, axis=[1, 1, 1], axis2=[0.0, -1.0, 1.0], cen=[0.375, 0.375, 0.375], label='D3')]}

def test_compound_elems_P432(showme=False):
   sym = 'P432'
   elems = wu.sym.symelems(sym, asdict=True)
   celems = _find_compound_symelems(sym)

   for k, v in celems.items():
      print(k)
      for x in v:
         print(x, flush=True)

   if showme: showsymelems(sym, elems)
   if showme: showsymelems(sym, celems)

   print(repr(celems), flush=True)
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

   for k, v in celems.items():
      print(k)
      for x in v:
         print(x, flush=True)

   if showme: showsymelems(sym, elems)
   if showme: showsymelems(sym, celems)

   print(repr(celems), flush=True)
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

   for k, v in celems.items():
      print(k)
      for x in v:
         print(x, flush=True)

   if showme: showsymelems(sym, elems)
   if showme: showsymelems(sym, celems)

   print(repr(celems), flush=True)
   assert celems == {'T': [SymElem('T32', axis=[1, 1, 1], axis2=[0.0, 0.0, 1.0], cen=[0.0, 0.0, 0.0], label='T')], 'D3': [SymElem(3, axis=[1, 1, 1], axis2=[0.0, -1.0, 1.0], cen=[0.125, 0.125, 0.125], label='D3')]}

def test_compound_elems_F432(showme=False):
   sym = 'F432'
   elems = wu.sym.symelems(sym, asdict=True)
   celems = _find_compound_symelems(sym)

   for k, v in celems.items():
      print(k)
      for x in v:
         print(x, flush=True)

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

   for k, v in celems.items():
      print(k)
      for x in v:
         print(x, flush=True)

   # if showme: showsymelems(sym, elems)
   if showme: showsymelems(sym, celems)
   print(repr(celems), flush=True)
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

   for k, v in celems.items():
      print(k)
      for x in v:
         print(x, flush=True)

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

   for k, v in celems.items():
      print(k)
      for x in v:
         print(x, flush=True)

   # if showme: showsymelems(sym, elems)
   if showme: showsymelems(sym, celems)

   assert celems == {}

def test_compound_elems_I23(showme=False):
   sym = 'I23'
   elems = wu.sym.symelems(sym, asdict=True)
   celems = _find_compound_symelems(sym)

   for k, v in celems.items():
      print(k)
      for x in v:
         print(x, flush=True)

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

   for k, v in celems.items():
      print(k)
      for x in v:
         print(x, flush=True)

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
   scale=12,
   offset=0,
   weight=2.0,
   scan=0,
):
   import pymol
   f = np.eye(4).reshape(1, 4, 4)
   if allframes: f = wu.sym.sgframes(sym, cells=cells, cellgeom=[scale])

   ii = 0
   labelcount = defaultdict(lambda: 0)
   for i, c in enumerate(symelems):
      for j, s in enumerate(symelems[c]):
         if colorbyelem: args.colors = [[(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)][ii]]
         f2 = f
         if scan:
            f2 = f[:, None] @ wu.htrans(s.axis[None] * np.linspace(0, scale * np.sqrt(3), scan)[:, None])[None]
            ic(f2.shape)
            f2 = f2.reshape(-1, 4, 4)
            ic(f2.shape)

         shift = wu.htrans(s.cen * scale + offset * wu.hvec([0.1, 0.2, 0.3]))

         if s.istet:
            configs = [
               ((s.axis, [0, 1, 0]), (None, None), [0.0, 0.8, 0.0]),
               ((-s.axis, [0, 1, 0]), (None, None), [0.0, 0.8, 0.0]),
               ((s.axis2, [1, 0, 0]), (None, None), [0.8, 0.0, 0.0]),
            ]
         elif s.isoct:
            configs = [
               (([0, 1, 1], [1, 0, 0]), (None, None), [0.7, 0.0, 0.0]),
               (([1, 1, 1], [0, 1, 0]), (None, None), [0.0, 0.7, 0.0]),
               (([0, 0, 1], [0, 0, 1]), (None, None), [0.0, 0.0, 0.7]),
            ]
         elif s.label == 'D2':
            configs = [
               ((s.axis, [1, 0, 0]), (s.axis2, [0, 1, 0]), [0.7, 0, 0]),
               ((s.axis, [0, 1, 0]), (s.axis2, [0, 0, 1]), [0.7, 0, 0]),
               ((s.axis, [0, 0, 1]), (s.axis2, [1, 0, 0]), [0.7, 0, 0]),
            ]
         elif s.label == 'D4':
            configs = [
               ((s.axis2, [1, 0, 0]), (s.axis, [0, 1, 0]), [0.7, 0, 0]),
               ((wu.hrot(s.axis, 45, s.cen) @ s.axis2, [1, 0, 0]), (s.axis, [0, 1, 0]), [0.7, 0, 0]),
               ((s.axis, [0, 0, 1]), (s.axis2, [1, 0, 0]), [0.0, 0, 0.9]),
            ]
         elif s.nfold == 2:
            configs = [((s.axis, [1, 0, 0]), (s.axis2, [0, 0, 1]), [1.0, 0.3, 0.6])]
         elif s.nfold == 3:
            configs = [((s.axis, [0, 1, 0]), (s.axis2, [1, 0, 0]), [0.6, 1, 0.3])]
         elif s.nfold == 4:
            configs = [((s.axis, [0, 0, 1]), (s.axis2, [1, 0, 0]), [0.6, 0.3, 1])]
         elif s.nfold == 6:
            configs = [((s.axis, [1, 1, 1]), (s.axis2, [-1, 1, 0]), [1, 1, 1])]
         else:
            assert 0
         name = s.label + '_' + ('ABCDEFGH')[labelcount[s.label]]

         cgo = list()
         for (tax, ax), (tax2, ax2), xyzlen in configs:
            xyzlen = np.array(xyzlen)
            if s.isdihedral:
               origin = wu.halign2(ax, ax2, tax, tax2)
               xyzlen[xyzlen == 0.6] = 1
            else:
               origin = wu.halign(ax, tax)
            wu.showme(
               f2 @ shift @ origin,
               name=name,
               bounds=[b * scale for b in bounds],
               xyzlen=xyzlen,
               addtocgo=cgo,
               make_cgo_only=True,
               weight=weight,
               colorset=labelcount[s.label],
            )
         pymol.cmd.load_cgo(cgo, name)
         labelcount[s.label] += 1
         ii += 1
   from willutil.viz.pymol_viz import showcube
   showcube(0, scale)

if __name__ == '__main__':
   main()
