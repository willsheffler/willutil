import itertools
import numpy as np
import willutil as wu
from willutil.sym.SymElem import SymElem, showsymelems
from willutil.sym.spacegroup_symelems import _compute_symelems, _find_compound_symelems
from numpy import array

def main():

   test_symelems_R32()
   test_symelems_P1211()
   test_symelems_P212121()
   test_symelems_P2221()
   test_symelems_P21212()
   test_symelems_P1()
   test_symelems_C121()
   test_symelems_P3()
   test_symelems_P222()
   test_symelems_P23()
   test_symelems_I213()
   test_symelems_P213()
   test_symelems_F23()
   test_symelems_I23()
   test_symelems_P432()
   test_symelems_I432()
   test_symelems_F432()
   test_symelems_F4132()
   test_symelems_I4132()
   test_symelems_P4232()
   test_symelems_P4132()
   test_symelems_P4332()

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

   test_screw_elem()

   ic('PASS test_spacegroup_symelems')

def _printelems(sym, elems):
   print(sym)
   print(f'   assert set(elems.keys()) == set(\'{" ".join(elems.keys())}\'.split())')
   for k, v in elems.items():
      print(f'   assert elems[\'{k}\'] == [')
      for e in v:
         print('     ', e, ',')
      print('   ]', flush=True)

def test_symelems_R32():
   sym = 'R32'

   elems = _compute_symelems(sym)
   _printelems(sym, elems)
   # ic(elems)

   scale = 5
   latt = wu.sym.lattice_vectors(sym, cellgeom='nonsingular')
   # ic(latt)
   # WTF do these come from?
   # testelem = [
   # SymElem(2, axis=[0, 1, 0], cen=[0.333333337, 0.0, 0.283333336], label='C2').tounit(latt),
   # SymElem(2, axis=[0, 1, 0], cen=[0.666666669, 0.0, 0.566666668], label='C2').tounit(latt),
   # SymElem(2, axis=[0, 1, 0], cen=[0.333333336, 0.0, 1.133333336], label='C2').tounit(latt),
   # SymElem(2, axis=[0, 1, 0], cen=[0.666666669, 0.0, 1.416666668], label='C2').tounit(latt),
   # ]
   # showsymelems(sym, dict(TST=testelem), bounds=[-9e9, 9e9], scale=scale, cells=1, offset=0, scan=30)

   # showsymelems(sym, elems, bounds=[-9e9, 9e9], scale=scale, cells=1, offset=0, scan=30)
   # f = wu.sym.sgframes('R32', cells=3, cellgeom='nonsingular')
   # wu.showme(f @ wu.htrans([0.01, 0.02, 0.03]), scale=12)
   # wu.showme(f, scale=scale, name='frames')

   assert set(elems.keys()) == set('C2 C3 C21 C31 C32'.split())
   assert elems['C2'] == [
      SymElem(2, axis=[-0.57735, -1.0, 0.0], cen=[0.5, 0.5, 0], label='C2'),
      SymElem(2, axis=[-0.57735, 1.0, 0.0], cen=[0.333333333, 0.166666667, 0.166666667], label='C2'),
   ]
   assert elems['C3'] == [
      SymElem(3, axis=[0, 0, 1], cen=[-0, -0, 0.333333334], label='C3'),
   ]
   assert elems['C21'] == [
      SymElem(2, axis=[1, 0, 0], cen=[0.083333333, 0.166666667, 0.166666667], hel=0.5, label='C21'),
      SymElem(2, axis=[1, 0, 0], cen=[0.25, 0.5, 0.0], hel=0.5, label='C21'),
      SymElem(2, axis=[1, 0, 0], cen=[0.25, 0.5, 0.5], hel=0.5, label='C21'),
      SymElem(2, axis=[0, 1, 0], cen=[0.500000003, 0.0, 0.500000001], hel=0.5, label='C21'),
   ]
   assert elems['C31'] == [
      SymElem(3, axis=[0, 0, 1], cen=[0.333333334, 0., 0.0], hel=0.333333333, label='C31'),
      SymElem(3, axis=[0, 0, 1], cen=[0.333333333, 0.333333333, 0.0], hel=0.333333333, label='C31'),
   ]
   assert elems['C32'] == [
      SymElem(3, axis=[0, 0, 1], cen=[0.0, 0.333333333, 0.0], hel=0.666666667, label='C32'),
   ]

def test_symelems_P1():
   sym = 'P1'
   elems = _compute_symelems(sym)
   # _printelems(sym, elems)
   # ic(elems)
   assert set(elems.keys()) == set('C11'.split())
   assert elems == dict(C11=[
      SymElem(1, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], hel=1.0, label='C11'),
      SymElem(1, axis=[0, 1, 0], cen=[0.0, 0.0, 0.0], hel=1.0, label='C11'),
      SymElem(1, axis=[1, 0, 0], cen=[0.0, 0.0, 0.0], hel=1.0, label='C11'),
   ])

def test_symelems_P222():
   sym = 'P222'
   elems = _compute_symelems(sym)
   # _printelems(sym, elems)
   # ic(elems)
   assert set(elems.keys()) == set('C2'.split())
   assert elems['C2'] == [
      SymElem(2, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], label='C2'),
      SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.0, 0.0], label='C2'),
      SymElem(2, axis=[0, 1, 0], cen=[0.0, 0.0, 0.0], label='C2'),
      SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.0, 0.5], label='C2'),
      SymElem(2, axis=[0, 1, 0], cen=[0.0, 0.0, 0.5], label='C2'),
      SymElem(2, axis=[0, 0, 1], cen=[0.0, 0.5, 0.0], label='C2'),
      SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.5, 0.0], label='C2'),
      SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.5, 0.5], label='C2'),
      SymElem(2, axis=[0, 0, 1], cen=[0.5, 0.0, 0.0], label='C2'),
      SymElem(2, axis=[0, 1, 0], cen=[0.5, 0.0, 0.0], label='C2'),
      SymElem(2, axis=[0, 1, 0], cen=[0.5, 0.0, 0.5], label='C2'),
      SymElem(2, axis=[0, 0, 1], cen=[0.5, 0.5, 0.0], label='C2'),
   ]

def test_symelems_P2221():
   sym = 'P2221'
   elems = _compute_symelems(sym)
   # _printelems(sym, elems)
   assert set(elems.keys()) == set('C2 C21'.split())
   assert elems['C2'] == [
      SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.0, 0.0], label='C2'),
      SymElem(2, axis=[0, 1, 0], cen=[0.0, 0.0, 0.25], label='C2'),
      SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.5, 0.0], label='C2'),
      SymElem(2, axis=[0, 1, 0], cen=[0.5, 0.0, 0.25], label='C2'),
   ]
   assert elems['C21'] == [
      SymElem(2, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], hel=0.5, label='C21'),
      SymElem(2, axis=[0, 0, 1], cen=[0.5, 0.0, 0.0], hel=0.5, label='C21'),
      SymElem(2, axis=[0, 0, 1], cen=[0.0, 0.5, 0.0], hel=0.5, label='C21'),
      SymElem(2, axis=[0, 0, 1], cen=[0.5, 0.5, 0.0], hel=0.5, label='C21'),
   ]

def test_symelems_P21212():
   sym = 'P21212'
   elems = _compute_symelems(sym)
   _printelems(sym, elems)
   assert set(elems.keys()) == set('C2 C21 C11'.split())
   assert elems['C2'] == [
      SymElem(2, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], label='C2'),
      SymElem(2, axis=[0, 0, -1], cen=[0.0, 0.5, 0.0], label='C2'),
   ]
   assert elems['C21'] == [
      SymElem(2, axis=[0, 1, 0], cen=[0.25, 0.0, 0.0], hel=0.5, label='C21'),
      SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.25, 0.0], hel=0.5, label='C21'),
      SymElem(2, axis=[0, 1, 0], cen=[0.25, 0.0, 0.5], hel=0.5, label='C21'),
      SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.25, 0.5], hel=0.5, label='C21'),
   ]
   assert elems['C11'] == [
      SymElem(1, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], hel=1.0, label='C11'),
   ]

def test_symelems_P212121():
   sym = 'P212121'
   elems = _compute_symelems(sym)
   _printelems(sym, elems)
   assert set(elems.keys()) == set('C21'.split())
   assert elems['C21'] == [
      SymElem(2, axis=[0, 0, 1], cen=[0.25, 0.0, 0.0], hel=0.5, label='C21'),
      SymElem(2, axis=[0, 0, 1], cen=[0.75, 0.0, 0.0], hel=0.5, label='C21'),
      SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.25, 0.0], hel=0.5, label='C21'),
      SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.75, 0.0], hel=0.5, label='C21'),
      SymElem(2, axis=[0, 1, 0], cen=[0.0, 0.0, 0.25], hel=0.5, label='C21'),
      SymElem(2, axis=[0, 1, 0], cen=[0.0, 0.0, 0.75], hel=0.5, label='C21'),
   ]

def test_symelems_P1211():
   sym = 'P1211'
   ic(f'test_symelems_{sym}')
   elems = _compute_symelems(sym)
   _printelems(sym, elems)
   assert set(elems.keys()) == set('C11 C21'.split())
   assert elems['C11'] == [
      SymElem(1, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], hel=1.0, label='C11'),
      SymElem(1, axis=[1, 0, 0], cen=[0.0, 0.0, 0.0], hel=1.0, label='C11'),
   ]
   assert elems['C21'] == [
      SymElem(2, axis=[0, 1, 0], cen=[0.0, 0.0, 0.0], hel=0.5, label='C21'),
      SymElem(2, axis=[0, 1, 0], cen=[0.5, 0.0, 0.0], hel=0.5, label='C21'),
      SymElem(2, axis=[0, 1, 0], cen=[0.0, 0.0, 0.5], hel=0.5, label='C21'),
      SymElem(2, axis=[0, 1, 0], cen=[0.5, 0.0, 0.5], hel=0.5, label='C21'),
   ]

def test_symelems_C121():
   sym = 'C121'
   ic(f'test_symelems_{sym}')
   elems = _compute_symelems(sym)
   _printelems(sym, elems)
   assert set(elems.keys()) == set('C2 C21'.split())
   assert elems['C2'] == [
      SymElem(2, axis=[0, 1, 0], cen=[0.0, 0.0, 0.0], label='C2'),
      SymElem(2, axis=[0, 1, 0], cen=[0.0, 0.0, 0.5], label='C2'),
   ]
   assert elems['C21'] == [
      SymElem(2, axis=[0, 1, 0], cen=[0.25, 0.0, 0.0], hel=0.5, label='C21'),
      SymElem(2, axis=[0, 1, 0], cen=[0.25, 0.0, 0.5], hel=0.5, label='C21'),
   ]

def test_symelems_P3():
   ic('test_symelems_P3')
   sym = 'P3'
   elems = _compute_symelems(sym)
   _printelems(sym, elems)
   assert set(elems.keys()) == set('C11 C3'.split())
   assert elems['C11'] == [
      SymElem(1, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], hel=1.0, label='C11'),
   ]
   assert elems['C3'] == [
      SymElem(3, axis=[0, 0, 1], cen=[0, 0, 0.0], label='C3'),
      SymElem(3, axis=[0, 0, 1], cen=[2 / 3, 1 / 3, 0.0], label='C3'),
      SymElem(3, axis=[0, 0, 1], cen=[1 / 3, 2 / 3, 0.0], label='C3'),
   ]

   # wu.showme(frames @ wu.htrans([0.01, 0.015, 0.02]), scale=10)
   # wu.showme(elems, scale=10)

   elems = list(itertools.chain(*elems.values()))
   celems = _find_compound_symelems(sym, elems)
   assert celems == {}

def test_symelems_P4132():
   sym = 'P4132'
   elems = _compute_symelems(sym)
   _printelems(sym, elems)
   print(repr(elems))
   assert set(elems.keys()) == set('C2 C3 C21 C41'.split())
   assert elems['C2'] == [SymElem(2, axis=[1, -1, 0], cen=[0.0, 0.75, 0.375], label='C2')]
   assert elems['C3'] == [SymElem(3, axis=[1, 1, 1], cen=[0.0, 0.0, 0.0], label='C3')]
   assert elems['C21'] == [SymElem(2, axis=[0, 1, 0], cen=[0.5, 0.0, 0.25], hel=0.5, label='C21')]
   assert elems['C41'] == [SymElem(4, axis=[0, 0, -1], cen=[0.25, 0.0, 0.0], hel=0.25, label='C41')]

def test_symelems_P4332():
   sym = 'P4332'
   elems = _compute_symelems(sym)
   _printelems(sym, elems)
   print(repr(elems))
   assert set(elems.keys()) == set('C2 C3 C21 C41'.split())
   assert elems['C2'] == [SymElem(2, axis=[1, -1, 0], cen=[0.0, 0.25, 0.125], label='C2')]
   assert elems['C3'] == [SymElem(3, axis=[1, 1, 1], cen=[0.0, 0.0, 0.0], label='C3')]
   assert elems['C21'] == [SymElem(2, axis=[0, 0, 1], cen=[0.25, 0.0, 0.0], hel=0.5, label='C21')]
   assert elems['C41'] == [SymElem(4, axis=[0, -1, 0], cen=[0.5, 0.0, 0.25], hel=0.25, label='C41')]

def test_compound_elems_P4132(showme=False):
   ic('test_compound_elems_P4132')
   sym = 'P4132'
   elems = wu.sym.symelems(sym, asdict=True)
   celems = _find_compound_symelems(sym)
   if showme: showsymelems(sym, elems)
   if showme: showsymelems(sym, celems)
   # print(repr(celems), flush=True)
   assert set(elems.keys()) == set('D3'.split())
   assert celems['D3'] == [SymElem(3, axis=[1, 1, 1], axis2=[-1.0, 1.0, 0.0], cen=[0.375, 0.375, 0.375], label='D3')]

def test_compound_elems_P432(showme=False):
   ic('test_compound_elems_P432')
   sym = 'P432'
   elems = wu.sym.symelems(sym, asdict=True)
   celems = _find_compound_symelems(sym)

   # for k, v in celems.items():
   #    print(k)
   #    for x in v:
   #       print(x, flush=True)

   if showme: showsymelems(sym, elems)
   if showme: showsymelems(sym, celems)

   print(repr(celems), flush=True)
   assert set(elems.keys()) == set('O D4'.split())
   assert elems['O'] == [
      SymElem('O43', axis=[0, 0, 1], axis2=[1.0, 1.0, 1.0], cen=[0.0, 0.0, 0.0], label='O'),
      SymElem('O43', axis=[1, 0, 0], axis2=[1.0, 1.0, 1.0], cen=[0.5, 0.5, 0.5], label='O'),
   ]
   assert elems['D4'] == [
      SymElem(4, axis=[1, 0, 0], axis2=[0.0, 1.0, 0.0], cen=[0.5, 0.0, 0.0], label='D4'),
      SymElem(4, axis=[0, 0, 1], axis2=[1.0, 0.0, 0.0], cen=[0.5, 0.5, 0.0], label='D4'),
   ]

def test_compound_elems_I432(showme=False):
   ic('test_compound_elems_I432')
   sym = 'I432'
   elems = wu.sym.symelems(sym, asdict=True)
   celems = _find_compound_symelems(sym)

   # for k, v in celems.items():
   #    print(k)
   #    for x in v:
   #       print(x, flush=True)

   if showme: showsymelems(sym, elems)
   if showme: showsymelems(sym, celems)

   print(repr(celems), flush=True)
   assert set(elems.keys()) == set('O D2 D3 D4'.split())
   assert elems['O'] == [SymElem('O43', axis=[0, 0, 1], axis2=[1.0, 1.0, 1.0], cen=[0.0, 0.0, 0.0], label='O')]
   assert elems['D2'] == [SymElem(2, axis=[0, 1, 0], axis2=[-1.0, 0.0, 1.0], cen=[0.5, 0.25, 0.0], label='D2')]
   assert elems['D4'] == [SymElem(4, axis=[1, 0, 0], axis2=[0.0, 1.0, 0.0], cen=[0.5, 0.0, 0.0], label='D4')]
   assert elems['D3'] == [SymElem(3, axis=[1, 1, 1], axis2=[-1.0, 1.0, 0.0], cen=[0.25, 0.25, 0.25], label='D3')]

def test_compound_elems_F4132(showme=False):
   ic('test_compound_elems_F4132')
   sym = 'F4132'
   elems = wu.sym.symelems(sym, asdict=True)
   celems = _find_compound_symelems(sym)

   # for k, v in celems.items():
   #    print(k)
   #    for x in v:
   #       print(x, flush=True)

   if showme: showsymelems(sym, elems)
   if showme: showsymelems(sym, celems)
   assert set(elems.keys()) == set('T D3'.split())
   print(repr(celems), flush=True)
   assert elems['T'] == [SymElem('T32', axis=[1, 1, 1], axis2=[1.0, 0.0, 0.0], cen=[0.0, 0.0, 0.0], label='T')]
   assert elems['D3'] == [SymElem(3, axis=[1, 1, 1], axis2=[0.0, -1.0, 1.0], cen=[0.125, 0.125, 0.125], label='D3')]

def test_compound_elems_F432(showme=False):
   ic('test_compound_elems_F432')
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
   for k, v in celems.items():
      ic(k)
      for e in v:
         ic(e)
   assert set(elems.keys()) == set('O D2 T'.split())
   assert elems['D2'] == [
      SymElem(2, axis=[1, 1, 0], axis2=[0.0, 0.0, 1.0], cen=[0.25, 0.25, 0.0], label='D2'),
   ]
   assert elems['O'] == [
      SymElem('O43', axis=[0, 0, 1], axis2=[1.0, 1.0, 1.0], cen=[0.0, 0.0, 0.0], label='O'),
      SymElem('O43', axis=[1, -1, 1], axis2=[1.0, 0.0, 1.0], cen=[0.5, 0.0, 0.0], label='O'),
   ]
   assert elems['T'] == [
      SymElem('T32', axis=[1, 1, 1], axis2=[1.0, 0.0, 0.0], cen=[0.25, 0.25, 0.25], label='T'),
   ]

def test_compound_elems_I4132(showme=False):
   ic('test_compound_elems_I4132')
   sym = 'I4132'
   elems = _find_compound_symelems(sym)

   # for k, v in celems.items():
   #    print(k)
   #    for x in v:
   #       print(x, flush=True)

   # if showme: showsymelems(sym, elems)
   if showme: showsymelems(sym, celems)
   # print(repr(celems), flush=True)
   for k, v in celems.items():
      ic(k)
      for e in v:
         ic(e)
   assert set(elems.keys()) == set('D2 D3'.split())
   assert elems['D2'] == [
      SymElem(2, axis=[-1, 1, 0], axis2=[0.0, 0.0, 1.0], cen=[0.5, 0.25, 0.375], label='D2'),
      SymElem(2, axis=[1, 0, 1], axis2=[-1.0, 0.0, 1.0], cen=[0.25, 0.125, 0.0], label='D2'),
   ]
   assert elems['D3'] == [
      SymElem(3, axis=[1, 1, 1], axis2=[-1.0, 1.0, 0.0], cen=[0.125, 0.125, 0.125], label='D3'),
      SymElem(3, axis=[1, 1, 1], axis2=[-1.0, 1.0, 0.0], cen=[0.375, 0.375, 0.375], label='D3'),
   ]

def test_compound_elems_P23(showme=False):
   ic('test_compound_elems_P23')
   sym = 'P23'
   elems = _find_compound_symelems(sym)

   # for k, v in celems.items():
   #    print(k)
   #    for x in v:
   #       print(x, flush=True)

   # if showme: showsymelems(sym, elems)
   if showme: showsymelems(sym, celems)
   assert set(elems.keys()) == set('D2 T'.split())
   assert elems['D2'] == [
      SymElem(2, axis=[1, 0, 0], axis2=[0.0, 1.0, 0.0], cen=[0.5, 0.0, 0.0]),
      SymElem(2, axis=[1, 0, 0], axis2=[0.0, 0.0, 1.0], cen=[0.5, 0.5, 0.0]),
   ]
   assert elems['T'] == [
      SymElem('T32', axis=[1, 1, 1], axis2=[0.0, 0.0, 1.0], cen=[0.0, 0.0, 0.0]),
      SymElem('T32', axis=[1, 1, 1], axis2=[1.0, 0.0, 0.0], cen=[0.5, 0.5, 0.5]),
   ]

def test_compound_elems_P213(showme=False):
   ic('test_compound_elems_P213')
   sym = 'P213'
   elems = _find_compound_symelems(sym)

   # for k, v in celems.items():
   #    print(k)
   #    for x in v:
   #       print(x, flush=True)

   # if showme: showsymelems(sym, elems)
   if showme: showsymelems(sym, celems)

   assert elems == {}

def test_compound_elems_I23(showme=False):
   ic('test_compound_elems_I23')
   sym = 'I23'
   elems = wu.sym.symelems(sym, asdict=True)
   celems = _find_compound_symelems(sym)

   # for k, v in celems.items():
   #    print(k)
   #    for x in v:
   #       print(x, flush=True)

   if showme: showsymelems(sym, elems)
   if showme: showsymelems(sym, celems)

   print(repr(celems), flush=True)
   assert set(elems.keys()) == set('D2 T'.split())
   assert elems['D2'] == [SymElem(2, axis=[1, 0, 0], axis2=[0, 1, 0], cen=[0.5, 0.0, 0.0], label='D2')]
   assert elems['T'] == [SymElem('T32', axis=[1, 1, 1], axis2=[0.0, 0.0, 1.0], cen=[0.0, 0.0, 0.0], label='T')]

def test_compound_elems_F23(showme=False):
   ic('test_compound_elems_F23')
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
   assert set(elems.keys()) == set('T'.split())
   assert elems['T'] == [
      SymElem('T32', axis=[1, 1, 1], axis2=[0.0, 0.0, 1.0], cen=[0.0, 0.0, 0.0], label='T'),
      SymElem('T32', axis=[1, 1, 1], axis2=[1.0, 0.0, 0.0], cen=[0.25, 0.25, 0.25], label='T'),
   ]

def test_screw_elem():
   ic('test_screw_elem')
   a = SymElem(4, axis=[0.0, 1.0, 0.0], cen=[0.5, 0.0, 0.25], hel=0.75)
   b = SymElem(4, axis=[0.0, -1.0, 0.0], cen=[0.5, 0.0, 0.25], hel=0.25)
   assert a == b
   assert np.allclose(b.axis, [0, -1, 0, 0])
   assert np.allclose(b.cen, [0.5, 0, 0.25, 1])
   assert b.hel == 0.25

def test_symelems_I4132(showme=False, **kw):
   ic('test_symelems_I4132')
   sym = 'I4132'
   elems = _compute_symelems(sym)
   _printelems(sym, elems)
   # ic(sym, elems)
   if showme: showsymelems(sym, elems, allframes=True, **kw)
   assert set(elems.keys()) == set('C2 C3 C41'.split())
   assert elems['C2'] == [
      SymElem(2, axis=[-1, 0, 0], cen=[0.25, 0.0, 0.25], label='C2'),
      SymElem(2, axis=[1, -1, 0], cen=[0.0, 0.25, 0.125], label='C2'),
      SymElem(2, axis=[1, -1, 0], cen=[0.0, 0.75, 0.375], label='C2'),
   ]
   assert elems['C3'] == [
      SymElem(3, axis=[1.0, 1.0, 1.0], cen=[0.0, 0.0, 0.0]),
   ]
   assert elems['C41'] == [
      SymElem(4, axis=[0.0, 0.0, -1.0], cen=[0.25, 0.0, 0.0], hel=0.25),
      SymElem(4, axis=[0.0, -1.0, 0.0], cen=[0.5, 0.0, 0.25], hel=0.25),
   ]

def test_symelems_P4232(showme=False, **kw):
   ic('test_symelems_P4232')
   sym = 'P4232'
   elems = _compute_symelems(sym)
   # ic(sym, elems)
   if showme: showsymelems(sym, elems, allframes=True, **kw)
   assert set(elems.keys()) == set('C2 C3 C42'.split())
   assert elems['C2'] == [
      SymElem(2, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], label='C2'),
      SymElem(2, axis=[1, -1, 0], cen=[0.0, 0.5, 0.25], label='C2'),
      SymElem(2, axis=[1, 0, 0], cen=[0.5, 0.0, 0.5], label='C2'),
      SymElem(2, axis=[0, 1, 0], cen=[0.0, 0.5, 0.5], label='C2'),
      SymElem(2, axis=[1, -1, 0], cen=[0.0, 0.5, 0.75], label='C2'),
   ]
   assert elems['C3'] == [SymElem(3, axis=[1.0, 1.0, 1.0], cen=[0.0, 0.0, 0.0])]
   assert elems['C42'] == [
      SymElem(4, axis=[0.0, 0.0, 1.0], cen=[0.5, 0.0, 0.0], hel=-0.5),
      SymElem(4, axis=[0.0, 1.0, 0.0], cen=[0.5, 0.0, 0.0], hel=0.5),
   ]

def test_symelems_I432(showme=False, **kw):
   ic('test_symelems_I432')
   sym = 'I432'
   elems = _compute_symelems(sym)
   # ic(sym, elems)
   if showme: showsymelems(sym, elems, allframes=True, **kw)
   assert set(elems.keys()) == set('C2 C21 C3 C4 C42'.split())
   assert elems['C2'] == [
      SymElem(2, axis=[1, 1, 0], cen=[0.0, 0.0, 0.0], label='C2'),
      SymElem(2, axis=[1, -1, 0], cen=[0.0, 0.5, 0.25], label='C2'),
      SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.0, 0.5], label='C2'),
   ]
   assert elems['C21'] == [SymElem(2, axis=[0, 0, 1], cen=[0.25, 0.25, 0.0], hel=0.5, label='C21')]
   assert elems['C3'] == [SymElem(3, axis=[1, 1, 1], cen=[0.0, 0.0, 0.0], label='C3')]
   assert elems['C4'] == [SymElem(4, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], label='C4')]
   assert elems['C42'] == [SymElem(4, axis=[0, 0, 1], cen=[0.5, 0.0, 0.0], hel=0.5, label='C42')]

def test_symelems_F432(showme=False, **kw):
   ic('test_symelems_F432')
   sym = 'F432'
   elems = _compute_symelems(sym)
   # _printelems(sym, elems)
   if showme: showsymelems(sym, elems, allframes=True, **kw)
   assert set(elems.keys()) == set('C2 C3 C4 C42 C21'.split())
   assert elems['C2'] == [
      SymElem(2, axis=[1, 1, 0], cen=[0.0, 0.0, 0.0]),
      SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.25, 0.25]),
      SymElem(2, axis=[0, -1, 1], cen=[0.0, 0.0, 0.5]),
   ]
   assert elems['C3'] == [SymElem(3, axis=[1.0, 1.0, 1.0], cen=[0.0, 0.0, 0.0])]
   assert elems['C4'] == [SymElem(4, axis=[0.0, 0.0, 1.0], cen=[0.0, 0.0, 0.0])]
   assert elems['C42'] == [SymElem(4, axis=[0.0, 0.0, 1.0], cen=[0.25, 0.25, 0.0], hel=-0.5)]
   assert elems['C21'] == [
      SymElem(2, axis=[0.0, 0.0, 1.0], cen=[0.25, 0.0, 0.0], hel=0.5),
      SymElem(2, axis=[1.0, 0.0, 1.0], cen=[0.25, 0.25, 0.0], hel=0.353553391, screw=1),
   ]

def test_symelems_F4132(showme=False, **kw):
   ic('test_symelems_F4132')
   sym = 'F4132'
   elems = _compute_symelems(sym)
   # _printelems(sym, elems)
   if showme: showsymelems(sym, elems, allframes=True, **kw)
   assert set(elems.keys()) == set('C2 C21 C3 C41'.split())
   assert elems['C2'] == [
      SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.0, 0.0], label='C2'),
      SymElem(2, axis=[-1, 0, 1], cen=[0.25, 0.125, 0.0], label='C2'),
   ]
   assert elems['C21'] == [
      SymElem(2, axis=[0.0, 1.0, 1.0], cen=[0.125, 0.0, 0.0], hel=0.3535533906),
      SymElem(2, axis=[0.0, 1.0, 1.0], cen=[0.625, 0.0, 0.0], hel=0.3535533906),
   ]
   assert elems['C3'] == [
      SymElem(3, axis=[1.0, 1.0, 1.0], cen=[0.0, 0.0, 0.0]),
   ]
   assert elems['C41'] == [
      SymElem(4, axis=[0.0, 0.0, 1.0], cen=[0.25, 0.0, 0.0], hel=0.25),
      SymElem(4, axis=[0.0, 1.0, 0.0], cen=[0.25, 0.0, 0.0], hel=0.25),
   ]

def test_symelems_P23(showme=False, **kw):
   ic('test_symelems_P23')
   sym = 'P23'
   elems = _compute_symelems(sym)
   # ic(sym, elems)
   if showme: showsymelems(sym, elems, scale=10, cells=2, **kw)
   assert set(elems.keys()) == set('C2 C3'.split())
   assert elems['C2'] == [
      SymElem(2, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], label='C2'),
      SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.0, 0.5], label='C2'),
      SymElem(2, axis=[0, 1, 0], cen=[0.0, 0.0, 0.5], label='C2'),
      SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.5, 0.5], label='C2'),
   ]
   assert elems['C3'] == [
      SymElem(3, axis=[1, 1, 1], cen=[0.0, 0.0, 0.0], label='C3'),
   ]

def test_symelems_F23(showme=False, **kw):
   ic('test_symelems_F23')
   sym = 'F23'
   elems = _compute_symelems(sym)
   # ic(sym, elems)
   if showme: showsymelems(sym, elems, colorbyelem=False, **kw)
   assert set(elems.keys()) == set('C2 C21 C3'.split())
   assert elems['C2'] == [
      SymElem(2, axis=[0.0, 0.0, 1.0], cen=[-0.0, -0.0, 0.0]),
      SymElem(2, axis=[1.0, 0.0, 0.0], cen=[0, 0.25, 0.25]),
   ]
   assert elems['C21'] == [
      SymElem(2, axis=[0.0, 0.0, 1.0], cen=[0.25, -0.0, 0.0], hel=0.5),
      SymElem(2, axis=[0.0, 1.0, 0.0], cen=[0.25, 0.0, -0.0], hel=0.5),
   ]
   assert elems['C3'] == [
      SymElem(3, axis=[1.0, 1.0, 1.0], cen=[0.0, 0.0, 0.0]),
   ]

def test_symelems_I23(showme=False, **kw):
   ic('test_symelems_I23')
   sym = 'I23'
   elems = _compute_symelems(sym)
   # ic(sym, elems)
   if showme: showsymelems(sym, elems, colorbyelem=False, **kw)
   assert set(elems.keys()) == set('C2 C21 C3'.split())
   assert elems['C2'] == [
      SymElem(2, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0]),
      SymElem(2, axis=[-1, 0, 0], cen=[0.5, 0.0, 0.5]),
   ]
   assert elems['C21'] == [
      SymElem(2, axis=[0.0, 0.0, 1.0], cen=[0.25, 0.25, 0.0], hel=0.5),
   ]
   assert elems['C3'] == [
      SymElem(3, axis=[1.0, 1.0, 1.0], cen=[0.0, 0.0, 0.0]),
   ]

def test_symelems_P432(showme=False, **kw):
   ic('test_symelems_P432')
   sym = 'P432'
   elems = _compute_symelems(sym)
   if showme: showsymelems(sym, elems, **kw)
   _printelems(sym, elems)
   assert set(elems.keys()) == set('C2 C3 C4'.split())
   assert elems['C2'] == [
      SymElem(2, axis=[1, 1, 0], cen=[0.0, 0.0, 0.0]),
      SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.0, 0.5]),
      SymElem(2, axis=[1, 1, 0], cen=[0.0, 0.0, 0.5]),
   ]
   assert elems['C3'] == [
      SymElem(3, axis=[1.0, 1.0, 1.0], cen=[0.0, 0.0, 0.0]),
   ]
   assert elems['C4'] == [
      SymElem(4, axis=[0.0, 0.0, 1.0], cen=[0.0, 0.0, 0.0]),
      SymElem(4, axis=[1.0, 0.0, 0.0], cen=[0.0, 0.5, 0.5]),
   ]

def test_symelems_I213(showme=False, **kw):
   ic('test_symelems_I213')
   sym = 'I213'
   elems = _compute_symelems(sym)
   # ic(sym, elems)
   if showme: showsymelems(sym, elems, **kw)
   assert set(elems.keys()) == set('C2 C21 C3'.split())
   assert elems['C2'] == [
      SymElem(2, axis=[0, 0, 1], cen=[0.0, 0.25, 0.0]),
   ]
   assert elems['C21'] == [
      SymElem(2, axis=[0.0, 0.0, 1.0], cen=[0.25, -0.0, 0.0], hel=0.5),
      SymElem(2, axis=[0.0, 1.0, 0.0], cen=[0.5, 0.0, 0.25], hel=0.5),
   ]
   assert elems['C3'] == [
      SymElem(3, axis=[1.0, 1.0, 1.0], cen=[0.0, 0.0, 0.0]),
   ]

def test_symelems_P213(showme=False, **kw):
   ic('test_symelems_P213')
   sym = 'P213'
   elems = _compute_symelems(sym)
   # ic(sym, elems)
   if showme: showsymelems(sym, elems, **kw)
   assert set(elems.keys()) == set('C21 C3'.split())
   assert elems['C21'] == [
      SymElem(2, axis=[0.0, 0.0, 1.0], cen=[0.25, -0.0, 0.0], hel=0.5),
      SymElem(2, axis=[0.0, 1.0, 0.0], cen=[0.5, 0.0, 0.25], hel=0.5),
   ]
   assert elems['C3'] == [
      SymElem(3, axis=[1.0, 1.0, 1.0], cen=[0.0, 0.0, 0.0]),
   ]

if __name__ == '__main__':
   main()
