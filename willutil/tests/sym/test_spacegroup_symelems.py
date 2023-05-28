import itertools, pytest
import numpy as np
import willutil as wu
from willutil.sym.SymElem import SymElem, showsymelems
from willutil.sym.spacegroup_symelems import _compute_symelems, _find_compound_symelems, _remove_redundant_screws
from numpy import array
# yapf: disable

def main():

   test_symelems_R3()
   test_symelems_P3121()
   test_symelems_P212121()
   test_symelems_P31()
   test_symelems_P32()
   test_symelems_P213()
   test_symelems_P3221()
   test_symelems_P41()
   test_symelems_P41212()
   test_symelems_P4132()
   test_symelems_P4232()
   test_symelems_P43()
   test_symelems_P432()
   # test_symelems_P43212() # missing central C21 repeated in unitcell
   test_symelems_P4332()
   test_symelems_P6()
   test_symelems_P61()
   test_symelems_P6122()
   test_symelems_P62()
   test_symelems_P63()
   test_symelems_P64()
   test_symelems_P65()
   test_symelems_P6522()
   test_symelems_I213()
   test_symelems_I23()
   test_symelems_I4()
   test_symelems_I41()
   test_symelems_I4132()
   test_symelems_I432()
   test_symelems_F4132()
   test_symelems_F432()

   test_remove_redundant_screws()
   assert 0

   test_symelems_R32()
   test_symelems_P1211()
   test_symelems_P2221()
   test_symelems_P21212()
   test_symelems_P1()
   test_symelems_C121()
   test_symelems_P3()
   test_symelems_P222()
   test_symelems_P23()
   test_symelems_F23()

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

   ic('PASS test_spacegroup_symelems')

def helper_test_symelem(sym, val=None, debug=False, **kw):
   elems = _compute_symelems(sym, profile=debug)
   if elems != val:
      if val is not None:
         vkey = set(val.keys())
         tkey = set(elems.keys())
         key = sorted(vkey.intersection(tkey))
         for k in vkey - tkey:
            print('MISSING', k)
         for k in tkey - vkey:
            print('EXTRA', k)
         for k in key:
            tval = set(elems[k])
            vval = set(val[k])
            x = vval - tval
            if x:
               print(k, 'MISSING')
               for v in x:
                  print('  ', v)
            x = tval - vval
            if x:
               print(k, 'EXTRA')
               for v in x:
                  print('  ', v)
            x = vval.intersection(tval)
            if x:
               print(k, 'COMMON')
               for v in x:
                  print('  ', v)

      _printelems(sym, elems)
      showsymelems(sym, elems, scale=12, scan=12, offset=0, **kw)
      assert elems == val

def test_symelems_R3(debug=False, **kw):
   val = dict(
      C3=[
         SymElem(3, axis=[0, 0, 1], cen=[1e-09, 1e-09, 0.333333334], label='C3'),
      ],
      C32=[
         SymElem(3, axis=[0, 0, 1], cen=[0.333333334, 0.0, 0.0], hel=0.666666667, label='C32'),
      ],
      C31=[
         SymElem(3, axis=[0, 0, 1], cen=[0.333333334, 0.333333333, 0.0], hel=0.333333333, label='C31'),
      ],
   )
   helper_test_symelem('R3', val, debug, **kw)

def test_symelems_P3121(debug=False, **kw):
   val = dict(
      C2=[
         SymElem(2, axis=[0.57735, 1.0, 0.0], cen=[0.0, 0.0, 1e-09], label='C2'),
         SymElem(2, axis=[-0.57735, 1.0, 0.0], cen=[0.0, 0.0, 0.166666667], label='C2'),
      ],
      C21=[
         SymElem(2, axis=[1, 0, 0], cen=[0.25, 0.5, 0.333333333], hel=0.5, label='C21'),
         SymElem(2, axis=[1, 0, 0], cen=[0.25, 0.5, 0.833333333], hel=0.5, label='C21'),
      ],
      C31=[
         SymElem(3, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], hel=0.333333333, label='C31'),
         SymElem(3, axis=[0, 0, 1], cen=[0.666666666, 0.333333333, 0.0], hel=0.333333333, label='C31'),
      ],
   )
   helper_test_symelem('P3121', val, debug, **kw)

def test_symelems_P212121(debug=False, **kw):
   val = dict(C21=[
      SymElem(2, axis=[0, 0, 1], cen=[0.25, 0.0, 0.0], hel=0.5, label='C21'),
      SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.25, 0.0], hel=0.5, label='C21'),
      SymElem(2, axis=[0, 1, 0], cen=[0.0, 0.0, 0.25], hel=0.5, label='C21'),
      SymElem(2, axis=[0, 1, 0], cen=[0.5, 0.0, 0.25], hel=0.5, label='C21'),
      SymElem(2, axis=[0, 0, 1], cen=[0.25, 0.5, 0.0], hel=0.5, label='C21'),
      SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.25, 0.5], hel=0.5, label='C21'),
   ], )
   helper_test_symelem('P212121', val, debug, **kw)

def test_symelems_P31(debug=False, **kw):
   val = dict(
      C31=[
         SymElem(3, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], hel=0.333333333, label='C31'),
         SymElem(3, axis=[0, 0, 1], cen=[0.666666666, 0.333333333, 0.0], hel=0.333333333, label='C31'),
         SymElem(3, axis=[0, 0, 1], cen=[0.333333333, 0.666666666, 0.0], hel=0.333333333, label='C31'),
      ],
      C11=[
         SymElem(1, axis=[0, 1, 0], cen=[0.0, 0.0, 0.0], hel=1.0, label='C11'),
      ],
   )
   helper_test_symelem('P31', val, debug, **kw)

def test_symelems_P32(debug=False, **kw):
   val = dict(
      C32=[
         SymElem(3, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], hel=0.666666667, label='C32'),
         SymElem(3, axis=[0, 0, 1], cen=[0.666666666, 0.333333333, 0.0], hel=0.666666667, label='C32'),
         SymElem(3, axis=[0, 0, 1], cen=[0.333333333, 0.666666666, 0.0], hel=0.666666667, label='C32'),
      ],
      C11=[
         SymElem(1, axis=[0, 1, 0], cen=[0.0, 0.0, 0.0], hel=1.0, label='C11'),
      ],
   )
   helper_test_symelem('P32', val, debug, **kw)

def test_symelems_P213(debug=False, **kw):
   val = dict(
      C3=[
         SymElem(3, axis=[1, 1, 1], cen=[0.0, 0.0, 0.0], label='C3'),
      ],
      C21=[
         SymElem(2, axis=[0, 0, 1], cen=[0.25, 0.0, 0.0], hel=0.5, label='C21'),
         SymElem(2, axis=[0, 1, 0], cen=[0.5, 0.0, 0.25], hel=0.5, label='C21'),
      ],
      C31=[
         SymElem(3, axis=[-1, 1, 1], cen=[0.166666667, 0.166666667, 0.0], hel=0.577350269, label='C31'),
      ],
      C32=[
         SymElem(3, axis=[1, 1, 1], cen=[0.0, 0.333333333, 0.666666667], hel=1.154700538, label='C32'),
      ],
   )
   helper_test_symelem('P213', val, debug, **kw)

def test_symelems_P3221(debug=False, **kw):
   val = dict(
      C2=[
         SymElem(2, axis=[0.57735, 1.0, 0.0], cen=[0.0, 0.0, -1e-09], label='C2'),
         SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.0, 0.166666667], label='C2'),
      ],
      C21=[
         SymElem(2, axis=[1, 0, 0], cen=[0.25, 0.5, 0.166666667], hel=0.5, label='C21'),
         SymElem(2, axis=[1, 0, 0], cen=[0.25, 0.5, 0.666666667], hel=0.5, label='C21'),
      ],
      C32=[
         SymElem(3, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], hel=0.666666667, label='C32'),
         SymElem(3, axis=[0, 0, 1], cen=[0.666666666, 0.333333333, 0.0], hel=0.666666667, label='C32'),
      ],
   )
   helper_test_symelem('P3221', val, debug, **kw)

def test_symelems_P41(debug=False, **kw):
   val = dict(
      C21=[
         SymElem(2, axis=[0, 0, 1], cen=[0.5, 0.0, 0.0], hel=0.5, label='C21'),
      ],
      C41=[
         SymElem(4, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], hel=0.25, label='C41'),
         SymElem(4, axis=[0, 0, 1], cen=[0.5, 0.5, 0.0], hel=0.25, label='C41'),
      ],
      C11=[
         SymElem(1, axis=[0, 1, 0], cen=[0.0, 0.0, 0.0], hel=1.0, label='C11'),
      ],
   )
   helper_test_symelem('P41', val, debug, **kw)

def test_symelems_P41212(debug=False, **kw):
   val = dict(
      C2=[
         SymElem(2, axis=[-1, 1, 0], cen=[0.0, 0.0, 0.75], label='C2'),
      ],
      C21=[
         SymElem(2, axis=[1, 1, 0], cen=[0.0, 0.5, 0.5], hel=0.707106781, label='C21'),
      ],
      C41=[
         SymElem(4, axis=[0, 0, 1], cen=[0.0, 0.5, 0.0], hel=0.25, label='C41'),
      ],
   )
   helper_test_symelem('P41212', val, debug, **kw)

def test_symelems_P4132(debug=False, **kw):
   val = dict(
      C2=[
         SymElem(2, axis=[-1, 1, 0], cen=[0.0, 0.75, 0.375], label='C2'),
      ],
      C3=[
         SymElem(3, axis=[1, 1, 1], cen=[0.0, 0.0, 0.0], label='C3'),
      ],
      C21=[
         SymElem(2, axis=[0, 1, 0], cen=[0.5, 0.0, 0.25], hel=0.5, label='C21'),
         SymElem(2, axis=[0, 1, 1], cen=[0.125, 0.25, 0.0], hel=0.707106781, label='C21'),
      ],
      C31=[
         SymElem(3, axis=[-1, 1, 1], cen=[0.166666667, 0.166666667, 0.0], hel=0.577350269, label='C31'),
      ],
      C32=[
         SymElem(3, axis=[1, 1, 1], cen=[0.0, 0.333333333, 0.666666667], hel=1.154700538, label='C32'),
      ],
      C41=[
         SymElem(4, axis=[0, 0, 1], cen=[0.25, 0.0, 0.0], hel=0.25, label='C41'),
      ],
   )
   helper_test_symelem('P4132', val, debug, **kw)

def test_symelems_P4232(debug=False, **kw):
   val = dict(
      C2=[
         SymElem(2, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], label='C2'),
         SymElem(2, axis=[-1, 1, 0], cen=[0.0, 0.5, 0.25], label='C2'),
         SymElem(2, axis=[1, 0, 0], cen=[0.5, 0.0, 0.5], label='C2'),
         SymElem(2, axis=[0, 1, 0], cen=[0.0, 0.5, 0.5], label='C2'),
         SymElem(2, axis=[-1, 1, 0], cen=[0.0, 0.5, 0.75], label='C2'),
      ],
      C3=[
         SymElem(3, axis=[1, 1, 1], cen=[0.0, 0.0, 0.0], label='C3'),
      ],
      C21=[
         SymElem(2, axis=[0, 1, 1], cen=[0.25, 0.0, 0.0], hel=0.707106781, label='C21'),
      ],
      C31=[
         SymElem(3, axis=[1, -1, 1], cen=[0.333333333, 0.333333333, 0.0], hel=0.577350269, label='C31'),
      ],
      C32=[
         SymElem(3, axis=[1, 1, -1], cen=[0.333333333, 0.0, 0.333333333], hel=1.154700538, label='C32'),
      ],
      C42=[
         SymElem(4, axis=[0, 1, 0], cen=[0.5, 0.0, 0.0], hel=0.5, label='C42'),
      ],
   )
   helper_test_symelem('P4232', val, debug, **kw)

def test_symelems_P43(debug=False, **kw):
   val = dict(
      C21=[
         SymElem(2, axis=[0, 0, 1], cen=[0.5, 0.0, 0.0], hel=0.5, label='C21'),
      ],
      C43=[
         SymElem(4, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], hel=0.75, label='C43'),
         SymElem(4, axis=[0, 0, 1], cen=[0.5, 0.5, 0.0], hel=0.75, label='C43'),
      ],
      C11=[
         SymElem(1, axis=[0, 1, 0], cen=[0.0, 0.0, 0.0], hel=1.0, label='C11'),
      ],
   )
   helper_test_symelem('P43', val, debug, **kw)

def test_symelems_P432(debug=False, **kw):
   val = None
   helper_test_symelem('P432', val, debug, **kw)

def test_symelems_P43212(debug=False, **kw):
   val = None
   helper_test_symelem('P43212', val, debug, **kw)

def test_symelems_P4332(debug=False, **kw):
   val = dict(
      C2=[
         SymElem(2, axis=[-1, 1, 0], cen=[0.0, 0.25, 0.125], label='C2'),
      ],
      C3=[
         SymElem(3, axis=[1, 1, 1], cen=[0.0, 0.0, 0.0], label='C3'),
      ],
      C21=[
         SymElem(2, axis=[0, 0, 1], cen=[0.25, 0.0, 0.0], hel=0.5, label='C21'),
         SymElem(2, axis=[0, 1, 1], cen=[0.375, 0.0, 0.25], hel=0.707106781, label='C21'),
      ],
      C31=[
         SymElem(3, axis=[-1, 1, 1], cen=[0.166666667, 0.166666667, 0.0], hel=0.577350269, label='C31'),
      ],
      C32=[
         SymElem(3, axis=[1, 1, 1], cen=[0.0, 0.333333333, 0.666666667], hel=1.154700538, label='C32'),
      ],
      C43=[
         SymElem(4, axis=[0, 1, 0], cen=[0.5, 0.0, 0.25], hel=0.75, label='C43'),
      ],
   )
   helper_test_symelem('P4332', val, debug, **kw)

def test_symelems_P6(debug=False, **kw):
   val = dict(
      C2=[
         SymElem(2, axis=[0, 0, 1], cen=[0.0, 0.5, 0.0], label='C2'),
      ],
      C3=[
         SymElem(3, axis=[0, 0, 1], cen=[-0.333333332, 0.333333334, 0.0], label='C3'),
      ],
      C6=[
         SymElem(6, axis=[0, 0, 1], cen=[1e-09, 0.0, 0.0], label='C6'),
      ],
      C11=[
         SymElem(1, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], hel=1.0, label='C11'),
      ],
   )
   helper_test_symelem('P6', val, debug, **kw)

def test_symelems_P61(debug=False, **kw):
   val = dict(
      C21=[
         SymElem(2, axis=[0, 0, 1], cen=[0.5, 0.0, 0.0], hel=0.5, label='C21'),
      ],
      C31=[
         SymElem(3, axis=[0, 0, 1], cen=[0.666666667, 0.333333333, 0.0], hel=0.333333333, label='C31'),
      ],
      C61=[
         SymElem(6, axis=[0, 0, 1], cen=[1e-09, 0.0, 0.0], hel=0.166666667, label='C61'),
      ],
      C11=[
         SymElem(1, axis=[0, 1, 0], cen=[0.0, 0.0, 0.0], hel=1.0, label='C11'),
      ],
   )
   helper_test_symelem('P61', val, debug, **kw)

def test_symelems_P6122(debug=False, **kw):
   val = dict(
      C2=[
         SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.0, 0.0], label='C2'),
      ],
      C21=[
         SymElem(2, axis=[1, 0, 0], cen=[0.25, 0.5, 0.0], hel=0.5, label='C21'),
         SymElem(2, axis=[-1, 1, 0], cen=[0.166666665, 0.333333331, 0.416666666], hel=0.707106781, label='C21'),
      ],
      C31=[
         SymElem(3, axis=[0, 0, 1], cen=[0.666666667, 0.333333333, 0.0], hel=0.333333333, label='C31'),
      ],
      C61=[
         SymElem(6, axis=[0, 0, 1], cen=[1e-09, 0.0, 0.0], hel=0.166666667, label='C61'),
      ],
   )
   helper_test_symelem('P6122', val, debug, **kw)

def test_symelems_P62(debug=False, **kw):
   val = dict(
      C2=[
         SymElem(2, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], label='C2'),
         SymElem(2, axis=[0, 0, 1], cen=[0.0, 0.5, 0.0], label='C2'),
      ],
      C32=[
         SymElem(3, axis=[0, 0, 1], cen=[0.666666667, 0.333333333, 0.0], hel=0.666666667, label='C32'),
      ],
      C62=[
         SymElem(6, axis=[0, 0, 1], cen=[1e-09, 0.0, 0.0], hel=0.333333333, label='C62'),
      ],
   )
   helper_test_symelem('P62', val, debug, **kw)

def test_symelems_P63(debug=False, **kw):
   val = dict(
      C3=[
         SymElem(3, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], label='C3'),
         SymElem(3, axis=[0, 0, 1], cen=[-0.333333332, 0.333333334, 0.5], label='C3'),
      ],
      C21=[
         SymElem(2, axis=[0, 0, 1], cen=[0.5, 0.0, 0.0], hel=0.5, label='C21'),
      ],
      C63=[
         SymElem(6, axis=[0, 0, 1], cen=[1e-09, 0.0, 0.0], hel=0.5, label='C63'),
      ],
   )
   helper_test_symelem('P63', val, debug, **kw)

def test_symelems_P64(debug=False, **kw):
   val = dict(
      C2=[
         SymElem(2, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], label='C2'),
         SymElem(2, axis=[0, 0, 1], cen=[0.0, 0.5, 0.0], label='C2'),
      ],
      C31=[
         SymElem(3, axis=[0, 0, 1], cen=[0.666666667, 0.333333333, 0.0], hel=0.333333333, label='C31'),
      ],
      C64=[
         SymElem(6, axis=[0, 0, 1], cen=[1e-09, 0.0, 0.0], hel=0.666666667, label='C64'),
      ],
   )
   helper_test_symelem('P64', val, debug, **kw)

def test_symelems_P65(debug=False, **kw):
   val = dict(
      C21=[
         SymElem(2, axis=[0, 0, 1], cen=[0.5, 0.0, 0.0], hel=0.5, label='C21'),
      ],
      C32=[
         SymElem(3, axis=[0, 0, 1], cen=[0.666666667, 0.333333333, 0.0], hel=0.666666667, label='C32'),
      ],
      C65=[
         SymElem(6, axis=[0, 0, 1], cen=[1e-09, 0.0, 0.0], hel=0.833333333, label='C65'),
      ],
      C11=[
         SymElem(1, axis=[0, 1, 0], cen=[0.0, 0.0, 0.0], hel=1.0, label='C11'),
      ],
   )
   helper_test_symelem('P65', val, debug, **kw)

def test_symelems_P6522(debug=False, **kw):
   val = dict(
      C2=[
         SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.0, 0.0], label='C2'),
      ],
      C21=[
         SymElem(2, axis=[1, 0, 0], cen=[0.25, 0.5, 0.0], hel=0.5, label='C21'),
         SymElem(2, axis=[-1, 1, 0], cen=[0.166666666, 0.333333331, 0.083333333], hel=0.707106781, label='C21'),
      ],
      C32=[
         SymElem(3, axis=[0, 0, 1], cen=[0.666666667, 0.333333333, 0.0], hel=0.666666667, label='C32'),
      ],
      C65=[
         SymElem(6, axis=[0, 0, 1], cen=[1e-09, 0.0, 0.0], hel=0.833333333, label='C65'),
      ],
   )
   helper_test_symelem('P6522', val, debug, **kw)

def test_symelems_I213(debug=False, **kw):
   val = dict(
      C2=[
         SymElem(2, axis=[0, 0, 1], cen=[0.0, 0.25, 0.0], label='C2'),
      ],
      C3=[
         SymElem(3, axis=[1, 1, 1], cen=[0.0, 0.0, 0.0], label='C3'),
      ],
      C21=[
         SymElem(2, axis=[0, 0, 1], cen=[0.25, 0.0, 0.0], hel=0.5, label='C21'),
         SymElem(2, axis=[0, 1, 0], cen=[0.5, 0.0, 0.25], hel=0.5, label='C21'),
      ],
      C31=[
         SymElem(3, axis=[-1, 1, 1], cen=[0.166666667, 0.166666667, 0.0], hel=0.577350269, label='C31'),
      ],
      C32=[
         SymElem(3, axis=[1, 1, 1], cen=[0.0, 0.333333333, 0.666666667], hel=1.154700538, label='C32'),
      ],
   )
   helper_test_symelem('I213', val, debug, **kw)

def test_symelems_I23(debug=False, **kw):
   val = dict(
      C2=[
         SymElem(2, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], label='C2'),
         SymElem(2, axis=[1, 0, 0], cen=[0.5, 0.0, 0.5], label='C2'),
      ],
      C3=[
         SymElem(3, axis=[1, 1, 1], cen=[0.0, 0.0, 0.0], label='C3'),
      ],
      C21=[
         SymElem(2, axis=[0, 0, 1], cen=[0.25, 0.25, 0.0], hel=0.5, label='C21'),
      ],
      C31=[
         SymElem(3, axis=[1, -1, 1], cen=[0.333333333, 0.333333333, 0.0], hel=0.577350269, label='C31'),
      ],
      C32=[
         SymElem(3, axis=[1, 1, -1], cen=[0.333333333, 0.0, 0.333333333], hel=1.154700538, label='C32'),
      ],
   )
   helper_test_symelem('I23', val, debug, **kw)

def test_symelems_I4(debug=False, **kw):
   val = dict(
      C2=[
         SymElem(2, axis=[0, 0, 1], cen=[0.0, 0.5, 0.0], label='C2'),
      ],
      C21=[
         SymElem(2, axis=[0, 0, 1], cen=[0.75, 0.75, 0.0], hel=0.5, label='C21'),
      ],
      C4=[
         SymElem(4, axis=[0, 0, 1], cen=[-0.5, 0.5, 1.0], label='C4'),
      ],
      C42=[
         SymElem(4, axis=[0, 0, 1], cen=[0.0, 0.5, 0.0], hel=0.5, label='C42'),
      ],
   )
   helper_test_symelem('I4', val, debug, **kw)

def test_symelems_I41(debug=False, **kw):
   val = dict(
      C2=[
         SymElem(2, axis=[0, 0, 1], cen=[0.5, 0.5, 0.0], label='C2'),
      ],
      C41=[
         SymElem(4, axis=[0, 0, 1], cen=[0.25, 0.75, 0.0], hel=0.25, label='C41'),
      ],
      C43=[
         SymElem(4, axis=[0, 0, 1], cen=[0.25, 0.25, 0.0], hel=0.75, label='C43'),
      ],
   )
   helper_test_symelem('I41', val, debug, **kw)

def test_symelems_I4132(debug=False, **kw):
   val = dict(
      C2=[
         SymElem(2, axis=[1, 0, 0], cen=[0.25, 0.0, 0.25], label='C2'),
         SymElem(2, axis=[-1, 1, 0], cen=[0.0, 0.25, 0.125], label='C2'),
         SymElem(2, axis=[-1, 1, 0], cen=[0.0, 0.75, 0.375], label='C2'),
      ],
      C3=[
         SymElem(3, axis=[1, 1, 1], cen=[0.0, 0.0, 0.0], label='C3'),
      ],
      C21=[
         SymElem(2, axis=[0, 1, 1], cen=[0.125, 0.25, 0.0], hel=0.707106781, label='C21'),
      ],
      C31=[
         SymElem(3, axis=[-1, 1, 1], cen=[0.166666667, 0.166666667, 0.0], hel=0.577350269, label='C31'),
      ],
      C32=[
         SymElem(3, axis=[1, 1, 1], cen=[0.0, 0.333333333, 0.666666667], hel=1.154700538, label='C32'),
      ],
      C41=[
         SymElem(4, axis=[0, 0, 1], cen=[0.25, 0.0, 0.0], hel=0.25, label='C41'),
      ],
      C43=[
         SymElem(4, axis=[0, 1, 0], cen=[0.5, 0.0, 0.25], hel=0.75, label='C43'),
      ],
   )
   helper_test_symelem('I4132', val, debug, **kw)

def test_symelems_I432(debug=False, **kw):
   val = dict(
      C2=[
         SymElem(2, axis=[1, 1, 0], cen=[0.0, 0.0, 0.0], label='C2'),
         SymElem(2, axis=[-1, 1, 0], cen=[0.0, 0.5, 0.25], label='C2'),
         SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.0, 0.5], label='C2'),
      ],
      C3=[
         SymElem(3, axis=[1, 1, 1], cen=[0.0, 0.0, 0.0], label='C3'),
      ],
      C4=[
         SymElem(4, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], label='C4'),
      ],
      C21=[
         SymElem(2, axis=[0, 0, 1], cen=[0.25, 0.25, 0.0], hel=0.5, label='C21'),
      ],
      C31=[
         SymElem(3, axis=[1, -1, 1], cen=[0.333333333, 0.333333333, 0.0], hel=0.577350269, label='C31'),
      ],
      C32=[
         SymElem(3, axis=[1, 1, -1], cen=[0.333333333, 0.0, 0.333333333], hel=1.154700538, label='C32'),
      ],
      C42=[
         SymElem(4, axis=[0, 1, 0], cen=[0.5, 0.0, 0.0], hel=0.5, label='C42'),
      ],
   )
   helper_test_symelem('I432', val, debug, **kw)

def test_symelems_F4132(debug=False, **kw):
   val = dict(
      C2=[
         SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.0, 0.0], label='C2'),
         SymElem(2, axis=[-1, 0, 1], cen=[0.25, 0.125, 0.0], label='C2'),
      ],
      C3=[
         SymElem(3, axis=[1, 1, 1], cen=[0.0, 0.0, 0.0], label='C3'),
      ],
      C31=[
         SymElem(3, axis=[-1, 1, 1], cen=[0.166666667, 0.166666667, 0.0], hel=0.577350269, label='C31'),
      ],
      C32=[
         SymElem(3, axis=[-1, 1, 1], cen=[0.166666667, 0.0, 0.166666667], hel=1.154700538, label='C32'),
      ],
      C41=[
         SymElem(4, axis=[0, 1, 0], cen=[0.25, 0.0, 0.0], hel=0.25, label='C41'),
      ],
      C43=[
         SymElem(4, axis=[0, 0, 1], cen=[0.25, 0.0, 0.0], hel=0.75, label='C43'),
      ],
   )
   helper_test_symelem('F4132', val, debug, **kw)

def test_symelems_F432(debug=False, **kw):
   val = dict(
      C2=[
         SymElem(2, axis=[1, 1, 0], cen=[0.0, 0.0, 0.0], label='C2'),
         SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.25, 0.25], label='C2'),
         SymElem(2, axis=[0, -1, 1], cen=[0.0, 0.0, 0.5], label='C2'),
      ],
      C3=[
         SymElem(3, axis=[1, 1, 1], cen=[0.0, 0.0, 0.0], label='C3'),
      ],
      C4=[
         SymElem(4, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], label='C4'),
      ],
      C21=[
         SymElem(2, axis=[0, 0, 1], cen=[0.25, 0.0, 0.0], hel=0.5, label='C21'),
      ],
      C31=[
         SymElem(3, axis=[-1, 1, 1], cen=[0.166666667, 0.166666667, 0.0], hel=0.577350269, label='C31'),
      ],
      C32=[
         SymElem(3, axis=[-1, 1, 1], cen=[0.166666667, 0.0, 0.166666667], hel=1.154700538, label='C32'),
      ],
      C42=[
         SymElem(4, axis=[0, 0, 1], cen=[0.25, 0.25, 0.0], hel=0.5, label='C42'),
      ],
   )
   helper_test_symelem('F432', val, debug, **kw)

def test_remove_redundant_screws():
   sym = 'P212121'
   f4cel = wu.sym.sgframes(sym, cells=6, cellgeom='nonsingular')
   lattice = wu.sym.lattice_vectors(sym, cellgeom='nonsingular')
   elems = {
      'C21': [
         SymElem(2, axis=[0, 0, 1], cen=[0.25, 0.0, 0.0], hel=0.5, label='C21'),
         SymElem(2, axis=[0, 0, 1], cen=[-0.25, 0.0, 0.0], hel=0.5, label='C21'),
         SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.25, 0.0], hel=0.5, label='C21'),
         SymElem(2, axis=[1, 0, 0], cen=[0.0, -0.25, 0.0], hel=0.5, label='C21'),
         SymElem(2, axis=[0, 1, 0], cen=[0.0, 0.0, 0.25], hel=0.5, label='C21'),
         SymElem(2, axis=[0, 1, 0], cen=[0.0, 0.0, -0.25], hel=0.5, label='C21'),
         SymElem(2, axis=[0, 0, 1], cen=[0.25, 0.5, 0.0], hel=0.5, label='C21'),
         SymElem(2, axis=[0, 0, 1], cen=[-0.25, 0.5, 0.0], hel=0.5, label='C21'),
         SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.25, 0.5], hel=0.5, label='C21'),
         SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.25, -0.5], hel=0.5, label='C21'),
         SymElem(2, axis=[1, 0, 0], cen=[0.0, -0.25, 0.5], hel=0.5, label='C21'),
         SymElem(2, axis=[1, 0, 0], cen=[0.0, -0.25, -0.5], hel=0.5, label='C21'),
         SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.75, 0.0], hel=0.5, label='C21'),
         SymElem(2, axis=[0, 1, 0], cen=[0.0, 0.0, 0.75], hel=0.5, label='C21'),
         SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.75, 0.5], hel=0.5, label='C21'),
         SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.75, -0.5], hel=0.5, label='C21'),
         SymElem(2, axis=[0, 0, 1], cen=[0.25, 1.0, 0.0], hel=0.5, label='C21'),
         SymElem(2, axis=[0, 0, 1], cen=[-0.25, 1.0, 0.0], hel=0.5, label='C21'),
         SymElem(2, axis=[0, 1, 0], cen=[0.5, 0.0, 0.75], hel=0.5, label='C21'),
         SymElem(2, axis=[1, 0, 0], cen=[0.0, 1.25, 0.0], hel=0.5, label='C21'),
         SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.25, 1.0], hel=0.5, label='C21'),
         SymElem(2, axis=[1, 0, 0], cen=[0.0, -0.25, 1.0], hel=0.5, label='C21'),
         SymElem(2, axis=[1, 0, 0], cen=[0.0, 1.25, 0.5], hel=0.5, label='C21'),
         SymElem(2, axis=[1, 0, 0], cen=[0.0, 1.25, -0.5], hel=0.5, label='C21'),
         SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.75, 1.0], hel=0.5, label='C21'),
         SymElem(2, axis=[0, 1, 0], cen=[0.0, 0.0, 1.25], hel=0.5, label='C21'),
         SymElem(2, axis=[0, 1, 0], cen=[0.5, 0.0, 1.25], hel=0.5, label='C21'),
         SymElem(2, axis=[1, 0, 0], cen=[0.0, 1.25, 1.0], hel=0.5, label='C21'),
      ]
   }
   # ic(f4cel.shape)
   # ic(lattice)
   elems2 = _remove_redundant_screws(elems, f4cel, lattice)
   # ic(elems2)
   assert elems2 == {
      'C21': [
         SymElem(2, axis=[0, 0, 1], cen=[0.25, 0.0, 0.0], hel=0.5, label='C21'),
         SymElem(2, axis=[0, 0, 1], cen=[-0.25, 0.0, 0.0], hel=0.5, label='C21'),
         SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.25, 0.0], hel=0.5, label='C21'),
         SymElem(2, axis=[1, 0, 0], cen=[0.0, -0.25, 0.0], hel=0.5, label='C21'),
         SymElem(2, axis=[0, 1, 0], cen=[0.0, 0.0, 0.25], hel=0.5, label='C21'),
         SymElem(2, axis=[0, 1, 0], cen=[0.0, 0.0, -0.25], hel=0.5, label='C21'),
      ]
   }

def _printelems(sym, elems):
   print('-' * 80)
   print(sym)
   # print(f'   assert set(elems.keys()) == set(\'{" ".join(elems.keys())}\'.split())')
   print('   val = dict(')
   for k, v in elems.items():
      print(f'      {k}=[')
      for e in v:
         print(f'         {e},')
      print('      ],')
   print(')')

   print('-' * 80, flush=True)

if __name__ == '__main__':
   main()
