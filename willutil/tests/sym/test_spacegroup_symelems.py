import itertools, pytest, os
import numpy as np
import willutil as wu
from willutil.sym.SymElem import SymElem, showsymelems
from willutil.sym.spacegroup_symelems import _compute_symelems, _find_compound_symelems, _remove_redundant_screws
from numpy import array

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

def main():

   # test_symelems_P1211()
   # test_symelems_P2221()
   # test_symelems_P21212()
   # test_symelems_P1()
   # test_symelems_C121()
   # test_symelems_P3()
   # test_symelems_P222()
   # test_symelems_P23()
   # test_symelems_F23()
   # test_symelems_R32()

   # assert 0

   # test_symelems_P3121()
   # test_symelems_P212121()
   # test_symelems_P31()
   # test_symelems_P32()
   # test_symelems_P213()
   # test_symelems_P3221()
   # test_symelems_P41()
   # test_symelems_P41212()
   # test_symelems_P4132()
   # test_symelems_P4232()
   # test_symelems_P43()
   # test_symelems_P432()
   # test_symelems_P43212()
   # test_symelems_P4332()
   # test_symelems_P6()
   # test_symelems_P61()
   # test_symelems_P6122()
   # test_symelems_P62()
   # test_symelems_P63()
   # test_symelems_P64()
   # test_symelems_P65()
   # test_symelems_P6522()
   # test_symelems_I213()
   # test_symelems_I23()
   # test_symelems_I4()
   # test_symelems_I41()
   # test_symelems_I4132()
   # test_symelems_I432()
   # test_symelems_F4132()
   # test_symelems_F432()

   # test_remove_redundant_screws()

   test_compound_elems_P23()
   assert 0
   test_compound_elems_F23()
   assert 0
   test_compound_elems_P213()
   test_compound_elems_I23()

   test_compound_elems_P4132()
   test_compound_elems_P432()
   test_compound_elems_I432()
   test_compound_elems_F432()
   test_compound_elems_I4132()
   test_compound_elems_F4132()

   ic('PASS test_spacegroup_symelems')

# yapf: disable



def test_compound_elems_P23(debug=False,**kw):
   sym = 'P23'
   val = {}
   # SymElem(2, axis=[1, 0, 0], axis2=[0.0, 1.0, 0.0], cen=[0.5, 0.0, 0.0]),
   # SymElem(2, axis=[1, 0, 0], axis2=[0.0, 0.0, 1.0], cen=[0.5, 0.5, 0.0]),
   # ]
   # assert elems['T'] == [
   # SymElem('T32', axis=[1, 1, 1], axis2=[0.0, 0.0, 1.0], cen=[0.0, 0.0, 0.0]),
   # SymElem('T32', axis=[1, 1, 1], axis2=[1.0, 0.0, 0.0], cen=[0.5, 0.5, 0.5]),
   # ]

   helper_test_symelem(sym,val,debug,compound=True,**kw)


def test_compound_elems_I23(debug=False,**kw):
   sym = 'I23'
   val = dict(
      D2=[
         SymElem(2, axis=[1, 0, 0], axis2=[0.0, 1.0, 0.0], cen=[0.5, 0.0, 0.0], label='D2'),
      ],
      T=[
         SymElem('T32', axis=[1, 1, 1], axis2=[0.0, 0.0, 1.0], cen=[0.0, 0.0, 0.0], label='T'),
      ],
   )
   helper_test_symelem(sym,val,debug,compound=True,**kw)

def test_compound_elems_F23(debug=False,**kw):
   sym = 'F23'
   val = {}
   # assert elems['T'] == [
   # SymElem('T32', axis=[1, 1, 1], axis2=[0.0, 0.0, 1.0], cen=[0.0, 0.0, 0.0], label='T'),
   # SymElem('T32', axis=[1, 1, 1], axis2=[1.0, 0.0, 0.0], cen=[0.25, 0.25, 0.25], label='T'),

   helper_test_symelem(sym,val,debug,compound=True,**kw)



# def test_compound_elems_P4132(showme=False):
#    ic('test_compound_elems_P4132')
#    sym = 'P4132'
#    elems = wu.sym.symelems(sym, asdict=True)
#    celems = _find_compound_symelems(sym)
#    if showme: showsymelems(sym, elems)
#    if showme: showsymelems(sym, celems)
#    # print(repr(celems), flush=True)
#    assert set(elems.keys()) == set('D3'.split())
#    assert celems['D3'] == [SymElem(3, axis=[1, 1, 1], axis2=[-1.0, 1.0, 0.0], cen=[0.375, 0.375, 0.375], label='D3')]

# def test_compound_elems_P432(showme=False):
#    ic('test_compound_elems_P432')
#    sym = 'P432'
#    elems = wu.sym.symelems(sym, asdict=True)
#    celems = _find_compound_symelems(sym)

#    # for k, v in celems.items():
#    #    print(k)
#    #    for x in v:
#    #       print(x, flush=True)

#    if showme: showsymelems(sym, elems)
#    if showme: showsymelems(sym, celems)

#    print(repr(celems), flush=True)
#    assert set(elems.keys()) == set('O D4'.split())
#    assert elems['O'] == [
#       SymElem('O43', axis=[0, 0, 1], axis2=[1.0, 1.0, 1.0], cen=[0.0, 0.0, 0.0], label='O'),
#       SymElem('O43', axis=[1, 0, 0], axis2=[1.0, 1.0, 1.0], cen=[0.5, 0.5, 0.5], label='O'),
#    ]
#    assert elems['D4'] == [
#       SymElem(4, axis=[1, 0, 0], axis2=[0.0, 1.0, 0.0], cen=[0.5, 0.0, 0.0], label='D4'),
#       SymElem(4, axis=[0, 0, 1], axis2=[1.0, 0.0, 0.0], cen=[0.5, 0.5, 0.0], label='D4'),
#    ]

# def test_compound_elems_I432(showme=False):
#    ic('test_compound_elems_I432')
#    sym = 'I432'
#    elems = wu.sym.symelems(sym, asdict=True)
#    celems = _find_compound_symelems(sym)

#    # for k, v in celems.items():
#    #    print(k)
#    #    for x in v:
#    #       print(x, flush=True)

#    if showme: showsymelems(sym, elems)
#    if showme: showsymelems(sym, celems)

#    print(repr(celems), flush=True)
#    assert set(elems.keys()) == set('O D2 D3 D4'.split())
#    assert elems['O'] == [SymElem('O43', axis=[0, 0, 1], axis2=[1.0, 1.0, 1.0], cen=[0.0, 0.0, 0.0], label='O')]
#    assert elems['D2'] == [SymElem(2, axis=[0, 1, 0], axis2=[-1.0, 0.0, 1.0], cen=[0.5, 0.25, 0.0], label='D2')]
#    assert elems['D4'] == [SymElem(4, axis=[1, 0, 0], axis2=[0.0, 1.0, 0.0], cen=[0.5, 0.0, 0.0], label='D4')]
#    assert elems['D3'] == [SymElem(3, axis=[1, 1, 1], axis2=[-1.0, 1.0, 0.0], cen=[0.25, 0.25, 0.25], label='D3')]

# def test_compound_elems_F4132(showme=False):
#    ic('test_compound_elems_F4132')
#    sym = 'F4132'
#    elems = wu.sym.symelems(sym, asdict=True)
#    celems = _find_compound_symelems(sym)

#    # for k, v in celems.items():
#    #    print(k)
#    #    for x in v:
#    #       print(x, flush=True)

#    if showme: showsymelems(sym, elems)
#    if showme: showsymelems(sym, celems)
#    assert set(elems.keys()) == set('T D3'.split())
#    print(repr(celems), flush=True)
#    assert elems['T'] == [SymElem('T32', axis=[1, 1, 1], axis2=[1.0, 0.0, 0.0], cen=[0.0, 0.0, 0.0], label='T')]
#    assert elems['D3'] == [SymElem(3, axis=[1, 1, 1], axis2=[0.0, -1.0, 1.0], cen=[0.125, 0.125, 0.125], label='D3')]

# def test_compound_elems_F432(showme=False):
#    ic('test_compound_elems_F432')
#    sym = 'F432'
#    elems = wu.sym.symelems(sym, asdict=True)
#    celems = _find_compound_symelems(sym)

#    # for k, v in celems.items():
#    #    print(k)
#    #    for x in v:
#    #       print(x, flush=True)

#    if showme: showsymelems(sym, elems)
#    if showme: showsymelems(sym, celems)
#    # print(repr(celems), flush=True)
#    for k, v in celems.items():
#       ic(k)
#       for e in v:
#          ic(e)
#    assert set(elems.keys()) == set('O D2 T'.split())
#    assert elems['D2'] == [
#       SymElem(2, axis=[1, 1, 0], axis2=[0.0, 0.0, 1.0], cen=[0.25, 0.25, 0.0], label='D2'),
#    ]
#    assert elems['O'] == [
#       SymElem('O43', axis=[0, 0, 1], axis2=[1.0, 1.0, 1.0], cen=[0.0, 0.0, 0.0], label='O'),
#       SymElem('O43', axis=[1, -1, 1], axis2=[1.0, 0.0, 1.0], cen=[0.5, 0.0, 0.0], label='O'),
#    ]
#    assert elems['T'] == [
#       SymElem('T32', axis=[1, 1, 1], axis2=[1.0, 0.0, 0.0], cen=[0.25, 0.25, 0.25], label='T'),
#    ]

# def test_compound_elems_I4132(showme=False):
#    ic('test_compound_elems_I4132')
#    sym = 'I4132'
#    elems = _find_compound_symelems(sym)

#    # for k, v in celems.items():
#    #    print(k)
#    #    for x in v:
#    #       print(x, flush=True)

#    # if showme: showsymelems(sym, elems)
#    if showme: showsymelems(sym, celems)
#    # print(repr(celems), flush=True)
#    for k, v in celems.items():
#       ic(k)
#       for e in v:
#          ic(e)
#    assert set(elems.keys()) == set('D2 D3'.split())
#    assert elems['D2'] == [
#       SymElem(2, axis=[-1, 1, 0], axis2=[0.0, 0.0, 1.0], cen=[0.5, 0.25, 0.375], label='D2'),
#       SymElem(2, axis=[1, 0, 1], axis2=[-1.0, 0.0, 1.0], cen=[0.25, 0.125, 0.0], label='D2'),
#    ]
#    assert elems['D3'] == [
#       SymElem(3, axis=[1, 1, 1], axis2=[-1.0, 1.0, 0.0], cen=[0.125, 0.125, 0.125], label='D3'),
#       SymElem(3, axis=[1, 1, 1], axis2=[-1.0, 1.0, 0.0], cen=[0.375, 0.375, 0.375], label='D3'),
#    ]

# def test_compound_elems_P213(showme=False):
#    ic('test_compound_elems_P213')
#    sym = 'P213'
#    elems = _find_compound_symelems(sym)

#    # for k, v in celems.items():
#    #    print(k)
#    #    for x in v:
#    #       print(x, flush=True)

#    # if showme: showsymelems(sym, elems)
#    if showme: showsymelems(sym, celems)

#    assert elems == {}


def helper_test_symelem(sym, eref=None, debug=False,compound=False, **kw):
   if compound:
      otherelems = wu.sym.symelems(sym, asdict=True)
      # otherelems = _compute_symelems(sym, profile=debug, aslist=False)
      symelems = list(itertools.chain(*otherelems.values()))
      elems0 = _find_compound_symelems(sym, symelems)
      assert 0
   else:
      otherelems = {}
      elems0 = _compute_symelems(sym, profile=debug)

   etst = elems0.copy()
   eref = eref.copy()
   if 'C11' in etst:   del etst['C11']
   if 'C11' in eref: del eref['C11']

   ok = True
   if eref is not None:
      vkey = set(eref.keys())
      tkey = set(etst.keys())
      key = sorted(vkey.intersection(tkey))
      for k in vkey - tkey:
         ok = False
         print('MISSING', k)
      for k in tkey - vkey:
         ok = False
         print('EXTRA', k)
      for k in key:
         tval = wu.misc.UnhashableSet(etst[k])
         vval = wu.misc.UnhashableSet(eref[k])
         x = vval.difference(tval)
         if x:
            ok = False
            print(k, 'MISSING')
            for v in x:
               print('  ', v)
         x = tval.difference(vval)
         if x:
            ok = False
            print(k, 'EXTRA')
            for v in x:
               print('  ', v)
         x = tval.intersection(vval)
         if x:
            print(k, 'COMMON')
            for v in x:
               print('  ', v)
   if not ok or debug:
      _printelems(sym, etst)
      showsymelems(sym, {**otherelems,**etst}, scale=12, scan=12, offset=0, **kw)
      assert ok
   assert not debug


def test_symelems_R32(debug=False, **kw):
   val = dict(
      C2=[
         SymElem(2, axis=[0.57735, 1.0, 0.0], cen=[1.0, 1.0, -1e-09], label='C2'),
         SymElem(2, axis=[-0.57735, 1.0, 0.0], cen=[0.333333333, 0.166666667, 0.166666667], label='C2'),
      ],
      C3=[
         SymElem(3, axis=[0, 0, 1], cen=[1e-09, 1e-09, 0.333333334], label='C3'),
      ],
      C21=[
         SymElem(2, axis=[1, 0, 0], cen=[0.083333333, 0.166666667, 0.166666667], hel=0.5, label='C21'),
         SymElem(2, axis=[1, 0, 0], cen=[0.25, 0.5, 0.0], hel=0.5, label='C21'),
      ],
      C32=[
         SymElem(3, axis=[0, 0, 1], cen=[0.333333334, 0.0, 0.0], hel=0.666666667, label='C32'),
      ],
      C31=[
         SymElem(3, axis=[0, 0, 1], cen=[0.666666667, 0.0, 0.0], hel=0.333333333, label='C31'),
      ],
   )
   helper_test_symelem('R32', val, debug, **kw)

def test_symelems_P1211(debug=False, **kw):
   val = dict(
      C21=[
         SymElem(2, axis=[0, 1, 0], cen=[0.0, 0.0, 0.0], hel=0.5, label='C21'),
         SymElem(2, axis=[0, 1, 0], cen=[0.5, 0.0, 0.0], hel=0.5, label='C21'),
         SymElem(2, axis=[0, 1, 0], cen=[0.0, 0.0, 0.5], hel=0.5, label='C21'),
         SymElem(2, axis=[0, 1, 0], cen=[0.5, 0.0, 0.5], hel=0.5, label='C21'),
      ],
      C11=[
         SymElem(1, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], hel=1.0, label='C11'),
      ],
   )
   helper_test_symelem('P1211', val, debug, **kw)

def test_symelems_P2221(debug=False, **kw):
   val = dict(
      C2=[
         SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.0, 0.0], label='C2'),
         SymElem(2, axis=[0, 1, 0], cen=[0.0, 0.0, 0.25], label='C2'),
         SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.5, 0.0], label='C2'),
         SymElem(2, axis=[0, 1, 0], cen=[0.5, 0.0, 0.25], label='C2'),
      ],
      C21=[
         SymElem(2, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], hel=0.5, label='C21'),
         SymElem(2, axis=[0, 0, 1], cen=[0.5, 0.0, 0.0], hel=0.5, label='C21'),
         SymElem(2, axis=[0, 0, 1], cen=[0.0, 0.5, 0.0], hel=0.5, label='C21'),
         SymElem(2, axis=[0, 0, 1], cen=[0.5, 0.5, 0.0], hel=0.5, label='C21'),
      ],
   )
   helper_test_symelem('P2221', val, debug, **kw)

def test_symelems_P21212(debug=False, **kw):
   val = dict(
      C2=[
         SymElem(2, axis=[0, 0, 1], cen=[0.0, 0.0, -1.0], label='C2'),
         SymElem(2, axis=[0, 0, 1], cen=[0.0, 0.5, 0.0], label='C2'),
      ],
      C21=[
         SymElem(2, axis=[0, 1, 0], cen=[0.25, 0.0, 0.0], hel=0.5, label='C21'),
         SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.25, 0.0], hel=0.5, label='C21'),
         SymElem(2, axis=[0, 1, 0], cen=[0.25, 0.0, 0.5], hel=0.5, label='C21'),
         SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.25, 0.5], hel=0.5, label='C21'),
      ],
   )
   helper_test_symelem('P21212', val, debug, **kw)

def test_symelems_P1(debug=False, **kw):
   val = dict()
   helper_test_symelem('P1', val, debug, **kw)

def test_symelems_C121(debug=False, **kw):
   val = dict(
      C2=[
         SymElem(2, axis=[0, 1, 0], cen=[0.0, 0.0, 0.0], label='C2'),
         SymElem(2, axis=[0, 1, 0], cen=[0.0, 0.0, 0.5], label='C2'),
      ],
      C21=[
         SymElem(2, axis=[0, 1, 0], cen=[0.25, 0.0, 0.0], hel=0.5, label='C21'),
         SymElem(2, axis=[0, 1, 0], cen=[0.25, 0.0, 0.5], hel=0.5, label='C21'),
      ],
   )
   helper_test_symelem('C121', val, debug, **kw)

def test_symelems_P3(debug=False, **kw):
   val = dict(
      C3=[
         SymElem(3, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], label='C3'),
         SymElem(3, axis=[0, 0, 1], cen=[-0.333333333, 0.333333335, 0.0], label='C3'),
         SymElem(3, axis=[0, 0, 1], cen=[0.333333334, 0.666666666, 0.0], label='C3'),
      ],
   )
   helper_test_symelem('P3', val, debug, **kw)

def test_symelems_P222(debug=False, **kw):
   val = dict(
      C2=[
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
      ],
   )
   helper_test_symelem('P222', val, debug, **kw)

def test_symelems_P23(debug=False, **kw):
   val = dict(
      C2=[
         SymElem(2, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], label='C2'),
         SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.0, 0.5], label='C2'),
         SymElem(2, axis=[0, 1, 0], cen=[0.0, 0.0, 0.5], label='C2'),
         SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.5, 0.5], label='C2'),
      ],
      C3=[
         SymElem(3, axis=[1, 1, 1], cen=[0.0, 0.0, 0.0], label='C3'),
      ],
      C31=[
         SymElem(3, axis=[1, -1, 1], cen=[0.333333333, 0.333333333, 0.0], hel=0.577350269, label='C31'),
      ],
      C32=[
         SymElem(3, axis=[1, 1, -1], cen=[0.333333333, 0.0, 0.333333333], hel=1.154700538, label='C32'),
      ],
   )
   helper_test_symelem('P23', val, debug, **kw)

def test_symelems_F23(debug=False, **kw):
   val = dict(
      C2=[
         SymElem(2, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], label='C2'),
         SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.25, 0.25], label='C2'),
      ],
      C3=[
         SymElem(3, axis=[1, 1, 1], cen=[0.0, 0.0, 0.0], label='C3'),
      ],
      C21=[
         SymElem(2, axis=[0, 0, 1], cen=[0.25, 0.0, 0.0], hel=0.5, label='C21'),
         SymElem(2, axis=[0, 1, 0], cen=[0.25, 0.0, 0.0], hel=0.5, label='C21'),
      ],
      C31=[
         SymElem(3, axis=[-1, 1, 1], cen=[0.16666666666666666, 0.16666666666666666, 0.0], hel=0.5773502691896257, label='C31'),
      ],
      C32=[
         SymElem(3, axis=[-1, 1, 1], cen=[0.16666666666666666, 0.0, 0.16666666666666666], hel=1.154700538368877, label='C32'),
      ],
   )
   helper_test_symelem('F23', val, debug, **kw)

def test_symelems_I41(debug=False, **kw):
   val = dict()
   helper_test_symelem('I41', val, debug, **kw)

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
         SymElem(2, axis=[0, 1, 1], cen=[0.375, 0.0, 0.25], hel=0.707106781, label='C21'),
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
         SymElem(2, axis=[0, 1, 1], cen=[0.25, 0.0, 0.0], hel=0.707106781, label='C21'),
         SymElem(2, axis=[-1, 1, 0], cen=[0.5, 0.0, 0.0], hel=0.707106781, label='C21'),
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
         SymElem(3, axis=[-1, 1, 1], cen=[0.16666666666666666, 0.16666666666666666, 0.0], hel=0.5773502691896257, label='C31'),
      ],
      C32=[
         SymElem(3, axis=[1, 1, 1], cen=[0.0, 0.3333333333333333, 0.6666666666666666], hel=1.154700538368877, label='C32'),
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
         SymElem(2, axis=[0, 1, 1], cen=[0.75, 0.0, 0.0], hel=0.707106781, label='C21'),
      ],
      C31=[
         SymElem(3, axis=[1, -1, 1], cen=[0.333333333, 0.333333333, 0.0], hel=0.577350269, label='C31'),
      ],
      C32=[
         SymElem(3, axis=[1, 1, -1], cen=[0.333333333, 0.0, 0.333333333], hel=1.154700538, label='C32'),
      ],
      C42=[
         SymElem(4, axis=[0, 1, 0], cen=[0.5, 0.0, 0.0], hel=0.5, label='C42'),
         SymElem(4, axis=[0, 0, 1], cen=[0.5, 0.0, 0.0], hel=0.5, label='C42'),
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
   val = dict(
      C2=[
         SymElem(2, axis=[1, 1, 0], cen=[0.0, 0.0, 0.0], label='C2'),
         SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.0, 0.5], label='C2'),
         SymElem(2, axis=[1, 1, 0], cen=[0.0, 0.0, 0.5], label='C2'),
      ],
      C3=[
         SymElem(3, axis=[1, 1, 1], cen=[0.0, 0.0, 0.0], label='C3'),
      ],
      C4=[
         SymElem(4, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], label='C4'),
         SymElem(4, axis=[1, 0, 0], cen=[0.0, -0.5, 0.5], label='C4'),
      ],
      C21=[
         SymElem(2, axis=[-1, 0, 1], cen=[0.5, 0.5, 0.0], hel=0.707106781, label='C21'),
         SymElem(2, axis=[-1, 1, 0], cen=[0.5, 0.0, 0.0], hel=0.707106781, label='C21'),
      ],
      C31=[
         SymElem(3, axis=[1, -1, 1], cen=[0.333333333, 0.333333333, 0.0], hel=0.577350269, label='C31'),
      ],
      C32=[
         SymElem(3, axis=[1, 1, -1], cen=[0.333333333, 0.0, 0.333333333], hel=1.154700538, label='C32'),
      ],
   )
   helper_test_symelem('P432', val, debug, **kw)

def test_symelems_P43212(debug=False, **kw):
   val = dict(
      C2=[
         SymElem(2, axis=[1, 1, 0], cen=[0.5, 0.5, 0.0], label='C2'),
      ],
      C21=[
         SymElem(2, axis=[0, 0, 1], cen=[0.0, 0.0, 0.0], hel=0.5, label='C21'),
         SymElem(2, axis=[0, 1, 0], cen=[0.25, 0.0, 0.875], hel=0.5, label='C21'),
         SymElem(2, axis=[1, 0, 0], cen=[0.0, 0.75, 0.125], hel=0.5, label='C21'),
         SymElem(2, axis=[1, 1, 0], cen=[0.0, 0.5, 0.0], hel=0.7071067811865476, label='C21'),
      ],
      C43=[
         SymElem(4, axis=[0, 0, 1], cen=[0.5, 0.0, 0.0], hel=0.75, label='C43'),
      ],
   )
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
         SymElem(2, axis=[0, 0, 1], cen=[0.5, 0.0, 0.0], hel=0.5, label='C21'),
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
         SymElem(2, axis=[0, 0, 1], cen=[0.5, 0.0, 0.0], hel=0.5, label='C21'),
      ],
      C32=[
         SymElem(3, axis=[0, 0, 1], cen=[0.666666667, 0.333333333, 0.0], hel=0.666666667, label='C32'),
      ],
      C65=[
         SymElem(6, axis=[0, 0, 1], cen=[1e-09, 0.0, 0.0], hel=0.833333333, label='C65'),
      ],
   )
   helper_test_symelem('P6522', val, debug, **kw)

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
