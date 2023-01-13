import numpy as np
import copy
import willutil as wu
from willutil.homog import *

def _populate_xtal_info_dict():
   global xtal_info_dict
   A = np.array
   ########################################################################################
   ############# IF YOU CHANGE THESE, REMOVE CACHE FILES OR DISABLE FRAME CACHING ############
   ############# IF YOU CHANGE THESE, REMOVE CACHE FILES OR DISABLE FRAME CACHING ############
   ############# IF YOU CHANGE THESE, REMOVE CACHE FILES OR DISABLE FRAME CACHING ############
   ############# IF YOU CHANGE THESE, REMOVE CACHE FILES OR DISABLE FRAME CACHING ############
   ############################################################################
   # yapf: disable
   xtal_info_dict = {
      'P 4 3 2'   : wu.Bunch( nsub=24 , spacegroup='P 4 3 2', symelems=[
         C4 ( axis= [ 1,  0,  0 ] , cen= A([ 0, 0, 0 ]) / 1 ),
         C4 ( axis= [ 0,  1,  0 ] , cen= A([ 1, 0, 1 ]) / 2 ),

      ]),
      'F 4 3 2'   : wu.Bunch( nsub=96 , spacegroup='F 4 3 2', symelems=[
         C4 ( axis= [ 1,  0,  0 ] , cen= A([ 0, 0, 1 ]) / 2 ),
         C3 ( axis= [ 1,  1,  1 ] , cen= A([ 4, 1, 1 ]) / 6 ),
      ]),

      'I 4 3 2'   : wu.Bunch( nsub=48 , spacegroup='I 4 3 2', symelems=[
         C4 ( axis= [ 0,  0,  1 ] , cen= A([ 0, 0, 0 ]) / 1 ),
         C2 ( axis= [ 0,  1,  1 ] , cen= A([-1,-1, 1 ]) / 4 ),
      ]),

      'P 2 3'    : wu.Bunch( nsub=12 , spacegroup='P 2 3', symelems=[
         C3 ( axis= [ 1,  1,  1 ] , cen= A([ 0, 0, 0 ]) / 1, label='C3_111_000' ),
         C2 ( axis= [ 0,  0,  1 ] , cen= A([ 0, 0, 0 ]) / 1, label='C2_001_000' ),
         C2 ( axis= [ 1,  0,  0 ] , cen= A([ 0, 1, 0 ]) / 2, label='C2_100_010' ),
      ]),
      'P 21 3'   : wu.Bunch( nsub=12 , spacegroup='P 21 3', symelems=[
         C3 ( axis= [ 1,  1,  1 ] , cen= A([ 0, 0, 0 ]) / 2, vizcol=(0.0, 1.0, 1.0) ),
         C3 ( axis= [ 1,  1, -1 ] , cen= A([ 1, 0, 1 ]) / 2, vizcol=(0.3, 1, 0.7) ),
      ]),
      'I 21 3'   : wu.Bunch( nsub=24 , spacegroup='I 21 3', symelems=[
         C3 ( axis= [ 1,  1,  1 ] , cen= A([ 0, 0, 0 ]) / 1 ),
         C2 ( axis= [ 0,  0,  1 ] , cen= A([ 2, 1, 0 ]) / 4 ),
      ]),
      'P 41 3 2'  : wu.Bunch( nsub=24 , spacegroup='P 41 3 2', symelems=[
         C3 ( axis= [ 1,  1,  1 ] , cen= A([ 0, 0, 0 ]) / 1 ),
         C2 ( axis= [ 1,  0,  1 ] , cen= A([ 2, 1, 0 ]) / 8 ),
      ]),
      'I 41 3 2' : wu.Bunch( nsub=48, spacegroup='I 41 3 2', symelems=[
         # D3 ( axis= [ 1,  1,  1 ] , axis2= [ 1, -1,  0 ] , cen= A([ 1, 1, 1 ]) / 8, label='D3_111_1m0_111_8' , vizcol=(0, 1, 0)),
         D2 ( axis= [ 1,  0,  0 ] , axis2= [ 0, -1,  1 ] , cen= A([ 1, 0, 2 ]) / 8, label='D2_100_0m1_102_8' , vizcol=(0, 1, 1)),
         D3 ( axis= [ 1,  1,  1 ] , axis2= [ 1, -1,  0 ] , cen= A([-1,-1,-1 ]) / 8, label='D3_111_1m0_mmm_8' , vizcol=(1, 0, 0)),
         # D2 ( axis= [ 1,  0,  0 ] , axis2= [ 0, -1,  1 ] , cen= A([-1, 0,-2 ]) / 8, label='D2_100_0m1_m12m_8', vizcol=(1, 1, 0)),
      ]),
      'I4132_322' : wu.Bunch( nsub=48, spacegroup='I 41 3 2', symelems=[
         # C3 ( axis= [ 1,  1,  1 ] , cen= A([ 2, 2, 2 ]) / 8, label='C3_111_1m0_111_8' , vizcol=(1, 0, 0)),
         # C2 ( axis= [ 1,  0,  0 ] , cen= A([ 3, 0, 2 ]) / 8, label='D2_100_0m1_102_8' , vizcol=(0, 1, 0)),
         # C2 ( axis= [ 1, -1,  0 ] , cen= A([-2.7, 0.7,-1 ]) / 8, label='D3_111_1m0_mmm_8' , vizcol=(0, 0, 1)),
         C3 ( axis= [ 1,  1,  1 ] , cen= A([ 0, 0, 0 ]) / 8, label='C3_111_1m0_111_8' , vizcol=(1, 0, 0)),
         C2 ( axis= [ 1,  0,  0 ] , cen= A([-1, 0, 2 ]) / 8, label='D2_100_0m1_102_8' , vizcol=(0, 1, 0)),
         C2 ( axis= [ 1,  1,  0 ] , cen= [-0.1625,  0.0875,  0.125 ], label='D3_111_1m0_mmm_8' , vizcol=(0, 0, 1)),
      ]),
      'L6_32'   : wu.Bunch( nsub=None , spacegroup=None, dimension=2, symelems=[
         C3 ( axis= [ 0,  0,  1 ] , cen= A([ 0, 0, 0 ])/2, vizcol=(0.0, 1.0, 1.0) ),
         C2 ( axis= [ 0,  0,  1 ] , cen= A([ 1, 0, 0 ])/2, vizcol=(0.3, 1, 0.7) ),
      ]),
      'L6M_322' : wu.Bunch( nsub=None , spacegroup=None, dimension=2, symelems=[
         C3 ( axis= [ 0,  0,  1 ] , cen= A([ 0, 0, 0 ])/2, vizcol=(0.0, 1.0, 1.0) ),
         C2 ( axis= [ 0,  0,  1 ] , cen= A([ 1, 0, 0 ])/2, vizcol=(0.3, 1, 0.7) ),
         C2 ( axis= [ 1,  0,  0 ] , cen= A([ 0, 0, 0 ])/2, vizcol=(0.5, 1, 0.8) ),
      ]),
      'L4_44'   : wu.Bunch( nsub=None , spacegroup=None, dimension=2, symelems=[
         C4 ( axis= [ 0,  0,  1 ] , cen= A([ 0, 0, 0 ])/2, vizcol=(0.0, 1.0, 1.0) ),
         C4 ( axis= [ 0,  0,  1 ] , cen= A([ 1, 0, 0 ])/2, vizcol=(0.3, 1, 0.7) ),
      ]),
      'L4_42'   : wu.Bunch( nsub=None , spacegroup=None, dimension=2, symelems=[
         C4 ( axis= [ 0,  0,  1 ] , cen= A([ 0, 0, 0 ])/2, vizcol=(0.0, 1.0, 1.0) ),
         C2 ( axis= [ 0,  0,  1 ] , cen= A([ 1, 0, 0 ])/2, vizcol=(0.3, 1, 0.7) ),
      ]),
      'L3_33'   : wu.Bunch( nsub=None , spacegroup=None, dimension=2, symelems=[
         C3 ( axis= [ 0,  0,  1 ] , cen= A([ 0, 0, 0 ])/2, vizcol=(0.0, 1.0, 1.0) ),
         C3 ( axis= [ 0,  0,  1 ] , cen= A([ 1, 0, 0 ])/2, vizcol=(0.3, 1, 0.7) ),
      ]),


   }
   # yapf: enable

class SymElem:
   def __init__(self, nfold, axis, cen=[0, 0, 0], axis2=None, label=None, vizcol=None):
      self.nfold = nfold
      self.origaxis = axis
      self.origaxis2 = axis2
      self.angle = np.pi * 2 / self.nfold
      self.axis = wu.homog.hgeom.hvec(axis)
      self.axis2 = axis2
      self.vizcol = vizcol
      if self.axis2 is not None:
         self.axis2 = wu.homog.hgeom.hvec(self.axis2)
      self.cen = wu.homog.hgeom.hpoint(cen)
      self.origin = np.eye(4)
      self.label = label
      if self.label is None:
         if axis2 is None: self.label = f'C{self.nfold}'
         else: self.label = f'D{self.nfold}'

   @property
   def operators(self):
      # ic(self)
      x = wu.homog.hgeom.hrot(self.axis, nfold=self.nfold, center=self.cen)
      ops = [wu.homog.hgeom.hpow(x, p) for p in range(self.nfold)]
      if self.axis2 is not None:
         xd2f = wu.homog.hgeom.hrot(self.axis2, nfold=2, center=self.cen)
         ops = ops + [xd2f @ x for x in ops]
      ops = np.stack(ops)
      assert wu.homog.hgeom.hvalid(ops)
      return ops

   @property
   def coords(self):
      axis2 = [0, 0, 0, 0] if self.axis2 is None else self.axis2
      return np.stack([self.axis, axis2, self.cen])

   def xformed(self, xform):
      assert xform.shape[-2:] == (4, 4)
      single = False
      if xform.ndim == 2:
         xform = xform.reshape(1, 4, 4)
         single = True
      result = list()
      for x in xform:
         other = copy.copy(self)
         other.axis = wu.hxform(x, self.axis)
         if self.axis2 is not None:
            other.axis2 = wu.hxform(x, self.axis2)
         other.cen = wu.hxform(x, self.cen)
         result.append(other)
      if single:
         result = result[0]
      return result

   def __repr__(self):
      # ax = self.axis / min(self.axis[self.axis != 0])
      ax = self.origaxis
      ax2 = self.origaxis2
      if self.origaxis2 is None:
         return f'SymElem({self.nfold}, ax={ax[:3]}, cen={self.cen[:3]})'
      else:
         return f'SymElem({self.nfold}, ax={ax[:3]}, dax{ax2[:3]}, cen={self.cen[:3]})'

def C2(**kw):
   return SymElem(nfold=2, **kw)

def C3(**kw):
   return SymElem(nfold=3, **kw)

def C4(**kw):
   return SymElem(nfold=4, **kw)

def C6(**kw):
   return SymElem(nfold=6, **kw)

def D2(**kw):
   return SymElem(nfold=2, **kw)

def D3(**kw):
   return SymElem(nfold=3, **kw)

def D4(**kw):
   return SymElem(nfold=4, **kw)

def D6(**kw):
   return SymElem(nfold=6, **kw)

xtal_info_dict = None

_populate_xtal_info_dict()

def is_known_xtal(name):
   try:
      xtalinfo(name)
      return True
   except KeyError:
      return False

def xtalinfo(name):
   name = name.upper().strip()

   alternate_names = dict(
      P213='P 21 3',
      P213_33='P 21 3',
      I213_32='I 21 3',
      I213='I 21 3',
      L6M322='L6M_322',
      L632='L6_32',
      P6_32='L6_32',
      P4_42='L4_42',
      P4_44='L4_44',
      P3_33='L3_33',
   )

   if not name in xtal_info_dict:
      if name in alternate_names:
         name = alternate_names[name]
   if not name in xtal_info_dict:
      name = name.replace('_', ' ')
   # ic(name)
   return name, xtal_info_dict[name]

   raise ValueError(f'unknown xtal "{name}"')
