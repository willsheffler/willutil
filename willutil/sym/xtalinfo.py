import numpy as np
import willutil as wu

class SymElem:
   def __init__(self, nfold, axis, cen=[0, 0, 0]):
      self.nfold = nfold
      self.origaxis = axis
      self.coords = wu.hray(cen, axis)
      x = wu.hrot(self.coords, nfold=nfold)
      self.operators = np.stack([wu.hpow(x, p) for p in range(nfold)])

   @property
   def cen(self):
      return self.coords[..., 0]

   @property
   def axis(self):
      return self.coords[..., 1]

   @property
   def angle(self):
      return np.pi * 2 / self.nfold

   def __repr__(self):
      # ax = self.axis / min(self.axis[self.axis != 0])
      ax = self.origaxis
      return f'SymElem({self.nfold},{ax[:3]},{self.cen[:3]})'

xtal_info_dict = None

def _populate_xtal_info_dict():
   global xtal_info_dict
   A = np.array

   # yapf: disable
   xtal_info_dict = {
      'P 2 3' : (12,[
         SymElem( nfold= 2 , axis= [ 0,  0,  1 ] , cen= A([ 0, 0, 0 ]) / 1 ),
         SymElem( nfold= 2 , axis= [ 1,  0,  0 ] , cen= A([ 0, 1, 0 ]) / 2 ),
         SymElem( nfold= 3 , axis= [ 1,  1,  1 ] , cen= A([ 0, 0, 0 ]) / 1 ),
      ]),
      'I 21 3' : (24,[
         SymElem( nfold= 2 , axis= [ 0,  0,  1 ] , cen= A([ 2, 1, 0 ]) / 4 ),
         SymElem( nfold= 3 , axis= [ 1,  1,  1 ] , cen= A([ 0, 0, 0 ]) / 1 ),
      ]),
      'I 41 3 2' : (48,[
         SymElem( nfold= 2 , axis= [ 0,  0,  1 ] , cen= A([ 0, 1, 0 ]) / 4 ),
         SymElem( nfold= 2 , axis= [ 0,  0,  1 ] , cen= A([ 2, 1, 0 ]) / 4 ),
         SymElem( nfold= 3 , axis= [ 1,  1,  1 ] , cen= A([ 0, 0, 0 ]) / 1 ),
         SymElem( nfold= 3 , axis= [ 1, -1, -1 ] , cen= A([ 1, 1, 0 ]) / 2 ),
         # Vec(1.125 / 2, 0.500, 0.625)Vec(0.875 / 2, 0.500, 0.375)
      ]),
   }
   # yapf: enable

_populate_xtal_info_dict()

def xtalinfo(name):

   name = name.upper().strip()
   if name in xtal_info_dict:
      return name, xtal_info_dict[name][1], xtal_info_dict[name][0],

   name = name.replace('_', ' ')
   if name in xtal_info_dict:
      return name, xtal_info_dict[name][1], xtal_info_dict[name][0],

   raise ValueError(f'unknown xtal "{name}"')
