import numpy as np
import willutil as wu
from willutil.sym.SymElem import SymElem
from numpy import array

def main():

   test_symelems_I4132()

   # test_symelems_P4132()
   test_symelems_P4232()
   # test_symelems_P4332()

   test_symelems_I213()
   test_symelems_P213()

   test_symelems_P23()
   test_symelems_F23()
   test_symelems_I23()

   test_symelems_P432()
   test_symelems_I432()
   test_symelems_F432()

   test_symelems_F4132()

   ic('PASS test_spacegroup_symelems')

def test_symelems_I4132(showme=False, **kw):
   sym = 'I4132'
   symelems = wu.sym.compute_symelems(sym, wu.sym.sgframes(sym, cellgeom='unit'))
   ic(sym, symelems)
   if showme: showsymelems(sym, symelems, allframes=True, **kw)

   assert symelems == {
      'C2': [
         SymElem(2, axis=[0.0, 1.0, 0.0], cen=[0.25, 0.0, -0.0]),
         SymElem(2, axis=[-0.0, 1.0, -1.0], cen=[0.125, 0.0, 0.25]),
         SymElem(2, axis=[-0.0, 1.0, -1.0], cen=[0.375, 0.875, -0.125]),
      ],
      'C21': [SymElem(2, axis=[0.0, 0.0, 1.0], cen=[0.25, -0.0, 0.0], hel=0.5), SymElem(2, axis=[0.0, 1.0, 0.0], cen=[0.5, 0.0, 0.25], hel=0.5)],
      'C3': [SymElem(3, axis=[1.0, 1.0, 1.0], cen=[0.0, 0.0, 0.0])],
      'C41': [SymElem(4, axis=[0.0, 0.0, 1.0], cen=[0.25, 0.0, 0.0], hel=-0.75)],
      'C43': [SymElem(4, axis=[0.0, 1.0, 0.0], cen=[0.5, 0.0, 0.25], hel=0.75)],
   }

def test_symelems_P4232(showme=False, **kw):
   sym = 'P4232'
   symelems = wu.sym.compute_symelems(sym, wu.sym.sgframes(sym, cellgeom='unit'))
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
   symelems = wu.sym.compute_symelems(sym, wu.sym.sgframes(sym, cellgeom='unit'))
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
   symelems = wu.sym.compute_symelems(sym, wu.sym.sgframes(sym, cellgeom='unit'))
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
   symelems = wu.sym.compute_symelems(sym, wu.sym.sgframes(sym, cellgeom='unit'))
   ic(sym, symelems)
   if showme: showsymelems(sym, symelems, allframes=True, **kw)
   assert symelems == {
      'C2': [SymElem(2, axis=[0.0, 0.0, 1.0], cen=[-0.0, -0.0, 0.0]), SymElem(2, axis=[-0.0, 1.0, -1.0], cen=[0.125, 0.0, 0.25])],
      'C21': [
         SymElem(2, axis=[0.0, 1.0, 1.0], cen=[0.125, 0.0, 0.0], hel=0.3535533906),
         SymElem(2, axis=[0.0, 0.0, 1.0], cen=[0.25, -0.0, 0.0], hel=0.5),
         SymElem(2, axis=[0.0, 1.0, 0.0], cen=[0.25, 0.0, -0.0], hel=0.5),
         SymElem(2, axis=[0.0, 1.0, 1.0], cen=[0.625, 0.0, 0.0], hel=0.3535533906),
      ],
      'C3': [SymElem(3, axis=[1.0, 1.0, 1.0], cen=[0.0, 0.0, 0.0])],
      'C41': [SymElem(4, axis=[0.0, 1.0, 0.0], cen=[0.25, 0.0, 0.0], hel=0.25)],
      'C43': [SymElem(4, axis=[0.0, 0.0, 1.0], cen=[0.25, 0.0, 0.0], hel=-0.25)],
   }

def test_symelems_P23(showme=False, **kw):
   sym = 'P23'
   symelems = wu.sym.compute_symelems(sym, wu.sym.sgframes(sym, cellgeom='unit'))
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
   symelems = wu.sym.compute_symelems(sym, wu.sym.sgframes(sym, cellgeom='unit'))
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
   symelems = wu.sym.compute_symelems(sym, wu.sym.sgframes(sym, cellgeom='unit'))
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
   symelems = wu.sym.compute_symelems(sym, wu.sym.sgframes(sym, cellgeom='unit'))
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
   symelems = wu.sym.compute_symelems(sym, wu.sym.sgframes(sym, cellgeom='unit'))
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
   symelems = wu.sym.compute_symelems(sym, wu.sym.sgframes(sym, cellgeom='unit'))
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

def showsymelems(sym, symelems, allframes=True, colorbyelem=False, cells=3, bounds=[-0.1, 1.1], scale=10, offset=0.2):
   f = np.eye(4).reshape(1, 4, 4)
   if allframes: f = wu.sym.sgframes(sym, cells=cells, cellgeom=[10])
   args = wu.Bunch(xyzlen=[0.3, 0.4, 1.0])

   ii = 0
   for i, c in enumerate(symelems):
      for j, s in enumerate(symelems[c]):
         if colorbyelem: args.colors = [[(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)][ii]]
         wu.showme(
            f @ wu.htrans(s.cen * scale + offset * wu.hvec([0.1, 0.2, 0.3])) @ wu.halign([0, 0, 1], s.axis),
            name=s.label,
            bounds=[b * 10 for b in bounds],
            **args,
         )
         ii += 1
   from willutil.viz.pymol_viz import showcube
   showcube(0, 10)

if __name__ == '__main__':
   main()
