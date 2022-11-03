import numpy as np
from willutil.homog.hxtal import SymElem, Xtal
from willutil.viz.pymol_viz import pymol_load, cgo_cyl, cgo_sphere, cgo_fan, cgo_cube, showcube
import willutil as wu

@pymol_load.register(SymElem)
def pymol_viz_SymElem(
   toshow,
   state,
   col='bycx',
   name='SymElem',
   center=np.array([0, 0, 0, 1]),
   scalefans=None,
   fansize=1,
   fancover=1.0,
   make_cgo_only=False,
   cyc_ang_match_tol=0.1,
   axislen=2,
   axisrad=0.1,
   addtocgo=None,
   scale=1,
   refpoint=[1, 2, 3, 1],
   **kw,
):
   import pymol

   state["seenit"][name] += 1

   v = pymol.cmd.get_view()

   ang = toshow.angle
   if np.isclose(ang, np.pi * 4 / 5, atol=1e-4): ang /= 2
   if col == 'bycx':
      if False: pass
      elif np.isclose(ang, np.pi * 2 / 2, atol=cyc_ang_match_tol): col = [1, 1, 0]
      elif np.isclose(ang, np.pi * 2 / 3, atol=cyc_ang_match_tol): col = [0, 1, 1]
      elif np.isclose(ang, np.pi * 2 / 4, atol=cyc_ang_match_tol): col = [1, 0, 1]
      elif np.isclose(ang, np.pi * 2 / 5, atol=cyc_ang_match_tol): col = [1, 0, 1]
      elif np.isclose(ang, np.pi * 2 / 6, atol=cyc_ang_match_tol): col = [1, 0, 1]
      else: col = [0.5, 0.5, 0.5]
   elif col == 'random':
      col = np.random.rand(3) / 2 + 0.5

   cen = wu.hscale(scale) @ toshow.cen
   # ic(cen)
   axis = toshow.axis
   c1 = cen + axis * axislen / 2
   c2 = cen - axis * axislen / 2

   mycgo = list()
   mycgo += cgo_cyl(c1, c2, axisrad, col=col)
   # ic(fansize, ang)
   fanthickness = 0.0

   mycgo += cgo_fan(axis, cen, fansize, arc=ang * fancover, thickness=fanthickness, col=col,
                    startpoint=refpoint)

   if addtocgo is None:
      pymol.cmd.load_cgo(mycgo, f'{name}_{state["seenit"][name]}')
      pymol.cmd.set_view(v)
   else:
      addtocgo.extend(mycgo)
   if make_cgo_only:
      return mycgo
   return None

@pymol_load.register(Xtal)
def pymol_viz_Xtal(
      toshow,
      state,
      name='xtal',
      scale=10,
      neighbors=1,
      cellshift=(0, 0, 0),
      showgenframes=False,
      splitobjs=False,
      **kw,
):
   import pymol
   state["seenit"][name] += 1
   name = f'{name}_{state["seenit"][name]}'
   xcellshift = wu.htrans(cellshift)

   allcgo = list()
   # for x in toshow.unitframes:
   # for s in toshow.symelems:
   # pymol_viz_SymElem(wu.hxform(x, s), scale=scale, **kw)
   for i, elems in enumerate(toshow.unitelems):
      cgo = list()
      for elem, xelem in elems:
         refpoint = wu.hxform(xelem, [1, 2, 3, 1])
         refpoint *= scale
         refpoint[3] = 1
         elem = wu.hxform(xcellshift, elem)
         pymol_viz_SymElem(elem, state, scale=scale, addtocgo=cgo, refpoint=refpoint, fansize=0.5,
                           **kw)
      if splitobjs:
         pymol.cmd.load_cgo(cgo, f'{name}_symelem{i}')
      allcgo += cgo

   cgo = list()
   for i, frame in enumerate(toshow.unitframes):
      p1 = [0.18, 0.03, 0.03]
      p2 = [0.18, 0.09, 0.03]
      p3 = [0.18, 0.03, 0.08]
      col = (1, 1, 1)
      frame = xcellshift @ frame
      cgo += cgo_sphere(scale * wu.hxform(frame, p1), rad=scale * 0.05, col=col)
      cgo += cgo_sphere(scale * wu.hxform(frame, p2), rad=scale * 0.03, col=col)
      cgo += cgo_sphere(scale * wu.hxform(frame, p3), rad=scale * 0.02, col=col)
   if splitobjs:
      pymol.cmd.load_cgo(cgo, f'{name}_pts{i}')
   allcgo += cgo

   if showgenframes:
      cgo = list()
      for i, frame in enumerate(toshow.genframes):
         p1 = [0.18, 0.03, 0.03]
         p2 = [0.18, 0.09, 0.03]
         p3 = [0.18, 0.03, 0.08]
         cgo += cgo_sphere(scale * wu.hxform(frame, p1), rad=scale * 0.05, col=col)
         cgo += cgo_sphere(scale * wu.hxform(frame, p2), rad=scale * 0.03, col=col)
         cgo += cgo_sphere(scale * wu.hxform(frame, p3), rad=scale * 0.02, col=col)
      pymol.cmd.load_cgo(cgo, f'{name}_GENPTS{i}')

   xshift2 = xcellshift.copy()
   xshift2[:3, 3] *= scale

   cgo = cgo_cube(wu.hxform(xshift2, [0, 0, 0]), wu.hxform(xshift2, [scale, scale, scale]), r=0.01)
   if splitobjs:
      pymol.cmd.load_cgo(cgo, f'{name}_cube')
   allcgo += cgo

   if not splitobjs:
      pymol.cmd.load_cgo(allcgo, f'{name}_all')
