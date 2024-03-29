import numpy as np
from willutil.sym.xtal import Xtal
from willutil.sym.xtalinfo import SymElem
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
   fansize=0.001,
   fanshift=0,
   fancover=1.0,
   make_cgo_only=False,
   cyc_ang_match_tol=0.1,
   axislen=0.2,
   axisrad=0.008,
   addtocgo=None,
   scale=1,
   fanrefpoint=[1, 2, 3, 1],
   symelemscale=1,
   **kw,
):
   import pymol

   state["seenit"][name] += 1

   v = pymol.cmd.get_view()

   axislen = axislen * scale * symelemscale
   axisrad = axisrad * scale * symelemscale
   fanthickness = 0.0 * scale * symelemscale
   fansize = fansize * scale * symelemscale
   fanshift = fanshift * scale * symelemscale

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

   mycgo += cgo_fan(axis, cen, fansize, arc=ang * fancover, thickness=fanthickness, col=col, startpoint=fanrefpoint,
                    fanshift=fanshift)

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
   cells=1,
   showgenframes=False,
   splitobjs=False,
   showpoints=0,
   fanshift=0,
   fansize=0.1,
   showcube=True,
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
      size = fansize[i] if isinstance(fansize, (list, tuple)) else fansize
      shift = fanshift[i] if isinstance(fanshift, (list, tuple)) else fanshift
      for elem, xelem in elems:
         fanrefpoint = get_fanrefpoint(toshow)
         fanrefpoint = wu.hxform(xelem, fanrefpoint)
         fanrefpoint = xcellshift @ fanrefpoint
         fanrefpoint = wu.hscale(scale) @ fanrefpoint
         # cgo += cgo_sphere(fanrefpoint, 0.5, col=(1, 1, 1))
         elem = wu.hxform(xcellshift, elem)
         pymol_viz_SymElem(elem, state, scale=scale, addtocgo=cgo, fanrefpoint=fanrefpoint, fansize=fansize,
                           fanshift=fanshift, **kw)
      if splitobjs:
         pymol.cmd.load_cgo(cgo, f'{name}_symelem{i}')
      allcgo += cgo
   xshift2 = xcellshift.copy()
   xshift2[:3, 3] *= scale
   if showcube:
      cgo = cgo_cube(wu.hxform(xshift2, [0, 0, 0]), wu.hxform(xshift2, [scale, scale, scale]), r=0.03)
      if splitobjs:
         pymol.cmd.load_cgo(cgo, f'{name}_cube')
   allcgo += cgo

   # for i, (elem, frame) in enumerate(toshow.unitelems[1]):

   showpts = xtal_show_points(showpoints, **kw)
   if showpoints:
      frames = toshow.cellframes(cellsize=1, cells=cells)
      cgo = cgo_frame_points(frames, scale, showpts)
      # cgo = list()
      # for i, frame in enumerate(frames):
      # for p, r, c in zip(*showpts):
      # cgo += cgo_sphere(scale * wu.hxform(frame, p), rad=scale * r, col=c)
      if splitobjs:
         pymol.cmd.load_cgo(cgo, f'{name}_pts{i}')
      allcgo += cgo

   if showgenframes:
      col = (1, 1, 1)
      cgo = cgo_frame_points(toshow.genframes, scale, showpts)
      # cgo = list()
      # for i, frame in enumerate(toshow.genframes):
      # cgo += cgo_sphere(scale * wu.hxform(frame, showpts[0]), rad=scale * 0.05, col=col)
      # cgo += cgo_sphere(scale * wu.hxform(frame, showpts[1]), rad=scale * 0.03, col=col)
      # cgo += cgo_sphere(scale * wu.hxform(frame, showpts[2]), rad=scale * 0.02, col=col)
      pymol.cmd.load_cgo(cgo, f'{name}_GENPTS{i}')

   if not splitobjs:
      pymol.cmd.load_cgo(allcgo, f'{name}_all')

   return state

def cgo_frame_points(frames, scale, showpts):
   cgo = list()
   for i, frame in enumerate(frames):
      for p, r, c in zip(*showpts):
         cgo += cgo_sphere(scale * wu.hxform(frame, p), rad=scale * r, col=c)
   return cgo

def xtal_show_points(which, pointscale=1, pointshift=(0, 0, 0), **kw):
   s = pointscale
   pointshift = np.asarray(pointshift)
   showpts = [
      np.empty(shape=(0, 3)),
      np.array([
         [0.28, 0.13, 0.13],
         [0.28, 0.13 + 0.06 * s, 0.13],
         [0.28, 0.13, 0.13 + 0.05 * s],
      ]),
      np.array([
         [0.18, 0.03, 0.03],
         [0.18, 0.03 + 0.06 * s, 0.03],
         [0.18, 0.03, 0.03 + 0.05 * s],
      ]),
   ]
   for p in showpts:
      p += pointshift

   radius = np.array([[0.05, 0.03, 0.02]] * len(showpts))
   radius *= pointscale
   colors = np.array([[(1, 1, 1), (1, 1, 1), (1, 1, 1)]] * len(showpts))
   return showpts[which], radius[which], colors[which]

def get_fanrefpoint(xtal):
   pt = [0, 1, 0, 1]
   # yapf: disable
   if xtal.name == 'P 2 3' : pt= [0, 1, 0, 1]
   if xtal.name == 'I 21 3': pt= wu.hxform(wu.hrot([0, 0, 1], -30), [0, 1, 0,1])
   # yapf: enable
   # ic(pt)
   return pt