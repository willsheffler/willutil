import sys, os, tempfile, numpy as np, time, functools
from collections import defaultdict

from logging import info
from functools import singledispatch
#
# from deferred_import import deferred_import
#
# # pymol = deferred_import('pymol')
# # cgo = deferred_import('pymol.cgo')
# # cmd = deferred_import('pymol.cmd')
try:
   import pymol
   from pymol import cgo, cmd
except:
   pass
from willutil import homog as hm
from willutil.viz.pymol_cgo import *
from willutil.sym.symfit import RelXformInfo
import willutil.viz.primitives as prim

_showme_state = wu.Bunch(
   launched=0,
   seenit=defaultdict(lambda: -1),
   _nsymops=0,
)

@singledispatch
def pymol_load(
   toshow,
   # state=_showme_state,
   # name=None,
   **kw,
):
   raise NotImplementedError("pymol_load: don't know how to show " + str(type(toshow)))

@pymol_load.register(prim.Cylinder)
def _(
   toshow,
   state=_showme_state,
   # name=None,
   addtocgo=None,
   make_cgo_only=False,
   **kw,
):
   v = pymol.cmd.get_view()
   mycgo = cgo_cyl(toshow.start, toshow.end, toshow.radius, col=toshow.color)

   if addtocgo is None:

      pymol.cmd.load_cgo(mycgo, 'cyl%i' % state._nsymops)
      pymol.cmd.set_view(v)
   else:
      addtocgo.extend(mycgo)

   if make_cgo_only:
      return mycgo

   pymol.cmd.load_cgo(mycgo, 'cyl%i' % state._nsymops)
   pymol.cmd.set_view(v)
   return None

@pymol_load.register(RelXformInfo)
def _(
   toshow,
   state=_showme_state,
   col='bycx',
   name='xrel',
   showframes=True,
   center=np.array([0, 0, 0, 1]),
   scalefans=None,
   fixedfansize=1,
   expand=1.0,
   fuzz=0,
   make_cgo_only=False,
   cyc_ang_match_tol=0.1,
   axislen=20,
   usefitaxis=False,
   axisrad=0.1,
   helicalrad=None,
   addtocgo=None,
   **kw,
):

   state._nsymops += 1
   v = pymol.cmd.get_view()

   ang = toshow.ang
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
      # col = (1, 1, 1)
   # cen = (toshow.framecen - center) * expand + center
   cen = toshow.framecen

   mycgo = list()

   if showframes:
      cgo = pymol_visualize_xforms(toshow.frames, state, make_cgo_only=True, **kw)
      mycgo += cgo

   cen1 = toshow.frames[0, :, 3]
   cen2 = toshow.frames[1, :, 3]
   axis = toshow.cenaxis if usefitaxis else toshow.axs

   if abs(toshow.ang) < 1e-6:
      mycgo += cgo_cyl(cen1, cen2, 0.01, col=(1, 0, 0))
      mycgo += cgo_sphere(cen=(cen1 + cen2) / 2, rad=0.1, col=(1, 1, 1))
   else:
      c1 = cen + axis * axislen / 2
      c2 = cen - axis * axislen / 2
      if 'isect_sphere' in toshow:
         mycgo += cgo_sphere(toshow.closest_to_cen, rad=0.2, col=col)
         mycgo += cgo_sphere(toshow.isect_sphere, rad=0.2, col=col)
         mycgo += cgo_fan(axis, toshow.isect_sphere, rad=0.5, arc=2 * np.pi, col=col)
         c1 = cen
         c2 = toshow.isect_sphere

      helicalrad = helicalrad or 3 * axisrad

      mycgo += cgo_cyl(c1, c2, axisrad, col=col)
      mycgo += cgo_cyl(cen + axis * toshow.hel / 2, cen - axis * toshow.hel / 2, helicalrad, col=col)
      shift = fuzz * (np.random.rand() - 0.5)
      mycgo += cgo_fan(axis, cen + axis * shift, fixedfansize if fixedfansize else toshow.rad * scalefans, arc=ang,
                       col=col, startpoint=cen1)

   if addtocgo is None:
      pymol.cmd.load_cgo(mycgo, 'symops%i' % state._nsymops)
      pymol.cmd.set_view(v)
   else:
      addtocgo.extend(mycgo)

   if make_cgo_only:
      return mycgo
   return None

@pymol_load.register(dict)
def _(
   toshow,
   state=_showme_state,
   name='stuff_in_dict',
   **kw,
):
   if "pose" in toshow:
      pymol_load(toshow["pose"], state, **kw)
      pymol_xform(toshow["position"], state["last_obj"])
   else:
      mycgo = list()
      for k, v in toshow.items():
         cgo = pymol_load(v, state, name=k, make_cgo_only=True, **kw)
         mycgo += cgo
      pymol.cmd.load_cgo(mycgo, name)

@pymol_load.register(list)
def _(
   toshow,
   state=_showme_state,
   name=None,
   **kw,
):
   mycgo = list()
   v = pymol.cmd.get_view()
   cmd.set('suspend_updates', 'on')
   col = [tuple(_) for _ in get_different_colors(len(toshow), **kw)]
   for i, t in enumerate(toshow):
      # print('    ##############', type(t), '################')
      mycgo.extend(pymol_load(t, state, addtocgo=mycgo, make_cgo_only=True, col=col[i], **kw))
   pymol.cmd.set_view(v)
   name = name or 'list'
   pymol.cmd.load_cgo(mycgo, name)
   cmd.set('suspend_updates', 'off')

@pymol_load.register(np.ndarray)
def _(
   toshow,
   state=_showme_state,
   line_strip=False,
   **kw,
):
   # showaxes()
   shape = toshow.shape
   if line_strip:
      return show_ndarray_line_strip(toshow, state, **kw)
   if shape[-2:] in [(3, 4), (5, 4)]:
      return show_ndarray_n_ca_c(toshow[:, :3], state, **kw)
   elif shape[-2:] == (4, 2):
      return show_ndarray_lines(toshow, state, **kw)
   elif len(shape) > 2 and shape[-2:] == (4, 4):
      return pymol_visualize_xforms(toshow, state, **kw)
   elif shape == (4, ) or len(shape) == 2 and shape[-1] == 4:
      return show_ndarray_point_or_vec(toshow, state, **kw)
   else:
      raise NotImplementedError(f'cant understand np.ndarray type {type(toshow)} shape {toshow.shape}')

_nxforms = 0

@functools.lru_cache
def get_different_colors(ncol, niter=1000, colorseed=1, **kw):
   rs = np.random.get_state()
   np.random.seed(colorseed)
   maxmincoldis, best = 0, None
   for i in range(niter):
      colors = np.random.rand(ncol, 3)
      # ic(np.max(colors[:, ], axis=1))
      while np.min(np.max(colors[:, ], axis=1)) < 0.8:
         colors = np.random.rand(ncol, 3)
      cdis2 = np.linalg.norm(colors[None] - colors[:, None], axis=-1)
      np.fill_diagonal(cdis2, 4.0)
      if np.min(cdis2) > maxmincoldis:
         maxmincoldis, best = np.min(cdis2), colors

   best = best[np.argsort(-np.min(best**2, axis=1))]
   # ic(np.argsort(-np.min(best**2, axis=1)))
   # ic(best)
   # assert 0
   np.random.set_state(rs)
   # paranoid sanity cheeck
   newrs = np.random.get_state()
   assert rs[0] == newrs[0] and np.all(rs[1] == newrs[1]) and rs[2:] == newrs[2:]
   return best

def get_cgo_name(name):
   names = pymol.cmd.get_names()
   if not name in names:
      return name
   i = 0
   while name + str(i) in names:
      i += 1
   return name + str(i)

   # if not 'seenit' in _showme_state:
   # _showme_state['seenit'] = defaultdict(int)
   # if name in _showme_state['seenit']:
   # _showme_state['seenit'][name] += 1
   # return name + '_%i' % _showme_state['seenit']
   # return name

def pymol_visualize_xforms(
   xforms,
   state=_showme_state,
   name='xforms',
   randpos=0.0,
   xyzlen=[5 / 4, 1, 4 / 5],
   xyzscale=1.0,
   scale=1.0,
   weight=1.0,
   spheres=0,
   make_cgo_only=False,
   center_weight=1.0,
   center=None,
   rays=0,
   framecolors=None,
   perturb=0,
   addtocgo=None,
   **kw,
):
   if perturb != 0: raise NotImplementedError

   xyzlen = [_ * xyzscale for _ in xyzlen]
   origlen = xforms.shape[0] if xforms.ndim > 3 else 1
   xforms = xforms.reshape(-1, 4, 4)
   global _nxforms
   _nxforms += 1

   if xforms.shape == (4, 4):
      xforms = xforms.reshape(1, 4, 4)

   name = get_cgo_name(name)
   colors = None
   if framecolors == 'rand':
      colors = get_different_colors(len(xforms) + 1, **kw)

   # mycgo = [cgo.BEGIN]
   mycgo = list()

   c0 = [0, 0, 0, 1]
   x0 = [xyzlen[0], 0, 0, 1]
   y0 = [0, xyzlen[1], 0, 1]
   z0 = [0, 0, xyzlen[2], 1]
   if randpos > 0:
      rr = wu.homog.rand_xform()
      rr[:3, 3] *= randpos
      c0 = rr @ c0
      x0 = rr @ x0
      y0 = rr @ y0
      z0 = rr @ z0

   for ix, xform in enumerate(xforms):
      xform[:3, 3] *= scale
      cen = xform @ c0
      x = xform @ x0
      y = xform @ y0
      z = xform @ z0
      color = framecolors if colors is None else colors[ix]
      col1 = [1, 0, 0] if color is None else color
      col2 = [0, 1, 0] if color is None else color
      col3 = [0, 0, 1] if color is None else color
      col4 = [1, 0.5, 0.5] if color is None else color
      mycgo.extend(cgo_cyl(cen, x, 0.05 * weight, col1))
      mycgo.extend(cgo_cyl(cen, y, 0.05 * weight, col2))
      mycgo.extend(cgo_cyl(cen, z, 0.05 * weight, col3))
      if spheres > 0:  # and ix % origlen == 0:
         mycgo.extend(cgo_sphere(cen, spheres, col=col4))

   if center is not None:
      color = col if colors is None else colors[-1]
      col1 = [1, 1, 1] if color is None else color
      mycgo += cgo_sphere(center, center_weight, col=col1)
   if rays > 0:
      center = np.mean(xforms[:, :, 3], axis=0) if center is None else center
      color = col if colors is None else colors[-1]
      col1 = [1, 1, 1] if color is None else color
      for ix, xform in enumerate(xforms):
         mycgo += cgo_cyl(center, xform[:, 3], rays, col=col1)

   if addtocgo is None:
      pymol.cmd.load_cgo(mycgo, name)
      pymol.cmd.zoom()
   else:
      addtocgo.extend(cgo)

   if make_cgo_only:
      return mycgo
   return None

def show_ndarray_lines(
   toshow,
   state=_showme_state,
   name=None,
   col=None,
   scale=100,
   bothsides=False,
   spheres=3,
   cyl=0,
   addtocgo=None,
   **kw,
):
   name = name or "ndarray_lines"
   state["seenit"][name] += 1
   name += "_%i" % state["seenit"][name]
   if col == 'rand':
      col = get_different_colors(len(toshow), **kw)
   if not isinstance(col[0], (list, np.ndarray, tuple)):
      col = [col] * len(toshow)

   assert toshow.shape[-2:] == (4, 2)
   toshow = toshow.reshape(-1, 4, 2)

   cgo = list()
   for i, ray in enumerate(toshow):
      color = col[i] if col else (1, 1, 1)
      if cyl:
         cgo.extend(cgo_cyl(ray[:3, 0] + ray[:3, 1], ray[:3, 0], rad=cyl, col=color), )
      else:
         cgo.extend(cgo_lineabs(ray[:3, 0] + ray[:3, 1], ray[:3, 0], col=color), )
      if spheres:
         cgo.extend(cgo_sphere(ray[:3, 0], col=color, rad=spheres))
         if bothsides:
            cgo.extend(cgo_sphere(ray[:3, 0] + ray[:3, 1], col=color, rad=spheres))
   if addtocgo is None:
      cmd.load_cgo(cgo, name + '_%i' % i)
   else:
      addtocgo.extend(cgo)

def show_ndarray_line_strip(
   toshow,
   state=_showme_state,
   name='lines',
   col=[1, 1, 1],
   stateno=1,
   linewidth=1,
   breaks=1,
   breaks_groups=1,
   whitetopn=0,
   addtocgo=None,
   **kw,
):
   # v = pymol.cmd.get_view()
   # print('!!!!!!!!!!!!!!!!!!!!!!!!', state)
   state["seenit"][name] += 1
   name += "_%i" % state["seenit"][name]

   if col == 'rand':
      col = get_different_colors(breaks // breaks_groups, **kw)
   if isinstance(col, list) and isinstance(col[0], (int, float)):
      col = [col] * breaks
   if isinstance(col, (tuple, str)):
      col = [col] * breaks
   col[:whitetopn] = [(1, 1, 1)] * whitetopn
   # print('-' * 10)
   # print(col)
   # print('-' * 10)

   mycgo = list()
   assert toshow.shape[-1] == 4
   toshow = toshow.reshape(-1, 4)

   toshow[:, 3] = cgo.VERTEX
   toshow = toshow.reshape(-1)
   n = len(toshow) // breaks
   nextra, nheader = 16, 3
   cgoary = np.empty((n + nextra) * breaks + 3)
   # print(f'len cgoary {len(cgoary)} {breaks*n} breaks {breaks}')
   cgoary[0] = cgo.BEGIN
   cgoary[1] = cgo.LINE_STRIP
   cgoary[2] = cgo.VERTEX
   for i in range(breaks):
      lb0 = (i + 0) * (n + nextra) + nheader + 4
      ub0 = (i + 1) * (n + nextra) + nheader + 4 - nextra
      lb1 = (i + 0) * n
      ub1 = (i + 1) * n
      cgoary[lb0 - 5:lb0 - 4] = cgo.COLOR
      cgoary[lb0 - 4:lb0 - 1] = col[i // breaks_groups]
      cgoary[lb0 - 1:lb0] = cgo.VERTEX
      cgoary[lb0:ub0] = toshow[lb1:ub1].reshape(-1)
      # color block gaps
      cgoary[ub0 - 1:ub0 + 4] = cgo.COLOR, 0, 0, 0, cgo.VERTEX
      cgoary[ub0 + 4:ub0 + 8] = toshow[ub1 - 4:ub1]
      ub1 = ub1 if ub1 < len(toshow) else 0
      cgoary[ub0 + 8:ub0 + 12] = toshow[ub1:ub1 + 4]
   cgoary[-1] = cgo.END  # overrwites last VERTEX op

   pymol.cmd.set('cgo_line_width', linewidth)

   if addtocgo is None:
      pymol.cmd.load_cgo(cgoary, name, -1)
   else:
      addtocgo.extend(cgoary)

   # print(stateno, flush=True)
   # pymol.cmd.load_cgo(cgoary, name, state=stateno)
   # pymol.cmd.set_view(v)

def show_ndarray_point_or_vec(
   toshow,
   state=_showme_state,
   name='points',
   col=None,
   sphere=1.0,
   addtocgo=None,
   **kw,
):
   v = pymol.cmd.get_view()
   state["seenit"][name] += 1
   name += "_%i" % state["seenit"][name]
   if col in get_color_dict():
      col = get_color_dict()[col]
   if col == 'rand':
      col = get_different_colors(len(toshow), **kw)
   mycgo = list()
   assert toshow.shape[-1] == 4
   if toshow.ndim == 1: toshow = [toshow]
   for i, p_or_v in enumerate(toshow):
      color = (1, 1, 1) if col is None else col
      if isinstance(color[0], (list, tuple, np.ndarray)):
         color = color[i]
      if p_or_v[3] > 0.999:
         mycgo += cgo_sphere(p_or_v, sphere, col=color)
      elif np.abs(p_or_v[3]) < 0.001:
         mycgo += cgo_vecfrompoint(p_or_v * 20, p_or_v, col=color)
      else:
         raise NotImplementedError

   if addtocgo is None:
      pymol.cmd.load_cgo(mycgo, name)
      pymol.cmd.set_view(v)
   else:
      addtocgo.extend(mycgo)

def show_ndarray_n_ca_c(
   toshow,
   state=_showme_state,
   name=None,
   col=[1, 1, 1],
   **kw,
):
   name = name or "ndarray_n_ca_c"
   state["seenit"][name] += 1
   name += "_%i" % state["seenit"][name]

   tmpdir = tempfile.mkdtemp()
   fname = tmpdir + "/" + name + ".pdb"
   assert toshow.shape[-2:] == (3, 4)
   with open(fname, "w") as out:
      for i, a1 in enumerate(toshow.reshape(-1, 3, 4)):
         for j, a in enumerate(a1):
            line = format_atom(
               atomi=3 * i + j,
               resn="GLY",
               resi=i,
               atomn=(" N  ", " CA ", " C  ")[j],
               x=a[0],
               y=a[1],
               z=a[2],
            )
            out.write(line)
   pymol.cmd.load(fname)

def showme_pymol(
   what,
   headless=False,
   block=False,
   fresh=False,
   png=None,
   pngturn=0,
   ray=True,
   one_png_only=False,
   **kw,
):
   global _showme_state
   if "PYTEST_CURRENT_TEST" in os.environ and not headless:
      print("NOT RUNNING PYMOL IN UNIT TEST")
      return

   pymol.pymol_argv = ["pymol"]
   if headless: pymol.pymol_argv = ["pymol", "-c"]
   if not _showme_state["launched"]:

      pymol.finish_launching()
      # pymol.cmd.viewport(400, 720)
      # v = list(cmd.get_view())
      # v[15] *= 2
      # cmd.set_view(v)
      # print(repr(v))
      # assert 0
      _showme_state["launched"] = 1

      # cmd.turn('x', -90)
      # cmd.turn('y', 100)

   # print('############## showme_pymol', type(what), '##############')

   if fresh:
      clear_pymol()

   # pymol.cmd.full_screen('on')
   result = pymol_load(what, state=_showme_state, **kw)
   # # pymol.cmd.set('internal_gui_width', '20')

   if png:
      pymol.cmd.set('ray_opaque_background', 1)
      if os.path.dirname(png):
         os.makedirs(os.path.dirname(png), exist_ok=True)
      pymol.cmd.turn('y', pngturn)
      if ray:
         pymol.cmd.ray()
      pymol.cmd.png(png)
      if one_png_only:
         assert 0

   while block:
      time.sleep(1)
   return result

def clear_pymol():
   pymol.cmd.delete('not axes')

def showme(*args, how="pymol", **kw):
   randstate = np.random.get_state()
   if how == "pymol":
      result = showme_pymol(*args, **kw)
   else:
      result = NotImplemented('showme how="%s" not implemented' % how)
   np.random.set_state(randstate)
   return result

_atom_record_format = ("ATOM  {atomi:5d} {atomn:^4}{idx:^1}{resn:3s} {chain:1}{resi:4d}{insert:1s}   "
                       "{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{b:6.2f}\n")

def format_atom(
   atomi=0,
   atomn="ATOM",
   idx=" ",
   resn="RES",
   chain="A",
   resi=0,
   insert=" ",
   x=0,
   y=0,
   z=0,
   occ=0,
   b=0,
):
   return _atom_record_format.format(**locals())

def pymol_xform(name, xform):
   assert name in pymol.cmd.get_object_list()
   pymol.cmd.transform_object(name, xform.flatten())

def get_palette(kind='default', rgb=True, blacklist=['white', 'black']):
   if kind is None:
      kind = 'default'

   palette = list()
   if kind == 'default':
      for col, idx in cmd.get_color_indices():
         if col in blacklist: continue
         if rgb: col = cmd.get_color_tuple(idx)
         palette.append(col)

   elif isinstance(kind, list):
      palette = kind  # pass through for convenience

   else:
      raise ValueError(f'unknown palette kind {kind}')

   return palette

def get_color_dict():
   colors = {col: cmd.get_color_tuple(idx) for col, idx in cmd.get_color_indices()}
   return colors
