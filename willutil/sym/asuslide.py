import numpy as np
import willutil as wu

def asuslide(
   sym,
   coords,
   frames=None,
   axes=None,
   existing_olig=None,
   alongaxis=None,
   towardaxis=None,
   iters=1,
   printme=False,
   cellsize=1,
   isxtal=False,
   nbrs='auto',
   doscale=False,
   **kw,
):
   cooords = wu.hpoint(coords)
   if axes is None:
      axes = wu.sym.axes(sym, cellsize=cellsize)
      ic(axes)
      if isinstance(axes, dict):
         axes = [(ax, wu.hpoint([0, 0, 0])) for ax in axes.values()]
      else:
         isxtal = True
         axes = [(elem.axis, elem.cen) for elem in axes]
      com = wu.hcom(coords)
      axes = list(sorted(axes, key=lambda ac: wu.hpointlinedis(com, ac[1], ac[0])))

   ic(alongaxis, towardaxis)
   if alongaxis is None and towardaxis is None:
      alongaxis = not isxtal
      towardaxis = isxtal
   if alongaxis is True and towardaxis is None: towardaxis = False
   if alongaxis is False and towardaxis is None: towardaxis = True
   if towardaxis is True and alongaxis is None: alongaxis = False
   if towardaxis is False and alongaxis is None: alongaxis = True
   ic(alongaxis, towardaxis)

   if frames is None:
      frames = wu.sym.frames(sym, cellsize=cellsize)

   # assert towardaxis
   # assert not alongaxis

   scaling = 1.0
   bodies = wu.rigid.RigidBodyFollowers(coords=coords, frames=frames, recenter=True, cellsize=cellsize, **kw)
   if doscale: scaling *= slide_scale(bodies, **kw)
   for i in range(iters):
      for axis, pos in reversed(axes):
         if towardaxis:
            axisperp = wu.hprojperp(axis, bodies.asym.com() - wu.hpoint(pos))
            slide = slide_axis(axisperp, bodies, perp=True, nbrs=nbrs, **kw)
            if printme: ic(f'slide along {axisperp[:3]} by {slide}')
         if alongaxis:
            slide = slide_axis(axis, bodies, nbrs=nbrs, **kw)
            printme: ic(f'slide along {axis[:3]} by {slide}')
         if doscale: scaling *= slide_scale(bodies, **kw)
         if doscale: printme: ic(f'scale by {scaling}')
   return bodies

def slide_axis(axis, bodies, nbrs='auto', contactfrac=None, perp=False, step=1.0, maxstep=100, showme=False, **kw):
   axis = wu.hnormalized(axis)
   idirn, dirn = 0, -1.0
   total = 0.0

   if nbrs == 'auto': nbrs = bodies.get_neighbors_by_axismatch(axis, perp)
   elif nbrs == 'all': nbrs = None

   if bodies.clashes(nbrs):
      idirn, dirn = -1, 1.0
   delta = wu.htrans(dirn * step * axis)
   for i in range(maxstep):
      if idirn + bodies.clashes(nbrs):
         break
      bodies.asym.moveby(delta)
      total += dirn * step
      if showme: wu.showme(bodies, name='slideaxis' % axis[0], **kw)
   else:
      bodies.asym.moveby(wu.htrans(-total))
      if showme: wu.showme(bodies, name='backoffaxis%f' % axis[0], **kw)
      return 0
   if idirn == 0:  # back off so no clash
      bodies.asym.moveby(wu.hinv(delta))
   return total

def slide_scale(bodies, nbrs='auto', contactfrac=None, step=1.0, maxstep=100, showme=False, **kw):
   begscale = wu.hnorm(bodies.asym.position[:, 3])
   scale = begscale
   assert begscale > 0.001
   idirn, dirn = 0, -1.0
   if bodies.clashes(nbrs):
      idirn, dirn = -1, 1.0
   for i in range(maxstep):
      if idirn + bodies.clashes(nbrs):
         break

      delta = (scale + dirn * step) / scale
      changed = bodies.scale_frames(delta, safe=False)
      if not changed:
         assert i == 0
         return 1.0
      if showme: wu.showme(bodies, name='scale', **kw)
   if idirn == 0:  # back off
      bodies.scale_frames(1.0 / delta, safe=False)
      if showme: wu.showme(bodies, 'backoffscale', **kw)
   endscale = wu.hnorm(bodies.asym.position[:, 3])
   return endscale / begscale