import functools
import numpy as np
import willutil as wu

def tooclose_clash(bodies, nbrs=None, **kw):
   return bodies.clashes(nbrs)

def tooclose_overlap(bodies, nbrs=None, contactfrac=0.1, **kw):
   cfrac = bodies.contact_fraction(nbrs)
   ic(cfrac)
   maxcfrac = max([np.mean(c) for c in cfrac])
   return maxcfrac > contactfrac

def asuslide(
   sym,
   coords,
   frames=None,
   axes=None,
   existing_olig=None,
   alongaxis=None,
   towardaxis=None,
   printme=False,
   cellsize=1,
   isxtal=False,
   nbrs='auto',
   doscale=True,
   iters=5,
   clashiters=5,
   step=10,
   **kw,
):
   kw = wu.Bunch(kw)
   cooords = wu.hpoint(coords)
   if axes is None:
      axes = wu.sym.axes(sym, cellsize=cellsize)
      # ic(axes)
      if isinstance(axes, dict):
         axes = [(ax, wu.hpoint([0, 0, 0])) for ax in axes.values()]
      else:
         isxtal = True
         axes = [(elem.axis, elem.cen) for elem in axes]
         doscale = True if doscale is None else doscale
      com = wu.hcom(coords)
      faxdist = lambda ac: wu.hpointlinedis(com, ac[1], wu.hscaled(cellsize, ac[0]))
      ic([faxdist(ac) for ac in axes])
      axes = list(sorted(axes, key=faxdist))

   # ic(alongaxis, towardaxis)
   if alongaxis is None and towardaxis is None:
      alongaxis = not isxtal
      towardaxis = isxtal
   if alongaxis is True and towardaxis is None: towardaxis = False
   if alongaxis is False and towardaxis is None: towardaxis = True
   if towardaxis is True and alongaxis is None: alongaxis = False
   if towardaxis is False and alongaxis is None: alongaxis = True
   # ic(alongaxis, towardaxis)

   if frames is None:
      frames = wu.sym.frames(sym, cellsize=cellsize)

   # assert towardaxis
   # assert not alongaxis
   clashfunc = tooclose_clash
   userfunc = functools.partial(kw.get('tooclose', tooclose_overlap), **kw)

   bodies = wu.rigid.RigidBodyFollowers(coords=coords, frames=frames, recenter=True, cellsize=cellsize, **kw)
   cellsize0 = cellsize
   if doscale and not alongaxis: cellsize = slide_scale(bodies, cellsize, step=step, **kw)
   for i in range(iters):
      for axis, axpos in axes:
         axis = wu.hnormalized(axis)
         axpos = wu.hscaled(cellsize / cellsize0, wu.hpoint(axpos))
         if towardaxis:
            # ic(axis, cellsize, bodies.asym.com(), axpos, bodies.asym.com() - axpos)
            axisperp = wu.hnormalized(wu.hprojperp(axis, bodies.asym.com() - axpos))  # points away from axis
            # ic(axisperp)
            # for a, c in axes:
            # ic(a, wu.hangle(a, axisperp))
            slide = slide_axis(axisperp, bodies, perp=True, nbrs=nbrs, step=step, **kw)
            if printme: ic(f'slide along {axisperp[:3]} by {slide}')
         if alongaxis:
            slide = slide_axis(axis, bodies, nbrs=nbrs, step=step, **kw)
            printme: ic(f'slide along {axis[:3]} by {slide}')
         if doscale and alongaxis: slide = slide_axis(bodies.asym.com(), bodies, nbrs=None, step=step, **kw)
         elif doscale: cellsize = slide_scale(bodies, cellsize, step=step, **kw)

      step *= 0.6
      if i >= clashiters:
         kw.tooclose = userfunc

   return bodies

def slide_axis(
   axis,
   bodies,
   nbrs='auto',
   tooclose=tooclose_clash,
   perp=False,
   step=1.0,
   maxstep=100,
   showme=False,
   **kw,
):
   axis = wu.hnormalized(axis)
   origpos = bodies.asym.position
   if nbrs == 'auto': nbrs = bodies.get_neighbors_by_axismatch(axis, perp)
   elif nbrs == 'all': nbrs = None
   # ic('SLIDE', perp, axis, nbrs)

   iflip, flip = 0, -1.0
   if tooclose(bodies, nbrs):
      # ic('REVERSE', axis, nbrs)
      iflip, flip = -1, 1.0
   # else:
   # ic('FORWARD', axis, nbrs)
   total = 0.0
   delta = wu.htrans(flip * step * axis)
   # ic(delta[:3, 3])

   for i in range(maxstep):
      bodies.asym.moveby(delta)
      total += flip * step
      if showme: wu.showme(bodies, name='slideaxis' % axis[0], **kw)
      close = tooclose(bodies, nbrs)
      if iflip + close:
         break
   else:
      bodies.asym.position = origpos
      if showme: wu.showme(bodies, name='resetaxis%f' % axis[0], **kw)
      return 0
   if iflip == 0:  # back off so no clash
      bodies.asym.moveby(wu.hinv(delta))
      if showme: wu.showme(bodies, name='backoffaxis%f' % axis[0], **kw)

   # assert 0
   return total

def slide_scale(
   bodies,
   cellsize,
   tooclose=tooclose_clash,
   step=1.0,
   maxstep=30,
   showme=False,
   **kw,
):
   # ic('SCALE')

   iflip, flip = 0, -1.0
   if tooclose(bodies): iflip, flip = -1, 1.0
   for i in range(maxstep):
      if iflip + tooclose(bodies): break

      delta = (cellsize + flip * step) / cellsize
      cellsize *= delta
      # ic(delta, cellsize)

      changed = bodies.scale_frames(delta, safe=False)
      if not changed:
         assert i == 0
         return cellsize

      newpos = bodies.asym.position
      newpos[:3, 3] *= delta
      bodies.asym.position = newpos
      if showme: wu.showme(bodies, name='scale', **kw)

   if iflip == 0:  # back off
      delta = 1.0 / delta
      bodies.scale_frames(delta, safe=False)
      newpos = bodies.asym.position
      newpos[:3, 3] *= delta
      bodies.asym.position = newpos
      cellsize *= delta

      if showme: wu.showme(bodies, 'backoffscale', **kw)

   return cellsize