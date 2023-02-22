import functools
import numpy as np
import willutil as wu
from willutil.rigid.objective import tooclose_clash, tooclose_overlap

def asuslide(
   sym,
   coords,
   frames=None,
   axes=None,
   existing_olig=None,
   alongaxis=None,
   towardaxis=None,
   boundscheck=lambda x: True,
   printme=False,
   cellsize=1,
   isxtal=False,
   nbrs='auto',
   doscale=True,
   iters=5,
   subiters=1,
   clashiters=5,
   receniters=2,
   step=10,
   scalestep=None,
   closestfirst=True,
   centerasu='toward_partner',
   centerasu_at_start=False,
   showme=False,
   scaleslides=1.0,
   iterstepscale=0.75,
   coords_to_asucen=False,
   along_extra_axes=[],
   xtalrad=0.5,
   **kw,
):
   if isinstance(cellsize, (int, float)): cellsize = [float(cellsize)] * 3
   if not isinstance(cellsize, np.ndarray): cellsize = np.array(cellsize, dtype=np.float64)
   if printme:
      coordstr = repr(coords).replace(' ', '').replace('\n', '').replace('\t', '').replace('float32', 'np.float32')
      framestr = repr(frames).replace(' ', '').replace('\n', '').replace('\t', '')
      print(f'''kw = {kw}
      coords=np.{coordstr}
      frames=np.{framestr}
      asuslide(sym='{sym}',coords=coords,frames=frames,axes={axes},existing_olig={existing_olig},alongaxis={alongaxis},towardaxis={towardaxis},printme=False,cellsize={repr(cellsize)},isxtal={isxtal},nbrs={repr(nbrs)},doscale={doscale},iters={iters},subiters={subiters},clashiters={clashiters},receniters={receniters},step={step},scalestep={scalestep},closestfirst={closestfirst},centerasu={repr(centerasu)},centerasu_at_start={centerasu_at_start},showme={showme},scaleslides={scaleslides},iterstepscale={iterstepscale},coords_to_asucen={coords_to_asucen},**kw)'''
            )
   kw = wu.Bunch(kw)
   kw.showme = showme
   kw.boundscheck = boundscheck
   kw.scaleslides = scaleslides
   coords = wu.hpoint(coords).copy()
   coords = coords.reshape(-1, 4)
   symelems_siblings = []
   if scalestep is None: scalestep = step
   if axes is None:
      axes0 = wu.sym.axes(sym, cellsize=cellsize)
      # ic(axes)
      if isinstance(axes0, dict):
         axes = [(ax, wu.hpoint([0, 0, 0])) for ax in axes0.values()]
      else:
         isxtal = True
         axes = [(elem.axis, elem.cen) for elem in axes0]
         doscale = True if doscale is None else doscale
         symelems_siblings = wu.sym.symelem_associations(axes0)
      com = wu.hcom(coords)
      faxdist = [wu.hpointlinedis(com, ac[1], wu.hscaled(cellsize, ac[0])) for ac in axes]
      faxorder = np.argsort(faxdist)
      # ic([faxdist(ac) for ac in axes])
      axes = [axes[i] for i in faxorder]
      if symelems_siblings: symelems_siblings = [symelems_siblings[i] for i in faxorder]
      if not closestfirst:
         axes = list(reversed(axes))
         symelems_siblings = list(reversed(symelems_siblings))
   symelems_siblings = symelems_siblings or ['auto'] * len(axes)
   # ic(symelems_siblings)

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
      frames = wu.sym.frames(sym, cellsize=cellsize, xtalrad=xtalrad, allowcellshift=True, **kw)

   # assert towardaxis
   # assert not alongaxis
   # clashfunc = tooclose_clash
   # kw.tooclosefunc = clashfunc
   # userfunc = functools.partial(kw.get('tooclosefunc', tooclose_overlap), printme=printme, **kw)
   kw.tooclosefunc = functools.partial(kw.get('tooclosefunc', tooclose_overlap), printme=printme, **kw)

   assembly = wu.rigid.RigidBodyFollowers(coords=coords, frames=frames, recenter=True, cellsize=cellsize, **kw)
   if showme: wu.showme(assembly, name='START', **kw)
   cellsize0 = cellsize
   if centerasu_at_start:
      recenter_asu_frames(assembly, symelem_siblings=None, method='to_center', axis=axes, **kw)
      if printme: ic(f'recenter {centerasu}')
   if doscale and not alongaxis:
      cellsize = slide_scale(assembly, cellsize, step=scalestep, **kw)
      if printme: ic(f'scale {cellsize}')
   for i in range(iters):
      # if i >= clashiters: kw.tooclosefunc = userfunc
      # ic(step)
      for j in range(subiters):
         # cellsize = slide_scale(assembly, cellsize, step=scalestep, **kw)
         for iax, (axis, axpos) in enumerate(axes):
            axis = wu.hnormalized(axis)
            axpos = wu.hscaled(cellsize / cellsize0, wu.hpoint(axpos))
            if towardaxis:
               # ic(axpos)
               axisperp = wu.hnormalized(wu.hprojperp(axis, assembly.asym.com() - axpos - wu.hrandvec() / 1000))
               # ic(axis, assembly.asym.com() - axpos, cellsize, axisperp)
               if centerasu and i < receniters:
                  recenter_asu_frames(assembly, symelem_siblings=symelems_siblings[iax], method=centerasu,
                                      axis=axisperp, **kw)
                  if printme: ic(f'recenter {centerasu}')
               else:
                  slide = slide_axis(axisperp, assembly, perp=True, nbrs=None, symelem_siblings=symelems_siblings[iax],
                                     step=step, **kw)
                  # if printme: ic(f'slide along {axisperp[:3]} by {slide}')
            if i < alongaxis:
               slide = slide_axis(axis, assembly, nbrs='auto', step=step, **kw)
               # printme: ic(f'slide along {axis[:3]} by {slide}')
            if doscale and alongaxis: slide = slide_axis(assembly.asym.com(), assembly, nbrs=None, step=step, **kw)
            elif doscale: cellsize = slide_cellsize(assembly, cellsize, step=scalestep, **kw)
            if showme == 'pdb': assembly.dump_pdb(f'slide_i{i}_j{j}_iax{iax}.pdb')
         for axis in along_extra_axes:
            slide = slide_axis(axis, assembly, nbrs='auto', step=step, **kw)
            # printme: ic(f'slide along {axis[:3]} by {slide}')
         if coords_to_asucen:
            cencoords = wu.sym.coords_to_asucen(sym, assembly.asym.coords)
            # ic(wu.hcom(assembly.asym.coords))
            # ic(wu.hcom(cencoords))
            assembly.set_asym_coords(cencoords)
      step *= iterstepscale
      scalestep *= iterstepscale

   if showme: wu.showme(assembly, name='FINISH', **kw)
   return assembly

def recenter_asu_frames(
   assembly,
   symelem_siblings=None,
   method=None,
   axis=None,
   showme=False,
   resetonfail=True,
   **kw,
):
   """symelem_siblings is ???
   """
   if symelem_siblings is None:
      assert method == 'to_center'
      assert axis is not None
      newcen = len(axis) * assembly.asym.com()
      for b in assembly.symbodies:
         newcen += b.com()
      newcen /= (len(assembly) + len(axis) - 1)
      assembly.asym.setcom(newcen)
      if showme: wu.showme(assembly, name='recenasuabs', **kw)
      return

   origcom = assembly.asym.com()
   partnercom = assembly.asym.com()
   othercom = assembly.asym.com()
   for p in symelem_siblings:
      partnercom += assembly.bodies[p].com()
   partnercom /= (len(symelem_siblings) + 1)
   othercom = assembly.asym.com()
   for i in range(1, len(assembly)):
      if i not in symelem_siblings:
         othercom += assembly.bodies[i].com()
   othercom /= (len(assembly) - len(symelem_siblings))
   comdir = wu.hnormalized(othercom - partnercom)
   halfdist = wu.hnorm(othercom - partnercom) / 2
   center = (partnercom + othercom) / 2
   # wu.showme(origcom, name='com')
   # wu.showme(partnercom, name='partnercom')
   # wu.showme(othercom, name='othercom')
   # wu.showme(center, name='center')

   if method == 'to_center':
      newcen = center
   else:
      if resetonfail:
         if method == 'toward_other':
            axis = wu.hnormalized(othercom - partnercom)
            dist = wu.hdot(axis, origcom - partnercom)
            dist = halfdist - dist
            newcen = origcom + axis * dist
         elif method == 'toward_partner':
            axis = wu.hnormalized(axis)
            dist = halfdist / wu.hdot(axis, comdir)
            proj = axis * dist
            newcen = partnercom + proj
            # wu.showme(axis, name='axis')
            # wu.showme(wu.hproj(axis, partnercom - othercom))
            # wu.showme(proj, name='proj')
         else:
            raise ValueError(f'bad method "{method}"')

   # wu.showme(newcen, name='newcen')
   pos = assembly.asym.setcom(newcen)
   if showme:
      wu.showme(assembly, name='recenterasu', **kw)

   # assert 0

def slide_axis(
   axis,
   assembly,
   nbrs='auto',
   tooclosefunc=None,
   perp=False,
   step=1.0,
   maxstep=100,
   showme=False,
   symelem_siblings=None,
   resetonfail=True,
   scaleslides=1.0,
   boundscheck=lambda x: True,
   nobadsteps=False,
   **kw,
):
   axis = wu.hnormalized(axis)
   origpos = assembly.asym.position
   if nbrs == 'auto': nbrs = assembly.get_neighbors_by_axismatch(axis, perp)
   elif nbrs == 'all': nbrs = None
   # ic(repr(nbrs))

   iflip, flip = 0, -1.0
   if tooclosefunc(assembly, symelem_siblings, **kw):
      iflip, flip = -1, 1.0

   total = 0.0
   delta = wu.htrans(flip * step * axis)
   success = False
   lastclose = 1.0
   for i in range(maxstep):
      assembly.asym.moveby(delta)
      if not boundscheck(assembly): break
      total += flip * step
      if showme: wu.showme(assembly, name='slideaxis%f' % axis[0], **kw)
      close = tooclosefunc(assembly, nbrs, **kw)
      if iflip and nobadsteps and close - lastclose > 0.01: break
      lastclose = close
      if iflip + bool(close):
         success = True
         break
   if not success and resetonfail:
      assembly.asym.position = origpos
      if showme: wu.showme(assembly, name='resetaxis%f' % axis[0], **kw)
      return 0
   if iflip == 0:  # back off so no clash
      total -= flip * step
      assembly.asym.moveby(wu.hinv(delta))
      if showme: wu.showme(assembly, name='backoffaxis%f' % axis[0], **kw)

   if iflip == 0 and abs(total) > 0.01 and scaleslides != 1.0:
      # if slid into contact, apply scaleslides (-1) is to undo slide
      # ic(total, total + (scaleslides - 1) * total)
      scaleslides_delta = wu.htrans((scaleslides - 1) * total * axis)
      total += (scaleslides - 1) * total

      assembly.asym.moveby(scaleslides_delta)

   return total

def slide_scale(*a, **kw):
   return slide_cellsize(*a, scalecoords=True, **kw)

def slide_cellsize(
   assembly,
   cellsize,
   tooclosefunc=None,  #tooclose_clash,
   step=1.0,
   maxstep=100,
   showme=False,
   cellscalelimit=9e9,
   resetonfail=True,
   scaleslides=1.0,
   boundscheck=lambda x: True,
   scalecoords=None,
   nobadsteps=False,
   **kw,
):
   if showme: wu.showme(assembly, name='scaleinput', **kw)
   orig_scalecoords = assembly.scale_com_with_cellsize
   if scalecoords is not None: assembly.scale_com_with_cellsize = scalecoords

   step = wu.to_xyz(step)
   # ic(cellsize, assembly.cellsize)

   assert np.allclose(cellsize, assembly.cellsize)
   orig_cellsize = assembly.cellsize
   cellsize = assembly.cellsize

   iflip, flip = 0, -1.0
   if tooclosefunc(assembly, **kw): iflip, flip = -1, 1.0

   initpos = assembly.asym.position.copy()
   success, lastclose = False, 1.0
   for i in range(maxstep):
      close = tooclosefunc(assembly, **kw)
      # ic('SLIDE CELLSIZE', bool(close), close)
      if iflip + bool(close):
         success = True
         break
      if iflip and nobadsteps and close - lastclose > 0.01: break
      lastclose = close
      # ic(cellsize, flip, step)
      delta = (cellsize + flip * step) / cellsize
      # ic(cellsize, flip * step)
      assert np.min(np.abs(delta)) > 0.0001
      cellsize *= delta
      assembly.scale_frames(delta, safe=False)
      # changed = assembly.scale_frames(delta, safe=False)
      # if not changed:
      # assert i == 0
      # return cellsize
      if not boundscheck(assembly): break
      if showme: wu.showme(assembly, name='scale', **kw)
      if np.all(cellsize / orig_cellsize > cellscalelimit):
         success = True
         break

   if not success:
      if resetonfail:
         assembly.scale_frames(orig_cellsize / cellsize)
      if showme: wu.showme(assembly, name='resetscale', **kw)
      return orig_cellsize

   if iflip == 0:  # back off
      delta = 1.0 / delta
      assembly.scale_frames(delta, safe=False)
      cellsize *= delta

      if showme: wu.showme(assembly, name='backoffscale', **kw)

   if iflip == 0 and scaleslides != 1.0 and np.sum((cellsize - orig_cellsize)**2) > 0.001:
      # if slid into contact, apply scaleslides (-1) is to undo slide

      newcellsize = (cellsize - orig_cellsize) * (scaleslides) + orig_cellsize
      newdelta = newcellsize / cellsize
      # ic(cellsize, newcellsize)
      cellsize = newcellsize

      assembly.scale_frames(newdelta, safe=False)

   assembly.scale_com_with_cellsize = orig_scalecoords
   return cellsize