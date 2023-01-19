import functools
import numpy as np
import willutil as wu
from willutil.rigid.objective import tooclose_clash, tooclose_overlap

def helix_slide(
   helix,
   coords,
   cellsize,
   coils=2,
   contactfrac=0,
   step=0.03,
   maxstep=10,
   iters=4,
   breathe=[2.5, 1, 0.5, 0],
   showme=False,
   closest=9,
   scalefirst=False,
   **kw,
):
   assert np.allclose(cellsize[0], cellsize[1])
   cellsize = cellsize.copy()
   coords = coords.astype(np.float64)

   hframes = helix.frames(maxdist=9e9, radius=cellsize[0], spacing=cellsize[2], coils=coils, closest=closest)
   rb = wu.RigidBodyFollowers(coords=coords, frames=hframes, symtype='H', cellsize=cellsize, clashdis=8, contactdis=16)
   if showme: rb.dump_pdb(f'helix_slide____.pdb')
   hstep = np.array([0.00, 0.00, step])
   rstep = np.array([step, step, 0.00])
   sstep = np.array([step, step, step])
   tooclose = functools.partial(wu.rigid.tooclose_overlap, contactfrac=contactfrac)
   # steps = [hstep, rstep]
   steps = [rstep, hstep]
   if scalefirst:
      steps = [sstep] + steps
   for i, expand in enumerate(breathe):
      for j in range(iters):

         for step in steps:
            scale = (1 + step * expand)
            wu.sym.slide_scale(
               rb,
               cellsize=1,
               step=step,
               tooclosefunc=tooclose,
               showme=showme,
               maxstep=maxstep,
               moveasymunit=False,
               **kw,
            )
            if expand > 0:
               rb.scale_frames(scale)
            if showme:
               wu.showme(rb, **kw)
               rb.dump_pdb(f'helix_slide_{i}_{j}.pdb')
   return rb

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
   subiters=1,
   clashiters=5,
   receniters=2,
   step=10,
   closestfirst=True,
   centerasu='toward_partner',
   centerasu_at_start=False,
   showme=False,
   **kw,
):
   kw = wu.Bunch(kw)
   kw.showme = showme
   coords = wu.hpoint(coords).copy()
   coords = coords.reshape(-1, 4)
   axassoc = []
   if axes is None:
      axes0 = wu.sym.axes(sym, cellsize=cellsize)
      # ic(axes)
      if isinstance(axes0, dict):
         axes = [(ax, wu.hpoint([0, 0, 0])) for ax in axes0.values()]
      else:
         isxtal = True
         axes = [(elem.axis, elem.cen) for elem in axes0]
         doscale = True if doscale is None else doscale
         axassoc = wu.sym.symelem_associations(axes0)
      com = wu.hcom(coords)
      faxdist = [wu.hpointlinedis(com, ac[1], wu.hscaled(cellsize, ac[0])) for ac in axes]
      faxorder = np.argsort(faxdist)
      # ic([faxdist(ac) for ac in axes])
      axes = [axes[i] for i in faxorder]
      if axassoc: axassoc = [axassoc[i] for i in faxorder]
      if not closestfirst:
         axes = list(reversed(axes))
         axassoc = list(reversed(axassoc))
   axassoc = axassoc or ['auto'] * len(axes)
   # ic(axassoc)

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
   userfunc = functools.partial(kw.get('tooclosefunc', tooclose_overlap), printme=printme, **kw)

   bodies = wu.rigid.RigidBodyFollowers(coords=coords, frames=frames, recenter=True, cellsize=cellsize, **kw)
   cellsize0 = cellsize
   if centerasu_at_start:
      recenter_asu_frames(bodies, partners=None, method='to_center', axis=axes, **kw)
      if printme: ic(f'recenter {centerasu}')
   if doscale and not alongaxis:
      cellsize = slide_scale(bodies, cellsize, step=step, **kw)
      if printme: ic(f'scale {cellsize}')
   for i in range(iters):
      if i >= clashiters: kw.tooclosefunc = userfunc
      for j in range(subiters):
         for iax, (axis, axpos) in enumerate(axes):
            axis = wu.hnormalized(axis)
            axpos = wu.hscaled(cellsize / cellsize0, wu.hpoint(axpos))
            if towardaxis:
               # ic(axpos)
               partners = axassoc[iax]
               axisperp = wu.hnormalized(wu.hprojperp(axis, bodies.asym.com() - axpos))  # points away from axis
               if centerasu and i < receniters:
                  recenter_asu_frames(bodies, partners=partners, method=centerasu, axis=axisperp, **kw)
                  if printme: ic(f'recenter {centerasu}')
               else:
                  slide = slide_axis(axisperp, bodies, perp=True, nbrs=None, partners=partners, step=step, **kw)
                  if printme: ic(f'slide along {axisperp[:3]} by {slide}')
            if alongaxis:
               slide = slide_axis(axis, bodies, nbrs='auto', step=step, **kw)
               printme: ic(f'slide along {axis[:3]} by {slide}')
            if doscale and alongaxis: slide = slide_axis(bodies.asym.com(), bodies, nbrs=None, step=step, **kw)
            elif doscale: cellsize = slide_scale(bodies, cellsize, step=step, **kw)
            if showme == 'pdb': bodies.dump_pdb(f'slide_i{i}_j{j}_iax{iax}.pdb')
         step *= 0.75

   return bodies

def recenter_asu_frames(
   bodies,
   partners=None,
   method=None,
   axis=None,
   showme=False,
   **kw,
):

   if partners is None:
      assert method == 'to_center'
      assert axis is not None
      newcen = len(axis) * bodies.asym.com()
      for b in bodies.symbodies:
         newcen += b.com()
      newcen /= (len(bodies) + len(axis) - 1)
      bodies.asym.setcom(newcen)
      if showme: wu.showme(bodies, name='recenasuabs', **kw)
      return

   com = bodies.asym.com()
   partnercom = bodies.asym.com()
   othercom = bodies.asym.com()
   for p in partners:
      partnercom += bodies.bodies[p].com()
   partnercom /= (len(partners) + 1)
   othercom = bodies.asym.com()
   for i in range(1, len(bodies)):
      if i not in partners:
         othercom += bodies.bodies[i].com()
   othercom /= (len(bodies) - len(partners))
   comdir = wu.hnormalized(othercom - partnercom)
   halfdist = wu.hnorm(othercom - partnercom) / 2
   center = (partnercom + othercom) / 2
   # wu.showme(com, name='com')
   # wu.showme(partnercom, name='partnercom')
   # wu.showme(othercom, name='othercom')
   # wu.showme(center, name='center')

   if method == 'to_center':
      newcen = center
   else:
      if method == 'toward_other':
         axis = wu.hnormalized(othercom - partnercom)
         dist = wu.hdot(axis, com - partnercom)
         dist = halfdist - dist
         newcen = com + axis * dist
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
   pos = bodies.asym.setcom(newcen)
   if showme:
      wu.showme(bodies, name='recenterasu', **kw)

   # assert 0

def slide_axis(
   axis,
   bodies,
   nbrs='auto',
   tooclosefunc=tooclose_clash,
   perp=False,
   step=1.0,
   maxstep=100,
   showme=False,
   partners=None,
   **kw,
):
   axis = wu.hnormalized(axis)
   origpos = bodies.asym.position
   if nbrs == 'auto': nbrs = bodies.get_neighbors_by_axismatch(axis, perp)
   elif nbrs == 'all': nbrs = None

   iflip, flip = 0, -1.0
   if tooclosefunc(bodies, partners):
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
      close = tooclosefunc(bodies, nbrs)
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
   tooclosefunc=tooclose_clash,
   step=1.0,
   maxstep=100,
   showme=False,
   cellscalelimit=9e9,
   moveasymunit=True,
   **kw,
):
   if showme: wu.showme(bodies, name='scaleinput', **kw)

   cellsize = wu.to_xyz(cellsize)
   step = wu.to_xyz(step)
   cellsize0 = cellsize

   iflip, flip = 0, -1.0
   if tooclosefunc(bodies):
      iflip, flip = -1, 1.0
      # 'SCALE rev'
   initpos, initcell = bodies.asym.position.copy(), cellsize
   for i in range(maxstep):
      close = tooclosefunc(bodies)
      if iflip + close: break

      # ic(cellsize, flip, step)
      delta = (cellsize + flip * step) / cellsize
      assert np.min(np.abs(delta)) > 0.0001
      cellsize *= delta

      changed = bodies.scale_frames(delta, safe=False)
      if not changed:
         assert i == 0
         return cellsize

      newpos = bodies.asym.position
      newpos[:3, 3] *= delta
      if moveasymunit: bodies.asym.position = newpos
      if showme: wu.showme(bodies, name='scale', **kw)
      if np.all(cellsize / cellsize0 > cellscalelimit):
         # assert 0
         break

   else:
      if moveasymunit: bodies.asym.position = initpos
      bodies.scale_frames(initcell / cellsize)
      if showme: wu.showme(bodies, name='resetscale', **kw)
      return initcell

   if iflip == 0:  # back off
      delta = 1.0 / delta
      bodies.scale_frames(delta, safe=False)
      newpos = bodies.asym.position
      newpos[:3, 3] *= delta
      if moveasymunit: bodies.asym.position = newpos
      cellsize *= delta

      if showme: wu.showme(bodies, 'backoffscale', **kw)

   return cellsize