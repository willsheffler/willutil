import numpy as np
import willutil as wu

def vispoints(pos, cell, frames, allframes):
   for i in range(len(pos)):
      # result0 = wu.hxform(wu.hscaled(newcell[i], frames), pos[i], is_points=True)
      # wu.showme(result0, sphere=25 / 2, kind='point'
      # result1 = wu.hxform(wu.hscaled(newcell[i], framesavoid), pos[i])
      # wu.showme(result1, sphere=3)
      f = np.concatenate([frames, allframes])
      colors = [(0, 1, 1)] + [(1, 0, 0)] * (len(frames) - 1) + [(1, 1, 1)] * (len(f))
      result = wu.hxform(wu.hscaled(cell[i], f), pos[i])
      wu.showme(result, sphere=9, col=colors)

def scaleunit(scale, val):
   if not isinstance(val, (int, float, np.ndarray)):
      return np.array([scaleunit(scale, v) for v in val])
   if np.all((-1 <= val) * (val <= 1)):
      return val * scale
   else:
      return val

def place_asu_grid_multiscale(
   pos,
   cellsize,
   *a,
   minpos=1,
   **kw,
):
   # print('place_asu_grid_multiscale', repr(pos), cellsize, flush=True)
   kw = wu.Bunch(kw)

   assert kw.lbub < 1
   assert kw.lbubcell < 1

   newpos, newcell = place_asu_grid(pos, cellsize, *a, **kw)
   if len(newpos) >= minpos: return newpos, newcell
   kw.distcontact = np.array(kw.distcontact)
   for i in range(5, 0, -1):
      if not 'refpos' in kw: kw.refpos = pos.copy()
      if not 'refcell' in kw: kw.refcell = cellsize
      # ic(i, repr(pos), cellsize)
      print('place_asu_grid_multiscale', i, flush=True)
      newpos, newcell = place_asu_grid(
         pos,
         cellsize,
         *a,
         **kw.sub(
            nsampcell=kw.nsampcell + (i - 1),
            lbub=kw.lbub + (i - 1) * 0.03,
            lbubcell=kw.lbubcell + (i - 1) * 0.01,
            dnistcontact=(kw.distcontact[0], kw.distcontact[1] + (i - 1)),
            distavoid=kw.distavoid - (i - 1),
            distspread=kw.distspread + (i - 1),
         ),
      )
      pos, cellsize = newpos[0], newcell[0]
      # ic(kw.refpos)
      # ic(newpos[1] - kw.refpos)
      # ic(newpos[:5])
      # ic(wu.hnorm(newpos - kw.refpos)[:5])

      # vispoints(newpos[:1], newcell[:1], kw.frames, kw.framesavoid)

   return newpos, newcell

def place_asu_grid(
      pos,
      cellsize,
      *,
      frames,
      framesavoid,
      lbub=(-10, 10),
      lbubcell=(-20, 20),
      nsamp=1,
      nsampcell=None,
      distcontact=(10, 15),
      distavoid=20,
      distspread=9e9,
      clusterdist=3,
      refpos=None,
      refcell=None,
      printme=False,
      **kw,
):
   if printme:
      print('   # yapf: disable')
      print('   kw =', repr(kw))
      print(f'''   wu.sym.place_asu_grid(
      pos={wu.misc.arraystr(pos)},
      cellsize={repr(cellsize)},
      frames={wu.misc.arraystr(frames)},
      framesavoid={wu.misc.arraystr(framesavoid)},
      lbub={repr(lbub)},
      lbubcell={repr(lbubcell)},
      nsamp={repr(nsamp)},
      nsampcell={repr(nsampcell)},
      distcontact={repr(distcontact)},
      distavoid={repr(distavoid)},
      distspread={repr(distspread)},
      clusterdist={repr(clusterdist)},
      refpos={repr(refpos)},
      refcell={repr(refcell)},
   )''')
      print('   # yapf: enable', flush=True)

   assert isinstance(cellsize, (int, float))
   nsampcell = nsampcell or nsamp
   if isinstance(lbub, (int, float)): lbub = (-lbub, lbub)
   if isinstance(lbubcell, (int, float)): lbubcell = (-lbubcell, lbubcell)
   cellsize0, frames0, framesavoid0 = cellsize, frames, framesavoid
   pos0 = scaleunit(cellsize0, pos)
   pos = scaleunit(cellsize0, pos)
   refpos = pos if refpos is None else refpos
   refpos = scaleunit(cellsize0, refpos)
   refcell = cellsize if refcell is None else refcell
   pos[3], pos0[3], refpos[3] = 1, 1, 1
   lbub = scaleunit(cellsize0, lbub)
   lbubcell = scaleunit(cellsize0, lbubcell)
   distcontact = scaleunit(cellsize0, distcontact)
   distavoid, distspread, clusterdist = scaleunit(cellsize0, [distavoid, distspread, clusterdist])

   samp = np.linspace(*lbub, nsamp)
   xyz = np.meshgrid(samp, samp, samp)
   delta = np.stack(xyz, axis=3).reshape(-1, 3)
   delta = wu.hvec(delta)
   posgrid = pos + delta
   posgrid = posgrid[np.all(posgrid > 0, axis=1)]
   # wu.showme(posgrid)
   cellsizes = cellsize + np.linspace(*lbubcell, nsampcell)
   if nsampcell < 2: cellsizes = np.array([cellsize])
   # ic(frames0.shape, framesavoid0.shape)
   allframes = np.concatenate([frames0, framesavoid0])
   frames = np.stack([wu.hscaled(s, frames[1:]) for s in cellsizes])
   framesavoid = np.stack([wu.hscaled(s, framesavoid) for s in cellsizes])

   contact = wu.hxformpts(frames, posgrid, outerprod=True)
   avoid = wu.hxformpts(framesavoid, posgrid, outerprod=True)
   dcontact = wu.hnorm(posgrid - contact)
   davoid = wu.hnorm(posgrid - avoid)
   dcontactmin = np.min(dcontact, axis=1)
   dcontactmax = np.max(dcontact, axis=1)
   davoidmin = np.min(davoid, axis=1)

   okavoid = davoidmin > distavoid
   okccontactmin = dcontactmin > distcontact[0]
   okccontactmax = dcontactmax < distcontact[1]
   okspread = dcontactmax - dcontactmin < distspread
   ic(np.sum(okavoid), np.sum(okccontactmin), np.sum(okccontactmax), np.sum(okspread))
   ok = okavoid * okccontactmin * okccontactmax * okspread
   w = np.where(ok)
   goodcell = cellsizes[w[:][0]]
   goodpos = posgrid[w[:][1]]
   # cellpos = goodpos / goodcell[:, None]
   # cellpos0 = pos0 / cellsize0
   origdist = wu.hnorm2(goodpos - refpos) + ((goodcell - refcell) * 0.6)**2
   order = np.argsort(origdist)
   goodcell, goodpos = goodcell[order], goodpos[order]

   if clusterdist > 0 and len(goodpos) > 1:
      coords = wu.hxformpts(frames0, goodpos, outerprod=True)
      coords = coords.swapaxes(0, 1).reshape(len(goodpos), -1)
      keep, clustid = wu.cpp.cluster.cookie_cutter(coords, float(clusterdist))
      goodpos = goodpos[keep]
      goodcell = goodcell[keep]

   # f = wu.hscaled(goodcell[0], frames0)
   # p = wu.hxformpts(f, goodpos[0])
   # ic(wu.hnorm(p[0] - p[1]), wu.hnorm(p[0] - p[-1]))

   # if len(goodpos):
   # ic(refpos)
   # ic(goodpos[:5])
   # ic(goodpos[1] - refpos)
   # ic(wu.hnorm(goodpos - refpos)[:5])

   return goodpos, goodcell

def place_asu_sample_dof(
   sym,
   coords,
   cellsize,
   axis,
   contactdist,
   cartnsamp,
   angnsamp,
   cartrange,
   angrange,
   cellrange,
   cellnsamp,
):
   axis = wu.hnormalized(axis)
   angsamp = wu.hrot(axis, np.linspace(-angrange, angrange, angnsamp))
   cartsamp = wu.htrans(axis * np.linspace(-cartrange, cartrange, cartnsamp))
   cellsamp = np.linspace(-cellrange, cellrange, cellnsamp)
   cframes = np.stack([wu.sym.frames(sym, cellsize=c) for c in cellsamp])
   ic(cframes.shape)