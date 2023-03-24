import numpy as np
import willutil as wu

def scaleunit(scale, val):
   if not isinstance(val, (int, float)):
      return np.array([scaleunit(scale, v) for v in val])
   if -1 <= val <= 1:
      return val * scale
   else:
      return val

def place_asu_grid(
      pos,
      cellsize,
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
      **kw,
):
   t = wu.Timer()
   nsampcell = nsampcell or nsamp
   if isinstance(lbub, (int, float)): lbub = (-lbub, lbub)
   if isinstance(lbubcell, (int, float)): lbubcell = (-lbubcell, lbubcell)
   pos0, cellsize0, frames0, framesavoid0 = pos, cellsize, frames, framesavoid
   lbub = scaleunit(cellsize0, lbub)
   ic(lbub)
   lbubcell = scaleunit(cellsize0, lbubcell)
   distcontact = scaleunit(cellsize0, distcontact)
   distavoid, distspread, clusterdist = scaleunit(cellsize0, [distavoid, distspread, clusterdist])

   samp = np.linspace(*lbub, nsamp)
   xyz = np.meshgrid(samp, samp, samp)
   delta = np.stack(xyz, axis=3).reshape(-1, 3)
   delta = wu.hvec(delta)
   pos = pos + delta
   # wu.showme(pos)
   cellsizes = cellsize + np.linspace(*lbubcell, nsampcell)
   if nsampcell < 2: cellsizes = np.array([cellsize])
   allframes = np.concatenate([frames0, framesavoid0])
   frames = np.stack([wu.hscaled(s, frames[1:]) for s in cellsizes])
   framesavoid = np.stack([wu.hscaled(s, framesavoid) for s in cellsizes])

   contact = wu.hxform(frames, pos)
   t.checkpoint('contact')
   avoid = wu.hxform(framesavoid, pos)
   t.checkpoint('avoid')
   dcontact = wu.hnorm(pos - contact)
   davoid = wu.hnorm(pos - avoid)
   dcontactmin = np.min(dcontact, axis=1)
   dcontactmax = np.max(dcontact, axis=1)
   davoidmin = np.min(davoid, axis=1)
   t.checkpoint('dists')

   okavoid = davoidmin > distavoid
   okccontactmin = dcontactmin > distcontact[0]
   okccontactmax = dcontactmax < distcontact[1]
   okspread = dcontactmax - dcontactmin < distspread
   ok = okavoid * okccontactmin * okccontactmax * okspread
   w = np.where(ok)
   goodcell = cellsizes[w[:][0]]
   goodpos = pos[w[:][1]]
   cellpos = goodpos / goodcell[:, None]
   cellpos0 = pos0 / cellsize0
   origdist = wu.hnorm(cellpos - cellpos0)
   order = np.argsort(origdist)
   goodcell, goodpos = goodcell[order], goodpos[order]
   ic(goodcell.shape)

   if clusterdist > 0 and len(goodpos) > 1:
      coords = wu.hxform(frames0, goodpos)
      coords = coords.swapaxes(0, 1).reshape(len(goodpos), -1)
      keep, clustid = wu.cpp.cluster.cookie_cutter(coords, float(clusterdist))
      goodpos = goodpos[keep]
      goodcell = goodcell[keep]
      t.checkpoint('clust')

   t.report()

   return goodpos, goodcell