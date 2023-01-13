import functools as ft
import numpy as np
import willutil as wu

def npscorefunc(xtal, scom, state):
   dis2 = 0
   for i, (s, com) in enumerate(zip(xtal.symelems, scom)):
      newcom = com + state.cartshift
      newcen = s.cen * state.cellsize
      dis2 = dis2 + wu.hpointlinedis(newcom, newcen, s.axis)**2
   err = np.sqrt(np.sum(dis2))
   return err

def torchscorefunc(xtal, scom, cellsize, cartshift, grad=True):
   import torch
   dis2 = torch.tensor([0.], requires_grad=grad)
   for i, (s, com) in enumerate(zip(xtal.symelems, scom)):
      com = torch.tensor(com[:3], requires_grad=grad)
      cen = torch.tensor(s.cen[:3], requires_grad=grad)
      axis = torch.tensor(s.axis[:3], requires_grad=grad)
      newcom = com + cartshift[:3]
      newcen = cen * cellsize
      dis2 = dis2 + wu.homog.thgeom.th_point_line_dist2(newcom, newcen, axis)
   err = torch.sqrt(torch.sum(dis2))
   return err

def fit_coords_to_xtal(xtal, coords, cellsize=None, domc=True, domin=False, noshift=False, **kw):
   'OK... this is a pretty inefficient way...'
   coms = wu.hcom(coords)
   if isinstance(cellsize, np.ndarray):
      cellsize = cellsize[0]
   cellsize = cellsize if cellsize is not None else 100.0
   # ic(coms)

   scom = list()
   n = 0
   for s in xtal.symelems:
      scom.append(coms[0].copy())
      nops = len(s.operators)
      for i in range(nops - 1):
         n += 1
         # ic(len(scom), n)
         scom[-1] += coms[n]
      scom[-1] /= nops
   # ic(scom)
   elem0 = xtal.symelems[0]
   if noshift:
      cartshift = wu.hvec([0, 0, 0])
   else:
      cartshift = -wu.hvec(wu.hpointlineclose(scom[0], elem0.cen, elem0.axis))

   assert not domin
   assert domc

   if domc:
      state = wu.Bunch(
         cellsize=cellsize,
         # cartshift=np.array([0., 0, 0, 0]),
         cartshift=cartshift,
      )
      step = 5
      mc = wu.MonteCarlo(ft.partial(npscorefunc, xtal, scom), temperature=0.3)
      for i in range(1000):
         # if i % 100 == 199:
         # state = mc.beststate
         prev = state.copy()
         state.cellsize += step * np.random.randn()
         if not noshift:
            state.cartshift += 0.02 * step * wu.hrandvec()

         acccepted = mc.try_this(state)
         if not acccepted:
            state = prev
         else:
            # print(state.cellsize, wu.hnorm(state.cartshift), mc.best)
            pass
            # ic(mc.acceptfrac, step)
            # if mc.acceptfrac > 0.25: step *= 1.01
            # else: step *= 0.99

      # print(mc.best)
      # print(mc.beststate)
      # print(mc.acceptfrac)

      return mc.beststate.cellsize, mc.beststate.cartshift
      assert 0

      cellsize, cartshift = mc.beststate

      return cellsize, cartshift.astype(coords.dtype)

   if domin:

      import torch
      # torch.autograd.set_detect_anomaly(True)

      # check
      v1 = npscorefunc(xtal, scom, wu.Bunch(cellsize=cellsize, cartshift=cartshift))
      v2 = torchscorefunc(xtal, scom, cellsize, cartshift, grad=False)
      assert np.allclose(v1, v2)

      cellsize = torch.tensor(cellsize, requires_grad=True)
      cartshift = torch.tensor(cartshift[:3], requires_grad=True)
      for i in range(10):
         err = torchscorefunc(xtal, scom, cellsize, cartshift)
         err.backward()
         cellgrad = cellsize.grad
         cartshiftgrad = cartshift.grad
         mul = 1
         cellsize = (cellsize - mul * cellgrad).clone().detach().requires_grad_(True)
         cartshift = (cartshift - mul * cartshiftgrad).clone().detach().requires_grad_(True)

         ic(err)  #, cellsize, cartshift, cartshiftgrad)

      assert 0

      cellsize = 100
      cartshift = np.array([0., 0, 0, 0])
      besterr, beststate = 9e9, None
      step = 10.0
      lasterr, laststate = 9e9, None
      for i in range(1000):
         offaxis = list()
         for s, com in zip(xtal.symelems, scom):
            offaxis.append(wu.hpointlinedis(com, cellsize * s.cen, s.axis)**2)
         err = np.sqrt(np.sum(offaxis))
         if err < besterr:
            besterr = err
            beststate = cellsize, cartshift
         if err - lasterr < 0:
            lasterr = err
            laststate = cellsize, cartshift
         else:
            cellsize, cartshift = laststate

         ic(cellsize, err, step)
         step *= 0.99
      ic(besterr, beststate)

   assert 0
