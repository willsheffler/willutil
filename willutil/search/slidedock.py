import willutil as wu
import numpy as np

def slide_dock_oligomer(sym, psym, nsym, xyzorig, step=1.0, clash_radius=2.4, contact_dis=8, startaxis=None):

   nrot = int(360 // int(psym[1]) // step)
   step = np.radians(step)

   if hasattr(xyzorig, 'cpu'):
      xyzorig = xyzorig.cpu().numpy()
      startaxis = startaxis.cpu().numpy()
   if startaxis is None:
      assert len(xyzorig) == int(psym[1])
      startaxis = wu.hnormalized(xyzorig.mean(axis=(0, 1, 2)))
   else:
      # ic(startaxis)
      assert xyzorig.ndim == 3
      # wu.dumppdb('start_xyzorig.pdb', xyzorig)
      pframes = wu.hrot(startaxis, [0, 120, 240])
      # ic(pframes.shape)
      xyzorig = wu.hxform(pframes, xyzorig)
      # ic(xyzorig.shape)
      # wu.dumppdb('start_cx.pdb', xyzorig)
      # assert 0

   # np.save('/tmp/startaxis.npy', startaxis)
   # np.save('/tmp/xyzorig.npy', xyzorig)
   # assert 0

   paxis = wu.sym.axes(sym=sym, nfold=psym)
   naxis = wu.sym.axes(sym=sym, nfold=nsym)

   ic(startaxis, paxis, naxis)
   initalign = wu.halign(startaxis, paxis)
   tonbr = wu.hrot(naxis, nfold=int(nsym[1]))
   paxis2 = wu.hxform(tonbr, paxis)
   # ic(tonbr)
   xyz = wu.hxform(initalign, xyzorig)
   axisang = wu.hangle(paxis, naxis)
   slidedirn = wu.hnormalized(wu.hxform(tonbr, paxis) - paxis)
   bvh = wu.cpp.bvh.BVH(xyz[:, :, :].reshape(-1, 3))

   best = 0, None, None
   for i in range(nrot):
      pos1 = wu.hrot(paxis, i * step)
      pos2 = tonbr @ pos1
      delta = wu.cpp.bvh.bvh_slide(bvh, bvh, pos1, pos2, clash_radius, slidedirn[:3])
      delta = delta / np.sin(axisang)
      # ic(delta)
      # assert 0
      # t.checkpoint('slide')
      pos1 = wu.htrans(-paxis * delta / 2, doto=pos1)
      pos2 = wu.htrans(-paxis2 * delta / 2, doto=pos2)
      score = wu.cpp.bvh.bvh_count_pairs_vec(bvh, bvh, pos1, pos2, contact_dis)
      if score > best[0]:
         best = score, pos1, pos2
         # ic(i, best[0])
         # wu.showme(np.concatenate([wu.hxform(pos1, xyz), wu.hxform(pos2, xyz)]), name=f'dock{i}')
         # wu.dumppdb(f'slidedock{i:04}.pdb', np.concatenate([wu.hxform(pos1, xyz), wu.hxform(pos2, xyz)]))
      # assert 0
   ic(best[0])
   score, pos1, pos2 = best
   # wu.showme(np.concatenate([wu.hxform(pos1, xyz), wu.hxform(pos2, xyz)]), name=f'dock{i}')
   newxyz = wu.hxform(pos1, xyz[0])

   # wu.dumppdb(f'slidedock_best.pdb', newxyz)

   # find closest to orig com
   com = wu.hcom(xyzorig[0, :, 1])
   symcom = wu.hxform(wu.sym.frames(sym), wu.hcom(newxyz[:, 1]))
   f = wu.sym.frames(sym)[np.argmin(wu.hnorm(com - symcom))]
   newxyz = wu.hxform(f, newxyz)

   # wu.dumppdb(f'slidedock_done.pdb', newxyz)

   return newxyz
