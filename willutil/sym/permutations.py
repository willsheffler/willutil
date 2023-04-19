import numpy as np
import willutil as wu

def permutations(sym, **kw):
   frames = wu.sym.frames(sym, **kw)
   perm = list()
   for i, f in enumerate(frames):
      local_frames = wu.hxform(wu.hinv(frames[i]), frames)
      dist = wu.hdiff(frames, local_frames, lever=3)
      idx = np.argmin(dist, axis=1)
      mindist = dist[np.arange(len(idx)), idx]
      missing = mindist > 1e-3
      idx[missing] = -1
      assert len(set(idx)) == len(idx)
      perm.append(idx)
   perm = np.stack(perm)
   if wu.sym.is_closed(sym):
      assert np.all(perm >= 0)
   return perm
