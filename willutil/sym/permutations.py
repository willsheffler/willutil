import functools, opt_einsum
import concurrent.futures as cf
import numpy as np
import willutil as wu
import torch

def symframe_permutations(frames, **kw):

   func = functools.partial(symperm1, frames=frames)
   with cf.ThreadPoolExecutor(max_workers=8) as exe:
      perm = exe.map(func, range(len(frames)))
   perm = np.stack(perm)
   return perm

def symframe_permutations_torch(frames):
   frames = torch.tensor(frames, device='cuda').to(torch.int32)
   perm = list()
   for i, frame in enumerate(frames):
      if i % 100 == 0:
         ic(i, len(frames))
      local_frames = opt_einsum.contract('ij,fjk->fik', torch.linalg.inv(frame), frames)
      dist2 = torch.sum((local_frames[None] - frames[:, None])**2, axis=(2, 3))
      idx = torch.argmin(dist2, axis=1)
      mindist = dist2[torch.arange(len(idx)), idx]
      missing = mindist > 1e-5
      idx[missing] = -1
      perm.append(idx)
   perm = torch.stack(perm).to(torch.int32)
   return perm.to('cpu').numpy()

def symperm1(i, frames):
   if i % 100 == 0:
      ic(i, len(frames))
   local_frames = wu.hxform(wu.hinv(frames[i]), frames)
   dist = wu.hdiff(frames, local_frames, lever=3)
   idx = np.argmin(dist, axis=1)
   mindist = dist[np.arange(len(idx)), idx]
   missing = mindist > 1e-3
   idx[missing] = -1
   return idx

def permutations(sym, **kw):
   frames = wu.sym.frames(sym, **kw)
   perm = symframe_permutations(frames, **kw)
   if wu.sym.is_closed(sym):
      for idx in perm:
         assert len(set(idx)) == len(idx)
      assert np.all(perm >= 0)
   return perm
