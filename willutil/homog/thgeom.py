import itertools as it, functools as ft
import deferred_import

np = deferred_import.deferred_import('numpy')
torch = deferred_import.deferred_import('torch')
import willutil as wu
from willutil.homog.hgeom import _hxform_impl
from willutil.homog.hgeom import rand_xform_small

def th_mean_along(vecs, along=None):
   vecs = th_vec(vecs)
   assert vecs.ndim == 2
   if not along:
      along = vecs[0]
   along = th_vec(along)
   sign = torch.sign(th_dot(along, vecs))
   flipped = (vecs.T * sign).T
   tot = torch.sum(flipped, axis=0)
   return th_normalized(tot)

def th_com_flat(points, closeto=None, closefrac=0.5):
   if closeto != None:
      dist = th_norm(points - closeto)
      close = torch.argsort(dist)[:closefrac * len(dist)]
      points = points[close]
   return torch.mean(points, axis=-2)

def th_com(points, **kw):
   points = th_point(points)
   oshape = points.shape
   points = points.reshape(-1, oshape[-2], 4)
   com = th_com_flat(points)
   com = com.reshape(*oshape[:-2], 4)
   return com

def th_rog_flat(points):
   com = th_com_flat(points).reshape(-1, 1, 4)
   delta = torch.linalg.norm(points - com, dim=2)
   rg = torch.sqrt(torch.mean(delta**2, dim=1))
   return rg

def th_rog(points, aboutaxis=None):
   points = th_point(points)
   oshape = points.shape
   points = points.reshape(-1, *oshape[-2:])
   if aboutaxis != None:
      aboutaxis = th_vec(aboutaxis)
      points = th_projperp(aboutaxis, points)
   rog = th_rog_flat(points)
   rog = rog.reshape(oshape[:-2])
   return rog

def th_proj(u, v):
   u = th_vec(u)
   v = th_point(v)
   return th_dot(u, v)[..., None] / th_norm2(u)[..., None] * u

def th_projperp(u, v):
   u = th_vec(u)
   v = th_point(v)
   return v - th_proj(u, v)

def th_axis_angle_cen(xforms, ident_match_tol=1e-8):
   # ic(xforms.dtype)
   origshape = xforms.shape[:-2]
   xforms = xforms.reshape(-1, 4, 4)
   axis, angle = th_axis_angle(xforms)
   not_ident = torch.abs(angle) > ident_match_tol
   cen = torch.tile(
      torch.tensor([0, 0, 0, 1]),
      angle.shape,
   ).reshape(*angle.shape, 4)

   assert torch.all(not_ident)
   xforms1 = xforms[not_ident]
   axis1 = axis[not_ident]
   #  sketchy magic points...
   p1, p2 = axis_ang_cen_magic_points_torch()
   p1 = p1.to(xforms.dtype)
   p2 = p2.to(xforms.dtype)
   tparallel = th_dot(axis, xforms[..., :, 3])[..., None] * axis
   q1 = xforms @ p1 - tparallel
   q2 = xforms @ p2 - tparallel
   n1 = th_normalized(q1 - p1).reshape(-1, 4)
   n2 = th_normalized(q2 - p2).reshape(-1, 4)
   c1 = (p1 + q1) / 2.0
   c2 = (p2 + q2) / 2.0

   isect, norm, status = th_intersect_planes(c1, n1, c2, n2)
   cen1 = isect[..., :]
   if len(cen) == len(cen1):
      cen = cen1
   else:
      cen = torch.where(not_ident, cen1, cen)

   axis = axis.reshape(*origshape, 4)
   angle = angle.reshape(origshape)
   cen = cen.reshape(*origshape, 4)
   return axis, angle, cen

def th_rot(axis, angle, center=None, hel=None, squeeze=True):
   if center is None: center = torch.tensor([0, 0, 0, 1], dtype=torch.float)
   angle = torch.as_tensor(angle)
   axis = th_vec(axis)
   center = th_point(center)
   if hel is None: hel = torch.tensor([0], dtype=torch.float)
   if axis.ndim == 1: axis = axis[None, ]
   if angle.ndim == 0: angle = angle[None, ]
   if center.ndim == 1: center = center[None, ]
   if hel.ndim == 0: hel = hel[None, ]
   rot = t_rot(axis, angle, shape=(4, 4), squeeze=False)
   shape = angle.shape

   # assert 0
   x, y, z = center[..., 0], center[..., 1], center[..., 2]
   center = torch.stack([
      x - rot[..., 0, 0] * x - rot[..., 0, 1] * y - rot[..., 0, 2] * z,
      y - rot[..., 1, 0] * x - rot[..., 1, 1] * y - rot[..., 1, 2] * z,
      z - rot[..., 2, 0] * x - rot[..., 2, 1] * y - rot[..., 2, 2] * z,
      torch.ones(*shape),
   ], axis=-1)
   shift = axis * hel[..., None]
   center = center + shift
   r = torch.cat([rot[..., :3], center[..., None, ]], axis=-1)
   if r.shape == (1, 4, 4): r = r.reshape(4, 4)
   return r

def th_rand_point(*a, **kw):
   return rand_point(*a, **kw)

def th_rand_vec(*a, **kw):
   return torch.from_numpy(rand_vec(*a, **kw))

def th_rand_xform_small(*a, **kw):
   return torch.from_numpy(rand_xform_small(*a, **kw))

def th_rand_xform(*a, **kw):
   return torch.from_numpy(rand_xform(*a, **kw))

def th_rand_quat(*a, **kw):
   return torch.from_numpy(rand_quat(*a, **kw))

def th_rot_to_quat(xform):
   raise NotImplemented
   x = np.asarray(xform)
   t0, t1, t2 = x[..., 0, 0], x[..., 1, 1], x[..., 2, 2]
   tr = t0 + t1 + t2
   quat = np.empty(x.shape[:-2] + (4, ))

   case0 = tr > 0
   S0 = np.sqrt(tr[case0] + 1) * 2
   quat[case0, 0] = 0.25 * S0
   quat[case0, 1] = (x[case0, 2, 1] - x[case0, 1, 2]) / S0
   quat[case0, 2] = (x[case0, 0, 2] - x[case0, 2, 0]) / S0
   quat[case0, 3] = (x[case0, 1, 0] - x[case0, 0, 1]) / S0

   case1 = ~case0 * (t0 >= t1) * (t0 >= t2)
   S1 = np.sqrt(1.0 + x[case1, 0, 0] - x[case1, 1, 1] - x[case1, 2, 2]) * 2
   quat[case1, 0] = (x[case1, 2, 1] - x[case1, 1, 2]) / S1
   quat[case1, 1] = 0.25 * S1
   quat[case1, 2] = (x[case1, 0, 1] + x[case1, 1, 0]) / S1
   quat[case1, 3] = (x[case1, 0, 2] + x[case1, 2, 0]) / S1

   case2 = ~case0 * (t1 > t0) * (t1 >= t2)
   S2 = np.sqrt(1.0 + x[case2, 1, 1] - x[case2, 0, 0] - x[case2, 2, 2]) * 2
   quat[case2, 0] = (x[case2, 0, 2] - x[case2, 2, 0]) / S2
   quat[case2, 1] = (x[case2, 0, 1] + x[case2, 1, 0]) / S2
   quat[case2, 2] = 0.25 * S2
   quat[case2, 3] = (x[case2, 1, 2] + x[case2, 2, 1]) / S2

   case3 = ~case0 * (t2 > t0) * (t2 > t1)
   S3 = np.sqrt(1.0 + x[case3, 2, 2] - x[case3, 0, 0] - x[case3, 1, 1]) * 2
   quat[case3, 0] = (x[case3, 1, 0] - x[case3, 0, 1]) / S3
   quat[case3, 1] = (x[case3, 0, 2] + x[case3, 2, 0]) / S3
   quat[case3, 2] = (x[case3, 1, 2] + x[case3, 2, 1]) / S3
   quat[case3, 3] = 0.25 * S3

   assert (np.sum(case0) + np.sum(case1) + np.sum(case2) + np.sum(case3) == np.prod(xform.shape[:-2]))

   return quat_to_upper_half(quat)

th_xform_to_quat = th_rot_to_quat

def th_is_valid_quat_rot(quat):
   assert quat.shape[-1] == 4
   return np.isclose(1, torch.linalg.norm(quat, axis=-1))

def th_quat_to_upper_half(quat):
   ineg0 = (quat[..., 0] < 0)
   ineg1 = (quat[..., 0] == 0) * (quat[..., 1] < 0)
   ineg2 = (quat[..., 0] == 0) * (quat[..., 1] == 0) * (quat[..., 2] < 0)
   ineg3 = ((quat[..., 0] == 0) * (quat[..., 1] == 0) * (quat[..., 2] == 0) * (quat[..., 3] < 0))
   # ic(ineg0.shape)
   # ic(ineg1.shape)
   # ic(ineg2.shape)
   # ic(ineg3.shape)
   ineg = ineg0 + ineg1 + ineg2 + ineg3
   quat2 = torch.where(ineg, -quat, quat)
   return th_normalized(quat2)

def th_homog(rot, trans=None, **kw):
   if trans is None:
      trans = torch.as_tensor([0, 0, 0, 0], **kw)
   trans = torch.as_tensor(trans)

   if rot.shape == (3, 3):
      rot = torch.cat([rot, torch.tensor([[0., 0., 0.]])], axis=0)
      rot = torch.cat([rot, torch.tensor([[0], [0], [0], [1]])], axis=1)

   assert rot.shape[-2:] == (4, 4)
   assert trans.shape[-1:] == (4, )

   h = torch.cat([rot[:, :3], trans[:, None]], axis=1)
   return h

def th_quat_to_rot(quat):
   assert quat.shape[-1] == 4
   qr = quat[..., 0]
   qi = quat[..., 1]
   qj = quat[..., 2]
   qk = quat[..., 3]

   rot = torch.cat([
      torch.tensor([[
         1 - 2 * (qj**2 + qk**2),
         2 * (qi * qj - qk * qr),
         2 * (qi * qk + qj * qr),
      ]]),
      torch.tensor([[
         2 * (qi * qj + qk * qr),
         1 - 2 * (qi**2 + qk**2),
         2 * (qj * qk - qi * qr),
      ]]),
      torch.tensor([[
         2 * (qi * qk - qj * qr),
         2 * (qj * qk + qi * qr),
         1 - 2 * (qi**2 + qj**2),
      ]])
   ])
   # ic(rot.shape)
   return rot

def th_quat_to_xform(quat, dtype='f8'):
   r = th_quat_to_rot(quat, dtype)
   r = torch.cat([r])
   return r

def t_rot(axis, angle, shape=(3, 3), squeeze=True):

   # axis = torch.tensor(axis, dtype=dtype, requires_grad=requires_grad)
   # angle = angle * np.pi / 180.0 if degrees else angle
   # angle = torch.tensor(angle, dtype=dtype, requires_grad=requires_grad)

   if axis.ndim == 1: axis = axis[None, ]
   if angle.ndim == 0: angle = angle[None, ]
   # if angle.ndim == 0
   if axis.shape and angle.shape and not is_broadcastable(axis.shape[:-1], angle.shape):
      raise ValueError(f'axis/angle not compatible: {axis.shape} {angle.shape}')
   zero = torch.zeros(*angle.shape)
   axis = th_normalized(axis)
   a = torch.cos(angle / 2.0)
   tmp = axis * -torch.sin(angle / 2)[..., None]
   b, c, d = tmp[..., 0], tmp[..., 1], tmp[..., 2]
   aa, bb, cc, dd = a * a, b * b, c * c, d * d
   bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
   if shape == (3, 3):
      rot = torch.stack([
         torch.stack([aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)], axis=-1),
         torch.stack([2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)], axis=-1),
         torch.stack([2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc], axis=-1),
      ], axis=-2)
   elif shape == (4, 4):
      rot = torch.stack([
         torch.stack([aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac), zero], axis=-1),
         torch.stack([2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab), zero], axis=-1),
         torch.stack([2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc, zero], axis=-1),
         torch.stack([zero, zero, zero, zero + 1], axis=-1),
      ], axis=-2)
   else:
      raise ValueError(f't_rot shape must be (3,3) or (4,4), not {shape}')
   # ic('foo')
   # ic(axis.shape)
   # ic(angle.shape)
   # ic(rot.shape)
   if squeeze and rot.shape == (1, 3, 3): rot = rot.reshape(3, 3)
   if squeeze and rot.shape == (1, 4, 4): rot = rot.reshape(4, 4)
   return rot

def th_rms(a, b):
   assert a.shape == b.shape
   return torch.sqrt(torch.sum(torch.square(a - b)) / len(a))

def th_xform(xform, stuff, homogout='auto', **kw):
   xform = torch.as_tensor(xform).to(stuff.dtype)
   nothomog = stuff.shape[-1] == 3
   if stuff.shape[-1] == 3:
      stuff = th_point(stuff)
   result = _hxform_impl(xform, stuff, **kw)
   if homogout is False or homogout == 'auto' and nothomog:
      result = result[..., :3]
   return result

def th_rmsfit(mobile, target):
   '''use kabsch method to get rmsd fit'''
   assert mobile.shape == target.shape
   assert mobile.ndim > 1
   assert mobile.shape[-1] in (3, 4)
   if mobile.dtype != target.dtype:
      mobile = mobile.to(target.dtype)
   mobile_cen = torch.mean(mobile, axis=0)
   target_cen = torch.mean(target, axis=0)
   mobile = mobile - mobile_cen
   target = target - target_cen
   # ic(mobile.shape)
   # ic(target.shape[-1] in (3, 4))
   covariance = mobile.T[:3] @ target[:, :3]
   V, S, W = torch.linalg.svd(covariance)
   if 0 > torch.det(V) * torch.det(W):
      S = torch.tensor([S[0], S[1], -S[2]], dtype=S.dtype)
      # S[-1] = -S[-1]
      # ic(S - S1)
      V = torch.cat([V[:, :-1], -V[:, -1, None]], dim=1)
      # V[:, -1] = -V[:, -1]
      # ic(V - V1)
      # assert 0
   rot_m2t = th_homog(V @ W).T
   trans_m2t = target_cen - rot_m2t @ mobile_cen
   xform_mobile_to_target = th_homog(rot_m2t, trans_m2t)

   mobile = mobile + mobile_cen
   target = target + target_cen
   mobile_fit_to_target = th_xform(xform_mobile_to_target, mobile)
   rms = th_rms(target, mobile_fit_to_target)

   return rms, mobile_fit_to_target, xform_mobile_to_target

def th_randpoint(shape=(), cen=[0, 0, 0], std=1, dtype=None):
   dtype = dtype or torch.float32
   cen = torch.as_tensor(cen)
   if isinstance(shape, int): shape = (shape, )
   p = th_point(torch.randn(*(shape) + (3, )) * std + cen)
   return p

def th_randvec(shape=(), std=1, dtype=None):
   dtype = dtype or torch.float32
   if isinstance(shape, int): shape = (shape, )
   return th_vec(torch.randn(*(shape + (3, ))) * std)

def th_randunit(shape=(), cen=[0, 0, 0], std=1):
   dtype = dtype or torch.float32
   if isinstance(shape, int): shape = (shape, )
   v = th_normalized(torch.randn(*(shape + (3, ))) * std)
   return v

def th_point(point, **kw):
   point = torch.as_tensor(point)
   shape = point.shape[:-1]
   points = torch.cat([point[..., :3], torch.ones(shape + (1, ))], axis=-1)
   if points.dtype not in (torch.float32, torch.float64):
      points = points.to(torch.float32)
   return points

def th_vec(vec):
   vec = torch.as_tensor(vec)
   if (vec.dtype not in (torch.float32, torch.float64)):
      vec = vec.to(torch.float32)
   if vec.shape[-1] == 4:
      if torch.any(vec[..., 3]) != 0:
         vec = torch.cat([vec[..., :3], torch.zeros(*vec.shape[:-1], 1)], dim=-1)
      return vec
   elif vec.shape[-1] == 3:
      r = torch.zeros(vec.shape[:-1] + (4, ), dtype=vec.dtype)
      r[..., :3] = vec
      return r
   else:
      raise ValueError('vec must len 3 or 4')

def th_normalized(a):
   return torch.nn.functional.normalize(a, dim=-1)
   # a = torch.as_tensor(a)
   # if (not a.shape and len(a) == 3) or (a.shape and a.shape[-1] == 3):
   #    a, tmp = torch.zeros(a.shape[:-1] + (4, ), dtype=a.type), a
   #    a[..., :3] = tmp
   # a2 = a[:]
   # a2[..., 3] = 0
   # return a2 / th_norm(a2)[..., None]

def th_norm(a):
   a = torch.as_tensor(a)
   return torch.sqrt(torch.sum(a[..., :3] * a[..., :3], axis=-1))

def th_norm2(a):
   a = torch.as_tensor(a)
   return torch.sum(a[..., :3] * a[..., :3], axis=-1)

def th_axis_angle_hel(xforms):
   axis, angle = th_axis_angle(xforms)
   hel = th_dot(axis, xforms[..., :, 3])
   return axis, angle, hel

def th_axis_angle_cen_hel(xforms):
   axis, angle, cen = th_axis_angle_cen(xforms)
   hel = th_dot(axis, xforms[..., :, 3])
   return axis, angle, cen, hel

def th_axis_angle(xforms):
   axis = th_axis(xforms)
   angl = th_angle(xforms)
   return axis, angl

def th_axis(xforms):
   if xforms.shape[-2:] == (4, 4):
      return th_normalized(
         torch.stack((
            xforms[..., 2, 1] - xforms[..., 1, 2],
            xforms[..., 0, 2] - xforms[..., 2, 0],
            xforms[..., 1, 0] - xforms[..., 0, 1],
            torch.zeros(xforms.shape[:-2]),
         ), axis=-1))
   if xforms.shape[-2:] == (3, 3):
      return th_normalized(
         torch.stack((
            xforms[..., 2, 1] - xforms[..., 1, 2],
            xforms[..., 0, 2] - xforms[..., 2, 0],
            xforms[..., 1, 0] - xforms[..., 0, 1],
         ), axis=-1))
   else:
      raise ValueError('wrong shape for xform/rotation matrix: ' + str(xforms.shape))

def th_angle(xforms):
   tr = xforms[..., 0, 0] + xforms[..., 1, 1] + xforms[..., 2, 2]
   cos = (tr - 1.0) / 2.0
   angl = torch.arccos(torch.clip(cos, -1, 1))
   return angl

def th_point_line_dist2(point, cen, norm):
   point = point - cen
   hproj = norm * torch.sum(norm * point) / torch.sum(norm * norm)
   perp = point - hproj
   return torch.sum(perp**2)

def th_dot(a, b, outerprod=False):
   if outerprod:
      shape1 = a.shape[:-1]
      shape2 = b.shape[:-1]
      a = a.reshape((1, ) * len(shape2) + shape1 + (-1, ))
      b = b.reshape(shape2 + (1, ) * len(shape1) + (-1, ))
   return torch.sum(a[..., :3] * b[..., :3], axis=-1)

def th_point_in_plane(point, normal, pt):
   inplane = torch.abs(th_dot(normal[..., :3], pt[..., :3] - point[..., :3]))
   return inplane < 0.00001

def th_ray_in_plane(point, normal, p1, n1):
   inplane1 = th_point_in_plane(point, normal, p1)
   inplane2 = th_point_in_plane(point, normal, p1 + n1)
   return inplane1 and inplane2

def th_intersect_planes(p1, n1, p2, n2):
   """
   intersect_Planes: find the 3D intersection of two planes
      Input:  two planes represented (point, normal) as (p1,n1), (p2,n2)
      Output: L = the intersection line (when it exists)
      Return: rays shape=(...,4,2), status
              0 = intersection returned
              1 = disjoint (no intersection)
              2 = the two planes coincide
   """
   """intersect two planes
   :param plane1: first plane represented by ray
   :type plane2: np.array shape=(..., 4, 2) 
   :param plane1: second planes represented by rays
   :type plane2: np.array shape=(..., 4, 2) 
   :return: line: np.array shape=(...,4,2), status: int (0 = intersection returned, 1 = no intersection, 2 = the two planes coincide)
   """
   origshape = p1.shape
   # shape = origshape[:-1] or [1]
   assert p1.shape[-1] == 4
   assert p1.shape == n1.shape
   assert p1.shape == p2.shape
   assert p1.shape == n2.shape
   p1 = p1.reshape(-1, 4)
   n1 = n1.reshape(-1, 4)
   p2 = p2.reshape(-1, 4)
   n2 = n2.reshape(-1, 4)
   N = len(p1)

   u = torch.cross(n1[..., :3], n2[..., :3])
   abs_u = torch.abs(u)
   planes_parallel = torch.sum(abs_u, axis=-1) < 0.000001
   p2_in_plane1 = th_point_in_plane(p1, n1, p2)
   status = torch.zeros(N)
   status[planes_parallel] = 1
   status[planes_parallel * p2_in_plane1] = 2
   d1 = -th_dot(n1, p1)
   d2 = -th_dot(n2, p2)

   amax = torch.argmax(abs_u, axis=-1)
   sel = amax == 0, amax == 1, amax == 2
   perm = torch.cat([
      torch.where(sel[0])[0],
      torch.where(sel[1])[0],
      torch.where(sel[2])[0],
   ])
   perminv = torch.empty_like(perm)
   perminv[perm] = torch.arange(len(perm))
   breaks = np.cumsum([0, sum(sel[0]), sum(sel[1]), sum(sel[2])])
   n1 = n1[perm]
   n2 = n2[perm]
   d1 = d1[perm]
   d2 = d2[perm]
   up = u[perm]

   zeros = torch.zeros(N)
   ones = torch.ones(N)
   l = []

   s = slice(breaks[0], breaks[1])
   y = (d2[s] * n1[s, 2] - d1[s] * n2[s, 2]) / up[s, 0]
   z = (d1[s] * n2[s, 1] - d2[s] * n1[s, 1]) / up[s, 0]
   l.append(torch.stack([zeros[s], y, z, ones[s]], axis=-1))

   s = slice(breaks[1], breaks[2])
   z = (d2[s] * n1[s, 0] - d1[s] * n2[s, 0]) / up[s, 1]
   x = (d1[s] * n2[s, 2] - d2[s] * n1[s, 2]) / up[s, 1]
   l.append(torch.stack([x, zeros[s], z, ones[s]], axis=-1))

   s = slice(breaks[2], breaks[3])
   x = (d2[s] * n1[s, 1] - d1[s] * n2[s, 1]) / up[s, 2]
   y = (d1[s] * n2[s, 0] - d2[s] * n1[s, 0]) / up[s, 2]
   l.append(torch.stack([x, y, zeros[s], ones[s]], axis=-1))

   isect_pt = torch.cat(l)
   isect_pt = isect_pt[perminv]
   isect_pt = isect_pt.reshape(origshape)

   isect_dirn = th_normalized(torch.cat([u, torch.zeros(N, 1)], axis=-1))
   isect_dirn = isect_dirn.reshape(origshape)

   return isect_pt, isect_dirn, status

def is_broadcastable(shape1, shape2):
   for a, b in zip(shape1[::-1], shape2[::-1]):
      if a == 1 or b == 1 or a == b:
         pass
      else:
         return False
   return True

def axis_ang_cen_magic_points_torch():
   return torch.from_numpy(wu.homog.hgeom._axis_ang_cen_magic_points_numpy).float()
