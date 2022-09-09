import itertools as it, functools as ft
import deferred_import

np = deferred_import.deferred_import('numpy')
torch = deferred_import.deferred_import('torch')

I = np.eye(4)
Ux = np.array([1, 0, 0, 0])
Uy = np.array([0, 1, 0, 0])
Uz = np.array([0, 0, 1, 0])

def th_axis_ang_cen(xforms):
   axis, angle = th_axis_angle(xforms)
   ev, cen = torch.linalg.eig(xforms)
   cen = torch.real(cen[..., 3])
   cen = cen / cen[..., 3][..., None]
   return axis, angle, cen

def th_rot(axis, angle, center=None, hel=None):

   rot = t_rot(axis, angle, shape=(4, 4))
   if center is None:
      center = torch.tensor([0, 0, 0, 1], dtype=torch.float)
   if hel is None:
      hel = torch.tensor([0], dtype=torch.float)

   x, y, z = center[..., 0], center[..., 1], center[..., 2]
   center = torch.stack([
      x - rot[..., 0, 0] * x - rot[..., 0, 1] * y - rot[..., 0, 2] * z,
      y - rot[..., 1, 0] * x - rot[..., 1, 1] * y - rot[..., 1, 2] * z,
      z - rot[..., 2, 0] * x - rot[..., 2, 1] * y - rot[..., 2, 2] * z,
      torch.tensor(1.0),
   ])
   center = center + axis * hel
   r = torch.cat([rot[:, :3], center[:, None]], axis=1)
   return r

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

   assert (np.sum(case0) + np.sum(case1) + np.sum(case2) + np.sum(case3) == np.prod(
      xform.shape[:-2]))

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
   # print(ineg0.shape)
   # print(ineg1.shape)
   # print(ineg2.shape)
   # print(ineg3.shape)
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
   print(rot.shape)
   return rot

def th_quat_to_xform(quat, dtype='f8'):
   r = th_quat_to_rot(quat, dtype)
   r = torch.cat([r])
   return r

def t_rot(axis, angle, shape=(3, 3)):

   # axis = torch.tensor(axis, dtype=dtype, requires_grad=requires_grad)
   # angle = angle * np.pi / 180.0 if degrees else angle
   # angle = torch.tensor(angle, dtype=dtype, requires_grad=requires_grad)

   if axis.shape and angle.shape and not is_broadcastable(axis.shape[:-1], angle.shape):
      raise ValueError('axis and angle not compatible: ' + str(axis.shape) + ' ' +
                       str(angle.shape))
   zero = torch.tensor(0)
   axis = th_normalized(axis)
   a = torch.cos(angle / 2.0)
   tmp = axis * -torch.sin(angle / 2)[..., None]
   b, c, d = tmp[..., 0], tmp[..., 1], tmp[..., 2]
   aa, bb, cc, dd = a * a, b * b, c * c, d * d
   bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
   if shape == (3, 3):
      rot = torch.stack([
         torch.stack([aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)]),
         torch.stack([2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)]),
         torch.stack([2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]),
      ])
   elif shape == (4, 4):
      rot = torch.stack([
         torch.stack([aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac), zero]),
         torch.stack([2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab), zero]),
         torch.stack([2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc, zero]),
         torch.stack([zero, zero, zero, zero + 1]),
      ])
   else:
      raise ValueError(f't_rot shape must be 3,3 or 4,4')

   return rot

def th_rms(a, b):
   assert a.shape == b.shape
   return torch.sqrt(torch.sum(torch.square(a - b)) / len(a))

def th_xform(xform, stuff, **kw):
   xform = xform.to(stuff.dtype)
   return _hxform_impl(xform, stuff, **kw)

def th_rmsfit(mobile, target):
   'use kabsch method to get rmsd fit'
   mobile_cen = torch.mean(mobile, axis=0)
   target_cen = torch.mean(target, axis=0)
   mobile = mobile - mobile_cen
   target = target - target_cen

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
   v = th_normalized(np.random.randn(*(shape + (3, ))) * std)
   return v

def th_point(point, **kw):
   if point.shape[-1] == 4:
      return point
   elif point.shape[-1] == 3:
      r = torch.ones(point.shape[:-1] + (4, ), **kw)
      r[..., :3] = point
      return r
   elif point.shape[-2:] == (4, 4):
      return point[..., :, 3]
   else:
      raise ValueError('point must len 3 or 4')

def th_vec(vec):
   vec = torch.as_tensor(vec)
   if vec.shape[-1] == 4:
      vec[..., 3] = 0
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

def th_axis_angle_hel(xforms):
   axis, angle = th_axis_angle(xforms)
   hel = th_dot(axis, xforms[..., :, 3])
   return axis, angle, hel

def th_axis_angle_cen_hel(xforms):
   axis, angle, cen = th_axis_ang_cen(xforms)
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
   assert p1.shape == (4, )
   assert n1.shape == (4, )
   assert p2.shape == (4, )
   assert n2.shape == (4, )
   u = torch.cross(n1[:3], n2[:3])
   abs_u = torch.abs(u)
   planes_parallel = torch.sum(abs_u, axis=-1) < 0.000001
   p2_in_plane1 = th_point_in_plane(p1, n1, p2)
   status = torch.zeros(1)
   status[planes_parallel] = 1
   status[planes_parallel * p2_in_plane1] = 2
   d1 = -th_dot(n1, p1)
   d2 = -th_dot(n2, p2)
   amax = torch.argmax(abs_u, axis=-1)
   sel0, sel1, sel2 = amax == 0, amax == 1, amax == 2
   n1a, n2a, d1a, d2a, ua = (x[sel0] for x in (n1, n2, d1, d2, u))
   n1b, n2b, d1b, d2b, ub = (x[sel1] for x in (n1, n2, d1, d2, u))
   n1c, n2c, d1c, d2c, uc = (x[sel2] for x in (n1, n2, d1, d2, u))

   ay = (d2a * n1a[..., 2] - d1a * n2a[..., 2]) / ua[..., 0]
   az = (d1a * n2a[..., 1] - d2a * n1a[..., 1]) / ua[..., 0]
   bz = (d2b * n1b[..., 0] - d1b * n2b[..., 0]) / ub[..., 1]
   bx = (d1b * n2b[..., 2] - d2b * n1b[..., 2]) / ub[..., 1]
   cx = (d2c * n1c[..., 1] - d1c * n2c[..., 1]) / uc[..., 2]
   cy = (d1c * n2c[..., 0] - d2c * n1c[..., 0]) / uc[..., 2]
   zero = torch.tensor([0.])
   one = torch.tensor([1.])
   if sel0:
      isect_pt = torch.cat([zero, ay, az, one])
   elif sel1:
      isect_pt = torch.cat([bx, zero, bz, one])
   elif sel2:
      isect_pt = torch.cat([cx, cy, zero, one])
   else:
      assert 0
   # isect_pt = torch.empty(shape[:-2] + (3, ), dtype=plane1.dtype)
   # isect_pt[sel0, 0] = 0
   # isect_pt[sel0, 1] = ay
   # isect_pt[sel0, 2] = az
   # isect_pt[sel1, 0] = bx
   # isect_pt[sel1, 1] = 0
   # isect_pt[sel1, 2] = bz
   # isect_pt[sel2, 0] = cx
   # isect_pt[sel2, 1] = cy
   # isect_pt[sel2, 2] = 0
   # isect = hray(isect_pt, u)
   isect_normal = th_normalized(torch.cat([u, zero]))
   return isect_pt, isect_normal, status

###############################################################################

def hdist(x, y):
   shape1 = x.shape[:-2]
   shape2 = y.shape[:-2]
   a = x.reshape(shape1 + (1, ) * len(shape1) + (4, 4))
   b = y.reshape((1, ) * len(shape2) + shape2 + (4, 4))
   dist = np.linalg.norm(a[..., :, 3] - b[..., :, 3], axis=-1)
   return dist

def hdiff(x, y, lever):
   shape1 = x.shape[:-2]
   shape2 = y.shape[:-2]
   a = x.reshape(shape1 + (1, ) * len(shape1) + (4, 4))
   b = y.reshape((1, ) * len(shape2) + shape2 + (4, 4))

   # xdelta = hinv(a) @ b
   # dist = np.linalg.norm(xdelta[..., 3], axis=-1)
   # dist = np.linalg.norm(a[..., 3] - b[..., 3], axis=-1)
   # ang = angle_of(xdelta) * lever

   axyz = a[..., :3, :3] * lever + a[..., :3, 3, None]
   bxyz = b[..., :3, :3] * lever + b[..., :3, 3, None]

   # print(axyz[..., :, 0])
   # print(axyz[..., :, 1])
   # print(axyz[..., :, 2])

   # print(bxyz[..., :, 0])
   # print(bxyz[..., :, 1])
   # print(bxyz[..., :, 2])

   # print(axyz[..., 0, :] - bxyz[..., 0, :])
   # print(axyz[..., 1, :] - bxyz[..., 1, :])
   # print(axyz[..., 2, :] - bxyz[..., 2, :])
   diff = np.linalg.norm(axyz - bxyz, axis=-1)
   diff = np.mean(diff, axis=-1)
   # print(diff)
   # assert 0

   return diff

def hxform(x, stuff, **kw):
   x = np.asarray(x)
   # print('hxform', x.shape, stuff.shape, outerprod, flat)
   stuff = np.asarray(stuff)
   return _hxform_impl(x, stuff, **kw)

def _hxform_impl(x, stuff, outerprod=False, flat=False):
   if stuff.shape[-1] == 3:
      stuff = hpoint(stuff)
   assert x.shape[-2:] == (4, 4)
   if stuff.shape[-2:] == (4, 4):
      if outerprod:
         shape1 = x.shape[:-2]
         shape2 = stuff.shape[:-2]
         a = x.reshape(shape1 + (1, ) * len(shape2) + (4, 4))
         b = stuff.reshape((1, ) * len(shape1) + shape2 + (4, 4))
         result = a @ b
         if flat:
            result = result.reshape(-1, 4, 4)
      else:
         result = x @ stuff
   elif stuff.shape[-2:] == (4, 1) or stuff.shape[-1] == 4:
      if stuff.shape[-1] != 1:
         stuff = stuff[..., None]
      if outerprod:
         shape1 = x.shape[:-2]
         shape2 = stuff.shape[:-2]
         # print(x.shape, stuff.shape)
         a = x.reshape(shape1 + (1, ) * len(shape2) + (4, 4))
         b = stuff.reshape((1, ) * len(shape1) + shape2 + (4, 1))
         result = a @ b

      # if stuff.shape[-1] == 4:
      # stuff = stuff[..., None]
      # stuff = stuff[None]
      else:
         result = x @ stuff
      result = result.squeeze()
      if flat:
         result = result.reshape(-1, 4)
      # assert 0
   else:
      raise ValueError(f'hxform cant understand shape {stuff.shape}')
   # print('result', result.shape)
   return result

def is_valid_quat_rot(quat):
   assert quat.shape[-1] == 4
   return np.isclose(1, np.linalg.norm(quat, axis=-1))

def quat_to_upper_half(quat):
   ineg0 = (quat[..., 0] < 0)
   ineg1 = (quat[..., 0] == 0) * (quat[..., 1] < 0)
   ineg2 = (quat[..., 0] == 0) * (quat[..., 1] == 0) * (quat[..., 2] < 0)
   ineg3 = ((quat[..., 0] == 0) * (quat[..., 1] == 0) * (quat[..., 2] == 0) * (quat[..., 3] < 0))
   # print(ineg0.shape)
   # print(ineg1.shape)
   # print(ineg2.shape)
   # print(ineg3.shape)
   ineg = ineg0 + ineg1 + ineg2 + ineg3
   quat = quat.copy()
   quat[ineg] = -quat[ineg]
   return quat

def rand_quat(shape=(), seed=None):
   if seed is not None:
      randstate = np.random.get_state()
      np.random.seed(seed)

   if isinstance(shape, int): shape = (shape, )
   q = np.random.randn(*shape, 4)
   q /= np.linalg.norm(q, axis=-1)[..., np.newaxis]
   if seed is not None: np.random.set_state(randstate)
   return quat_to_upper_half(q)

def rot_to_quat(xform):
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

   assert (np.sum(case0) + np.sum(case1) + np.sum(case2) + np.sum(case3) == np.prod(
      xform.shape[:-2]))

   return quat_to_upper_half(quat)

xform_to_quat = rot_to_quat

def quat_to_rot(quat, dtype='f8', shape=(3, 3)):
   quat = np.asarray(quat)
   assert quat.shape[-1] == 4
   qr = quat[..., 0]
   qi = quat[..., 1]
   qj = quat[..., 2]
   qk = quat[..., 3]
   outshape = quat.shape[:-1]
   rot = np.zeros(outshape + shape, dtype=dtype)
   rot[..., 0, 0] = 1 - 2 * (qj**2 + qk**2)
   rot[..., 0, 1] = 2 * (qi * qj - qk * qr)
   rot[..., 0, 2] = 2 * (qi * qk + qj * qr)
   rot[..., 1, 0] = 2 * (qi * qj + qk * qr)
   rot[..., 1, 1] = 1 - 2 * (qi**2 + qk**2)
   rot[..., 1, 2] = 2 * (qj * qk - qi * qr)
   rot[..., 2, 0] = 2 * (qi * qk - qj * qr)
   rot[..., 2, 1] = 2 * (qj * qk + qi * qr)
   rot[..., 2, 2] = 1 - 2 * (qi**2 + qj**2)
   return rot

def quat_to_xform(quat, dtype='f8'):
   r = quat_to_rot(quat, dtype, shape=(4, 4))
   r[..., 3, 3] = 1
   return r

def quat_multiply(q, r):
   q, r = np.broadcast_arrays(q, r)
   q0, q1, q2, q3 = np.moveaxis(q, -1, 0)
   r0, r1, r2, r3 = np.moveaxis(r, -1, 0)
   assert np.all(q1 == q[..., 1])
   t = np.empty_like(q)
   t[..., 0] = r0 * q0 - r1 * q1 - r2 * q2 - r3 * q3
   t[..., 1] = r0 * q1 + r1 * q0 - r2 * q3 + r3 * q2
   t[..., 2] = r0 * q2 + r1 * q3 + r2 * q0 - r3 * q1
   t[..., 3] = r0 * q3 - r1 * q2 + r2 * q1 + r3 * q0
   return t

def h_rand_points(shape=(1, ), seed=None):
   if seed is not None:
      randstate = np.random.get_state()
      np.random.seed(seed)

   pts = np.ones(shape + (4, ))
   pts[..., 0] = np.random.randn(*shape)
   pts[..., 1] = np.random.randn(*shape)
   pts[..., 2] = np.random.randn(*shape)
   if seed is not None: np.random.set_state(randstate)
   return pts

def guess_is_degrees(angle):
   return np.max(np.abs(angle)) > 2 * np.pi

def is_broadcastable(shp1, shp2):
   for a, b in zip(shp1[::-1], shp2[::-1]):
      if a == 1 or b == 1 or a == b:
         pass
      else:
         return False
   return True

def fast_axis_of(xforms):
   if xforms.shape[-2:] == (4, 4):

      return np.stack((
         xforms[..., 2, 1] - xforms[..., 1, 2],
         xforms[..., 0, 2] - xforms[..., 2, 0],
         xforms[..., 1, 0] - xforms[..., 0, 1],
         np.zeros(xforms.shape[:-2]),
      ), axis=-1)
   if xforms.shape[-2:] == (3, 3):
      return np.stack((
         xforms[..., 2, 1] - xforms[..., 1, 2],
         xforms[..., 0, 2] - xforms[..., 2, 0],
         xforms[..., 1, 0] - xforms[..., 0, 1],
      ), axis=-1)
   else:
      raise ValueError('wrong shape for xform/rotation matrix: ' + str(xforms.shape))

def axis_of(xforms, tol=1e-7, debug=False):
   xdim = xforms.shape[-1]
   origshape = xforms.shape
   if xforms.ndim != 3:
      xforms = xforms.reshape(-1, xdim, xdim)

   axs = fast_axis_of(xforms)
   norm = np.linalg.norm(axs, axis=-1)
   is180 = (norm < tol)
   axs[~is180] = axs[~is180] / norm[~is180].reshape(-1, 1)
   if np.sum(is180) > 0:
      x180 = xforms[is180]
      is_ident = np.all(np.isclose(np.eye(3), x180[:, :3, :3], atol=tol), axis=(-2, -1))
      axs[np.where(is180)[0][is_ident]] = [1, 0, 0, 0]
      is180[np.where(is180)[0][is_ident]] = False
      x180 = x180[~is_ident]

      eig = np.linalg.eig(x180[..., :3, :3])
      eigval, eigvec = np.real(eig[0]), np.real(eig[1])
      eigval_is_1 = np.abs(eigval - 1) < 1e-6
      ixform, ieigval = np.where(eigval_is_1)

      # print(ixform)
      # print(ieigval)
      try:
         axs[is180, :3] = eigvec[ixform, :, ieigval]
      except Exception as e:

         # print(is_ident)
         # print(is180)
         # print(x180.shape)
         for a, b, c in zip(eigval, eigvec, eigval_is_1):
            print(a)
            print(b)
            print(c)
         print()
         raise e
      # assert 0

      if debug:
         n_unit_eigval_1 = np.sum(np.abs(eigval - 1) < tol, axis=-1) == 1
         n_unit_eigval_3 = np.sum(np.abs(eigval - 1) < tol, axis=-1) == 3
         np.all(np.logical_or(n_unit_eigval_1, n_unit_eigval_3))
         # assert np.allclose(np.all(np.sum(np.abs(eigval - 1) < tol, axis=-1) == 1)

   return axs.reshape(origshape[:-1])

def is_homog_xform(xforms):
   return ((xforms.shape[-2:] == (4, 4)) and (np.allclose(1, np.linalg.det(xforms[..., :3, :3])))
           and (np.allclose(xforms[..., 3, :], [0, 0, 0, 1])))

def hinv(xforms):
   return np.linalg.inv(xforms)

def axis_angle_of(xforms, debug=False):
   axis = axis_of(xforms, debug=debug)
   angl = angle_of(xforms, debug=debug)
   return axis, angl

def axis_angle_hel_of(xforms):
   axis, angle = axis_angle_of(xforms)
   hel = hdot(axis, trans_of(xforms))
   return axis, angle, hel

def angle_of(xforms, debug=False):
   tr = xforms[..., 0, 0] + xforms[..., 1, 1] + xforms[..., 2, 2]
   cos = (tr - 1.0) / 2.0
   angl = np.arccos(np.clip(cos, -1, 1))
   return angl

def angle_of_degrees(xforms, debug=False):
   tr = xforms[..., 0, 0] + xforms[..., 1, 1] + xforms[..., 2, 2]
   cos = (tr - 1.0) / 2.0
   angl = np.arccos(np.clip(cos, -1, 1))
   return np.degrees(angl)

def rot(axis, angle, degrees='auto', dtype='f8', shape=(3, 3)):
   axis = np.array(axis, dtype=dtype)
   angle = np.array(angle, dtype=dtype)
   if degrees == 'auto': degrees = guess_is_degrees(angle)
   angle = angle * np.pi / 180.0 if degrees else angle
   if axis.shape and angle.shape and not is_broadcastable(axis.shape[:-1], angle.shape):
      raise ValueError('axis and angle not compatible: ' + str(axis.shape) + ' ' +
                       str(angle.shape))
   axis /= np.linalg.norm(axis, axis=-1)[..., np.newaxis]
   a = np.cos(angle / 2.0)
   tmp = axis * -np.sin(angle / 2)[..., np.newaxis]
   b, c, d = tmp[..., 0], tmp[..., 1], tmp[..., 2]
   aa, bb, cc, dd = a * a, b * b, c * c, d * d
   bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
   outshape = angle.shape if angle.shape else axis.shape[:-1]
   rot3 = np.zeros(outshape + shape, dtype=dtype)
   rot3[..., 0, 0] = aa + bb - cc - dd
   rot3[..., 0, 1] = 2 * (bc + ad)
   rot3[..., 0, 2] = 2 * (bd - ac)
   rot3[..., 1, 0] = 2 * (bc - ad)
   rot3[..., 1, 1] = aa + cc - bb - dd
   rot3[..., 1, 2] = 2 * (cd + ab)
   rot3[..., 2, 0] = 2 * (bd + ac)
   rot3[..., 2, 1] = 2 * (cd - ab)
   rot3[..., 2, 2] = aa + dd - bb - cc
   return rot3

def hrot(axis, angle, center=None, dtype='f8', **kws):
   axis = np.array(axis, dtype=dtype)
   angle = np.array(angle, dtype=dtype)
   center = (np.array([0, 0, 0], dtype=dtype) if center is None else np.array(
      center, dtype=dtype))
   r = rot(axis, angle, dtype=dtype, shape=(4, 4), **kws)
   x, y, z = center[..., 0], center[..., 1], center[..., 2]
   r[..., 0, 3] = x - r[..., 0, 0] * x - r[..., 0, 1] * y - r[..., 0, 2] * z
   r[..., 1, 3] = y - r[..., 1, 0] * x - r[..., 1, 1] * y - r[..., 1, 2] * z
   r[..., 2, 3] = z - r[..., 2, 0] * x - r[..., 2, 1] * y - r[..., 2, 2] * z
   r[..., 3, 3] = 1
   return r

def hpoint(point):
   point = np.asanyarray(point)
   if point.shape[-1] == 4: return point
   elif point.shape[-1] == 3:
      r = np.ones(point.shape[:-1] + (4, ))
      r[..., :3] = point
      return r
   elif point.shape[-2:] == (4, 4):
      return point[..., :, 3]
   else:
      raise ValueError('point must len 3 or 4')

def hvec(vec):
   vec = np.asanyarray(vec)
   if vec.shape[-1] == 4:
      vec[..., 3] = 0
      return vec
   elif vec.shape[-1] == 3:
      r = np.zeros(vec.shape[:-1] + (4, ))
      r[..., :3] = vec
      return r
   else:
      raise ValueError('vec must len 3 or 4')

def hray(origin, direction):
   origin = hpoint(origin)
   direction = hnormalized(direction)
   s = np.broadcast(origin, direction).shape
   r = np.empty(s[:-1] + (4, 2))
   r[..., :origin.shape[-1], 0] = origin
   r[..., 3, 0] = 1
   r[..., :, 1] = direction
   return r

def hframe(u, v, w, cen=None):
   u, v, w = hpoint(u), hpoint(v), hpoint(w)
   assert u.shape == v.shape == w.shape
   if not cen: cen = u
   cen = hpoint(cen)
   assert cen.shape == u.shape
   stubs = np.empty(u.shape[:-1] + (4, 4))
   stubs[..., :, 0] = hnormalized(u - v)
   stubs[..., :, 2] = hnormalized(hcross(stubs[..., :, 0], w - v))
   stubs[..., :, 1] = hcross(stubs[..., :, 2], stubs[..., :, 0])
   stubs[..., :, 3] = hpoint(cen[..., :])
   return stubs

def rot_of(xforms):
   return xforms[..., :3, :3]

def trans_of(xforms):
   return xforms[..., :, 3]

def xaxis_of(xforms):
   return xforms[..., :, 0]

def yaxis_of(xforms):
   return xforms[..., :, 1]

def zaxis_of(xforms):
   return xforms[..., :, 2]

def htrans(trans, dtype='f8'):
   trans = np.asanyarray(trans)
   if trans.shape[-1] == 4:
      trans = trans[..., :3]
   if trans.shape[-1] != 3:
      raise ValueError('trans should be shape (..., 3)')
   tileshape = trans.shape[:-1] + (1, 1)
   t = np.tile(np.identity(4, dtype), tileshape)
   t[..., :trans.shape[-1], 3] = trans
   return t

def hdot(a, b, outerprod=False):
   a = np.asanyarray(a)
   b = np.asanyarray(b)
   if outerprod:
      shape1 = a.shape[:-1]
      shape2 = b.shape[:-1]
      a = a.reshape((1, ) * len(shape2) + shape1 + (-1, ))
      b = b.reshape(shape2 + (1, ) * len(shape1) + (-1, ))
   return np.sum(a[..., :3] * b[..., :3], axis=-1)

def hcross(a, b):
   a = np.asanyarray(a)
   b = np.asanyarray(b)
   c = np.zeros(np.broadcast(a, b).shape, dtype=a.dtype)
   c[..., :3] = np.cross(a[..., :3], b[..., :3])
   return c

def hnorm(a):
   a = np.asanyarray(a)
   return np.sqrt(np.sum(a[..., :3] * a[..., :3], axis=-1))

def hnorm2(a):
   a = np.asanyarray(a)
   return np.sum(a[..., :3] * a[..., :3], axis=-1)

def normalized_3x3(a):
   return a / np.linalg.norm(a, axis=-1)[..., np.newaxis]

def hnormalized(a):
   a = np.asanyarray(a)
   if (not a.shape and len(a) == 3) or (a.shape and a.shape[-1] == 3):
      a, tmp = np.zeros(a.shape[:-1] + (4, )), a
      a[..., :3] = tmp
   a2 = a.copy()
   a2[..., 3] = 0
   return a2 / hnorm(a2)[..., np.newaxis]

def is_valid_rays(r):
   r = np.asanyarray(r)
   if r.shape[-2:] != (4, 2): return False
   if np.any(r[..., 3, :] != (1, 0)): return False
   if np.any(abs(np.linalg.norm(r[..., :3, 1], axis=-1) - 1) > 0.000001):
      return False
   return True

def rand_point(shape=(), seed=None):
   if seed is not None:
      randstate = np.random.get_state()
      np.random.seed(seed)

   if isinstance(shape, int): shape = (shape, )
   p = hpoint(np.random.randn(*(shape + (3, ))))
   if seed is not None: np.random.set_state(randstate)
   return p

def rand_vec(shape=(), seed=None):
   if seed is not None:
      randstate = np.random.get_state()
      np.random.seed(seed)

   if isinstance(shape, int): shape = (shape, )
   if seed is not None: np.random.set_state(randstate)
   return hvec(np.random.randn(*(shape + (3, ))))

def rand_unit(shape=(), seed=None):
   if seed is not None:
      randstate = np.random.get_state()
      np.random.seed(seed)

   if isinstance(shape, int): shape = (shape, )
   v = hnormalized(np.random.randn(*(shape + (3, ))))
   if seed is not None: np.random.set_state(randstate)
   return v

def angle(u, v, outerprod=False):
   u, v = hnormalized(u), hnormalized(v)
   d = hdot(u, v, outerprod=outerprod)
   # todo: handle special cases... 1,-1
   return np.arccos(np.clip(d, -1, 1))

def angle_degrees(u, v):
   return angle(u, v) * 180 / np.pi

def line_angle(u, v, outerprod=False):
   a = angle(u, v, outerprod=outerprod)
   return np.minimum(a, np.pi - a)

def line_angle_degrees(u, v, outerprod=False):
   return np.degrees(line_angle(u, v, outerprod))

def rand_ray(shape=(), cen=(0, 0, 0), sdev=1, seed=None):
   if seed is not None:
      randstate = np.random.get_state()
      np.random.seed(seed)
   if isinstance(shape, int): shape = (shape, )
   cen = np.asanyarray(cen)
   if cen.shape[-1] not in (3, 4):
      raise ValueError('cen must be len 3 or 4')
   shape = shape or cen.shape[:-1]
   cen = cen + np.random.randn(*(shape + (3, ))) * sdev
   norm = np.random.randn(*(shape + (3, )))
   norm /= np.linalg.norm(norm, axis=-1)[..., np.newaxis]
   r = np.zeros(shape + (4, 2))
   r[..., :3, 0] = cen
   r[..., 3, 0] = 1
   r[..., :3, 1] = norm
   if seed is not None: np.random.set_state(randstate)
   return r

def rand_xform_aac(shape=(), axis=None, ang=None, cen=None, seed=None):
   if seed is not None:
      randstate = np.random.get_state()
      np.random.seed(seed)
   if isinstance(shape, int): shape = (shape, )
   if axis is None:
      axis = rand_unit(shape)
   if ang is None:
      ang = np.random.rand(*shape) * np.pi  # todo: make uniform!
   if cen is None:
      cen = rand_point(shape)
   # q = rand_quat(shape)
   if seed is not None: np.random.set_state(randstate)
   return hrot(axis, ang, cen)

def rand_xform_small(shape=(), cart_sd=0.001, rot_sd=0.001, seed=None):
   if seed is not None:
      randstate = np.random.get_state()
      np.random.seed(seed)
   if isinstance(shape, int): shape = (shape, )
   axis = rand_unit(shape)
   ang = np.random.normal(0, rot_sd, shape) * np.pi
   x = hrot(axis, ang, [0, 0, 0, 1], degrees=False).squeeze()
   trans = np.random.normal(0, cart_sd, shape + (3, ))
   x[..., :3, 3] = trans
   if seed is not None: np.random.set_state(randstate)
   return x.squeeze()

def rand_xform(shape=(), cart_cen=0, cart_sd=1, seed=None):
   if seed is not None:
      randstate = np.random.get_state()
      np.random.seed(seed)
   if isinstance(shape, int): shape = (shape, )
   q = rand_quat(shape, )
   x = quat_to_xform(q)
   x[..., :3, 3] = np.random.randn(*shape, 3) * cart_sd + cart_cen
   if seed is not None: np.random.set_state(randstate)
   return x

def rand_rot(shape=(), seed=None):
   if seed is not None:
      randstate = np.random.get_state()
      np.random.seed(seed)

   if isinstance(shape, int): shape = (shape, )
   quat = rand_quat(shape)
   rot = quat_to_rot(quat)
   if seed is not None: np.random.set_state(randstate)
   return rot

def rand_rot_small(shape=(), rot_sd=0.001, seed=None):
   if seed is not None:
      randstate = np.random.get_state()
      np.random.seed(seed)

   if isinstance(shape, int): shape = (shape, )
   axis = rand_unit(shape)
   ang = np.random.normal(0, rot_sd, shape) * np.pi
   r = rot(axis, ang, degrees=False).squeeze()
   if seed is not None: np.random.set_state(randstate)
   return r.squeeze()

def proj(u, v):
   u = np.asanyarray(u)
   v = np.asanyarray(v)
   return hdot(u, v)[..., None] / hnorm2(u)[..., None] * u

def proj_perp(u, v):
   u = np.asanyarray(u)
   v = np.asanyarray(v)
   # return v - hdot(u, v)[..., None] / hnorm2(u)[..., None] * u
   return v - proj(u, v)

def point_in_plane(plane, pt):
   return np.abs(hdot(plane[..., :3, 1], pt[..., :3] - plane[..., :3, 0])) < 0.000001

def ray_in_plane(plane, ray):
   assert ray.shape[-2:] == (4, 2)
   return (point_in_plane(plane, ray[..., :3, 0]) *
           point_in_plane(plane, ray[..., :3, 0] + ray[..., :3, 1]))

def intesect_line_plane(p0, n, l0, l):
   l = hm.hnormalized(l)
   d = hm.hdot(p0 - l0, n) / hm.hdot(l, n)
   return l0 + l * d

def intersect_planes(plane1, plane2):
   """
   intersect_Planes: find the 3D intersection of two planes
      Input:  two planes represented by rays shape=(..., 4, 2)
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
   if not is_valid_rays(plane1): raise ValueError('invalid plane1')
   if not is_valid_rays(plane2): raise ValueError('invalid plane2')
   shape1, shape2 = np.array(plane1.shape), np.array(plane2.shape)
   if np.any((shape1 != shape2) * (shape1 != 1) * (shape2 != 1)):
      raise ValueError('incompatible shapes for plane1, plane2:')
   p1, n1 = plane1[..., :3, 0], plane1[..., :3, 1]
   p2, n2 = plane2[..., :3, 0], plane2[..., :3, 1]
   shape = tuple(np.maximum(plane1.shape, plane2.shape))
   u = np.cross(n1, n2)
   abs_u = np.abs(u)
   planes_parallel = np.sum(abs_u, axis=-1) < 0.000001
   p2_in_plane1 = point_in_plane(plane1, p2)
   status = np.zeros(shape[:-2])
   status[planes_parallel] = 1
   status[planes_parallel * p2_in_plane1] = 2
   d1 = -hdot(n1, p1)
   d2 = -hdot(n2, p2)
   amax = np.argmax(abs_u, axis=-1)
   sel0, sel1, sel2 = amax == 0, amax == 1, amax == 2
   n1a, n2a, d1a, d2a, ua = (x[sel0] for x in (n1, n2, d1, d2, u))
   n1b, n2b, d1b, d2b, ub = (x[sel1] for x in (n1, n2, d1, d2, u))
   n1c, n2c, d1c, d2c, uc = (x[sel2] for x in (n1, n2, d1, d2, u))

   ay = (d2a * n1a[..., 2] - d1a * n2a[..., 2]) / ua[..., 0]
   az = (d1a * n2a[..., 1] - d2a * n1a[..., 1]) / ua[..., 0]
   bz = (d2b * n1b[..., 0] - d1b * n2b[..., 0]) / ub[..., 1]
   bx = (d1b * n2b[..., 2] - d2b * n1b[..., 2]) / ub[..., 1]
   cx = (d2c * n1c[..., 1] - d1c * n2c[..., 1]) / uc[..., 2]
   cy = (d1c * n2c[..., 0] - d2c * n1c[..., 0]) / uc[..., 2]
   isect_pt = np.empty(shape[:-2] + (3, ), dtype=plane1.dtype)
   isect_pt[sel0, 0] = 0
   isect_pt[sel0, 1] = ay
   isect_pt[sel0, 2] = az
   isect_pt[sel1, 0] = bx
   isect_pt[sel1, 1] = 0
   isect_pt[sel1, 2] = bz
   isect_pt[sel2, 0] = cx
   isect_pt[sel2, 1] = cy
   isect_pt[sel2, 2] = 0
   isect = hray(isect_pt, u)
   return isect, status

def axis_ang_cen_of_eig(xforms, debug=False):
   # raise NotImplementedError('this is a bad way to get rotation axis')
   axis, angle = axis_angle_of(xforms)
   # seems to numerically unstable??
   ev, cen = np.linalg.eig(xforms)
   cen = np.real(cen[..., 3])
   cen = cen / cen[..., 3][..., None]  # normalize homogeneous coord
   cen = cen - axis * np.sum(axis * cen)
   return axis, angle, cen

def axis_ang_cen_of_planes(xforms, debug=False, ident_match_tol=1e-8):
   origshape = xforms.shape[:-2]
   xforms = xforms.reshape(-1, 4, 4)

   axis, angle = axis_angle_of(xforms)
   not_ident = np.abs(angle) > ident_match_tol
   cen = np.tile([0, 0, 0, 1], np.shape(angle)).reshape(*np.shape(angle), 4)

   if np.any(not_ident):
      xforms1 = xforms[not_ident]
      axis1 = axis[not_ident]
      #  sketchy magic points...
      p1 = (-32.09501046777237, 03.36227004372687, 35.34672781477340, 1)
      p2 = (21.15113978202345, 12.55664537217840, -37.48294301885574, 1)
      # p1 = rand_point()
      # p2 = rand_point()
      tparallel = hdot(axis, xforms[..., :, 3])[..., None] * axis
      q1 = xforms @ p1 - tparallel
      q2 = xforms @ p2 - tparallel
      n1 = hnormalized(q1 - p1)
      n2 = hnormalized(q2 - p2)
      c1 = (p1 + q1) / 2.0
      c2 = (p2 + q2) / 2.0
      plane1 = hray(c1, n1)
      plane2 = hray(c2, n2)
      isect, status = intersect_planes(plane1, plane2)
      cen1 = isect[..., :, 0]

      if len(cen) == len(cen1):
         cen = cen1
      else:
         cen[not_ident] = cen1

   axis = axis.reshape(*origshape, 4)
   angle = angle.reshape(origshape)
   cen = cen.reshape(*origshape, 4)
   return axis, angle, cen

axis_ang_cen_of = axis_ang_cen_of_planes

def line_line_distance_pa(pt1, ax1, pt2, ax2):
   # point1, point2 = hpoint(point1), hpoint(point2)
   # axis1, axis2 = hnormalized(axis1), hnormalized(axis2)
   n = abs(hdot(pt2 - pt1, hcross(ax1, ax2)))
   d = hnorm(hcross(ax1, ax2))
   r = np.zeros_like(n)
   i = abs(d) > 0.00001
   r[i] = n[i] / d[i]
   pp = hnorm(proj_perp(ax1, pt2 - pt1))
   return np.where(np.abs(hdot(ax1, ax2)) > 0.9999, pp, r)

def line_line_distance(ray1, ray2):
   pt1, pt2 = ray1[..., :, 0], ray2[..., :, 0]
   ax1, ax2 = ray1[..., :, 1], ray2[..., :, 1]
   return line_line_distance_pa(pt1, ax1, pt2, ax2)

def line_line_closest_points_pa(pt1, ax1, pt2, ax2, verbose=0):
   assert pt1.shape == pt2.shape == ax1.shape == ax2.shape
   # origshape = pt1.shape
   # print(pt1.shape)
   C21 = pt2 - pt1
   M = hcross(ax1, ax2)
   m2 = np.sum(M**2, axis=-1)[..., None]
   R = hcross(C21, M / m2)
   t1 = hdot(R, ax2)[..., None]
   t2 = hdot(R, ax1)[..., None]
   Q1 = pt1 - t1 * ax1
   Q2 = pt2 - t2 * ax2
   if verbose:
      print('C21', C21)
      print('M', M)
      print('m2', m2)
      print('R', R)
      print('t1', t1)
      print('t2', t2)
      print('Q1', Q1)
      print('Q2', Q2)
   return Q1, Q2

def line_line_closest_points(ray1, ray2, verbose=0):
   "currently errors if ax1==ax2"
   # pt1, pt2 = hpoint(pt1), hpoint(pt2)
   # ax1, ax2 = hnormalized(ax1), hnormalized(ax2)
   pt1, pt2 = ray1[..., :, 0], ray2[..., :, 0]
   ax1, ax2 = ray1[..., :, 1], ray2[..., :, 1]
   return line_line_closest_points_pa(pt1, ax1, pt2, ax2)

def dihedral(p1, p2, p3, p4):
   p1, p2, p3, p4 = hpoint(p1), hpoint(p2), hpoint(p3), hpoint(p4)
   a = hnormalized(p2 - p1)
   b = hnormalized(p3 - p2)
   c = hnormalized(p4 - p3)
   x = np.clip(hdot(a, b) * hdot(b, c) - hdot(a, c), -1, 1)
   y = np.clip(hdot(a, hcross(b, c)), -1, 1)
   return np.arctan2(y, x)

def align_around_axis(axis, u, v):
   return hrot(axis, -dihedral(u, axis, [0, 0, 0, 0], v))

def align_vector(a, b):
   return hrot((hnormalized(a) + hnormalized(b)) / 2, np.pi)

def align_vectors(a1, a2, b1, b2):
   "minimizes angular error"
   a1, a2, b1, b2 = (hnormalized(v) for v in (a1, a2, b1, b2))
   aaxis = (a1 + a2) / 2.0
   baxis = (b1 + b2) / 2.0
   Xmiddle = align_vector(aaxis, baxis)
   Xaround = align_around_axis(baxis, Xmiddle @ a1, b1)
   X = Xaround @ Xmiddle
   assert (angle(b1, a1) + angle(b2, a2)) + 0.001 >= (angle(b1, X @ a1) + angle(b2, X @ a2))
   return X

def calc_dihedral_angle(p1, p2, p3, p4):
   p1, p2, p3, p4 = hpoint(p1), hpoint(p2), hpoint(p3), hpoint(p4)
   p1, p2, p3, p4 = p1.reshape(4), p2.reshape(4), p3.reshape(4), p4.reshape(4)
   # Calculate coordinates for vectors q1, q2 and q3
   q1 = np.subtract(p2, p1)  # b - a
   q2 = np.subtract(p3, p2)  # c - b
   q3 = np.subtract(p4, p3)  # d - c
   q1_x_q2 = hcross(q1, q2)
   q2_x_q3 = hcross(q2, q3)
   n1 = hnormalized(q1_x_q2)
   n2 = hnormalized(q2_x_q3)
   u1 = n2
   u3 = hnormalized(q2)
   u2 = hcross(u3, u1)
   cos_theta = np.sum(n1 * u1)
   sin_theta = np.sum(n1 * u2)
   theta = -np.arctan2(sin_theta, cos_theta)
   return theta

def rotation_around_dof_for_target_angle(target_angle, dof_angle, fix_to_dof_angle):
   assert fix_to_dof_angle < np.pi / 2
   assert dof_angle <= np.pi / 2 + 0.00001
   assert target_angle <= np.pi

   if target_angle + dof_angle < fix_to_dof_angle: return np.array([-12345.0])
   if (dof_angle < 1e-6 or target_angle < 1e-6 or fix_to_dof_angle < 1e-6):
      return np.array([-12345.0])

   hdof = np.sin(dof_angle)
   l_dof = np.cos(dof_angle)
   h_tgt = np.sin(target_angle)
   l_tgt = np.cos(target_angle)
   # print('l_dof', l_dof)
   # print('l_tgt', l_tgt)
   xdof = np.sin(fix_to_dof_angle) * l_dof
   ydof = np.cos(fix_to_dof_angle) * l_dof
   assert np.allclose(np.sqrt(xdof**2 + ydof**2), l_dof)
   ytgt = np.cos(target_angle)
   slope = -np.tan(np.pi / 2 - fix_to_dof_angle)

   # print('ytgt', ytgt, 'xdof', xdof, 'ydof', ydof)

   yhat = ytgt
   xhat = xdof + (ytgt - ydof) * slope
   lhat = np.sqrt(xhat**2 + yhat**2)

   lhat = min(lhat, 1.0)

   # this caused occasional test failures
   # if lhat > 0.999999:
   #    if lhat > 1.000001:
   #       return np.array([-12345.0])
   #    else:
   #       return np.array([0.0])

   hhat = np.sqrt(1.0 - lhat**2)
   ahat = np.arcsin(hhat / hdof)

   # print('xhat', xhat, 'yhat', yhat, 'slope', slope, 'lhat', lhat, 'hhat', hhat, 'ahat', ahat)

   # print('ytgt', ytgt)
   # print('xdof', xdof)
   # print('ydof', ydof)
   # print('xhat', xhat)
   # print('yhat', yhat)
   # print('ahat', ahat, np.degrees(ahat))

   return ahat

def xform_around_dof_for_vector_target_angle(fix, mov, dof, target_angle):
   if hdot(dof, fix) < 0:
      dof = -dof
   if angle(dof, mov) > np.pi / 2:
      mov = -mov
   dang = calc_dihedral_angle(fix, [0.0, 0.0, 0.0, 0.0], dof, mov)
   assert angle(dof, mov) <= np.pi / 2 + 0.000001
   ahat = rotation_around_dof_for_target_angle(target_angle, angle(mov, dof), angle(fix, dof))
   if ahat == -12345.0:
      return []
   elif ahat == 0:
      mov1 = (hrot(dof, 0.000 - dang) @ mov[..., None]).reshape(1, 4)
      mov2 = (hrot(dof, np.pi - dang) @ mov[..., None]).reshape(1, 4)
      if np.allclose(angle(fix, mov1), target_angle):
         return [hrot(dof, np.pi - dang)]
         return
      elif np.allclose(angle(fix, mov1), target_angle):
         return [hrot(dof, np.pi - dang)]
      else:
         return []
   else:
      angles = [-dang + ahat, -dang - ahat, np.pi - dang + ahat, np.pi - dang - ahat]
      moves = [(hrot(dof, ang + 0.000) @ mov[..., None]).reshape(1, 4) for ang in angles]
      if not (np.allclose(angle(moves[0], fix), angle(moves[1], fix))
              or np.allclose(angle(moves[2], fix), angle(moves[3], fix))):
         return []

      if np.allclose(angle(moves[0], fix), target_angle):
         return [hrot(dof, angles[0]), hrot(dof, angles[1])]
      elif np.allclose(angle(moves[2], fix), target_angle):
         return [hrot(dof, angles[2]), hrot(dof, angles[3])]
      else:
         return []

def align_lines_isect_axis2(pt1, ax1, pt2, ax2, ta1, tp1, ta2, sl2, strict=True):
   '''zomg, point/axis reversed for second half of args...'''
   ## make sure to align with smaller axis choice
   assert np.allclose(np.linalg.norm(tp1[..., :3]), 0.0)
   if angle(ax1, ax2) > np.pi / 2: ax2 = -ax2
   if angle(ta1, ta2) > np.pi / 2: ta2 = -ta2
   if strict:
      assert np.allclose(angle(ta1, ta2), angle(ax1, ax2))
   if abs(angle(ta1, ta2)) < 0.01:
      assert 0, 'case not tested'
      # vector delta between pt2 and pt1
      d = proj_perp(ax1, pt2 - pt1)
      Xalign = align_vectors(ax1, d, ta1, sl2)  # align d to Y axis
      Xalign[..., :, 3] = -Xalign @ pt1
      slide_dist = (Xalign @ pt2)[..., 1]
   else:
      try:
         Xalign = align_vectors(ax1, ax2, ta1, ta2)
         # print(Xalign @ ax1, ta1)
         # assert np.allclose(Xalign @ ax1, ta1, atol=0.0001)
         # assert np.allclose(Xalign @ ax2, ta2, atol=0.0001)
         # print(Xalign)
      except AssertionError as e:
         print("align_vectors error")
         print("   ", ax1)
         print("   ", ax2)
         print("   ", ta1)
         print("   ", ta2)
         raise e
      Xalign[..., :, 3] = -Xalign @ pt1  ## move pt1 to origin
      Xalign[..., 3, 3] = 1
      cen2_0 = Xalign @ pt2  # moving pt2 by Xalign
      D = np.stack([ta1[:3], sl2[:3], ta2[:3]]).T
      A1offset, slide_dist, _ = np.linalg.inv(D) @ cen2_0[:3]
      # print(A1offset, slide_dist)
      Xalign[..., :, 3] = Xalign[..., :, 3] - (A1offset * ta1)

   return Xalign, slide_dist

def scale_translate_lines_isect_lines(pt1, ax1, pt2, ax2, tp1, ta1, tp2, ta2):
   _pt1 = hpoint(pt1.copy())
   _ax1 = hnormalized(ax1.copy())
   _pt2 = hpoint(pt2.copy())
   _ax2 = hnormalized(ax2.copy())
   _tp1 = hpoint(tp1.copy())
   _ta1 = hnormalized(ta1.copy())
   _tp2 = hpoint(tp2.copy())
   _ta2 = hnormalized(ta2.copy())

   if abs(angle(_ax1, _ax2) - angle(_ta1, _ta2)) > 0.00001:
      _ta2 = -_ta2
   # print(_ax1)
   # print(_ax2)
   # print(_ta1, ta1)
   # print(_ta2)
   # print(line_angle(_ax1, _ax2), line_angle(_ta1, _ta2))
   assert np.allclose(line_angle(_ax1, _ax2), line_angle(_ta1, _ta2))

   # scale target frame to match input line separation
   d1 = line_line_distance_pa(_pt1, _ax1, _pt2, _ax2)
   d2 = line_line_distance_pa(_tp1, _ta1, _tp2, _ta2)
   scale = np.array([d1 / d2, d1 / d2, d1 / d2, 1])
   _tp1 *= scale
   _tp2 *= scale

   # compute rotation to align line pairs, check "handedness" and correct if necessary
   xalign = align_vectors(_ax1, _ax2, _ta1, _ta2)
   a, b = line_line_closest_points_pa(_pt1, _ax1, _pt2, _ax2)
   c, d = line_line_closest_points_pa(_tp1, _ta1, _tp2, _ta2)
   _shift1 = xalign @ (b - a)
   _shift2 = d - c
   if hdot(_shift1, _shift2) < 0:
      if np.allclose(angle(_ax1, _ax2), np.pi / 2):
         xalign = align_vectors(-_ax1, _ax2, _ta1, _ta2)
      else:
         scale[:3] = -scale[:3]
         _tp1 *= -1
         _tp2 *= -1
         # rays = np.array([
         #    hm.hray(xalign @ pt1, xalign @ ax1),
         #    hm.hray(xalign @ pt2, xalign @ ax2),
         #    hm.hray(scale * tp1, scale * ta1),
         #    hm.hray(scale * tp2, scale * ta2),
         # ])
         # colors = [(1, 0, 0), (0, 0, 1), (0.8, 0.5, 0.5), (0.5, 0.5, 0.8)]
         # rp.viz.showme(rays, colors=colors, block=False)

   _pt1 = xalign @ _pt1
   _ax1 = xalign @ _ax1
   _pt2 = xalign @ _pt2
   _ax2 = xalign @ _ax2

   assert np.allclose(_ax1, _ta1, atol=1e-3) or np.allclose(-_ax1, _ta1, atol=1e-3)
   assert np.allclose(_ax2, _ta2, atol=1e-3) or np.allclose(-_ax2, _ta2, atol=1e-3)

   # move to overlap pa1,_ta1, aligning first axes
   delta1 = _tp1 - _pt1
   _pt1 += delta1
   _pt2 += delta1

   # delta align second axes by moving alone first
   pp = proj_perp(_ta2, _tp2 - _pt2)
   d = np.linalg.norm(pp)
   if d < 0.00001:
      delta2 = 0
   else:
      a = line_angle(_ta1, _ta2)
      l = d / np.sin(a)
      delta2 = l * hnormalized(proj(_ta1, _tp2 - _pt2))
      if hdot(pp, delta2) < 0:
         delta2 *= -1
   _pt1 += delta2
   _pt2 += delta2
   xalign[:, 3] = delta1 + delta2
   xalign[3, 3] = 1

   if np.any(np.isnan(xalign)):
      print('=============================')
      print(xalign)
      print(delta1, delta2)

   # rays = np.array([
   #    hm.hray(xalign @ pt1, xalign @ ax1),
   #    hm.hray(xalign @ pt2, xalign @ ax2),
   #    hm.hray(scale * tp1, scale * ta1),
   #    hm.hray(scale * tp2, scale * ta2),
   # ])
   # colors = [(1, 0, 0), (0, 0, 1), (0.8, 0.5, 0.5), (0.5, 0.5, 0.8)]
   # rp.viz.showme(rays, colors=colors, block=False)
   # assert 0

   return xalign, scale

def hcoherence(xforms, lever):
   xforms = xforms.copy()
   xforms[:, :3, :3] *= lever
   dist = xforms.reshape(1, -1, 4, 4) - xforms.reshape(-1, 1, 4, 4)
   dist = np.sqrt(np.sum(dist**2) / 4 / len(xforms))
   return dist

def hconstruct(rot, trans):
   x = np.zeros((rot.shape[:-2] + (4, 4)))
   x[..., :3, :3] = rot[..., :3, :3]
   x[..., :3, 3] = trans[..., :3]
   x[..., 3, 3] = 1
   return x

def hmean(xforms):
   q = rot_to_quat(xforms)
   idx = np.dot(q, [13, 7, 3, 1]) < 0
   if np.any(idx):
      q[idx] *= -1
   # print('hmean')
   # print(q)
   # for i in range(10):
   # qmean = np.mean(q, axis=0)
   # dot = hdot(qmean, q)
   # if np.all(dot >= 0):
   # q[dot < 0] *= -1
   # else:
   # assert 0, 'hmean cant find coherent quat mean'

   q = np.mean(q, axis=0)
   q = q / np.linalg.norm(q)
   r = quat_to_rot(q)
   t = np.mean(xforms[..., :3, 3], axis=0)
   x = hconstruct(r, t)
   # print(x)
   return x
