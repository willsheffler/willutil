import deferred_import
import sys
import numpy as np

t = deferred_import.deferred_import("torch")
import willutil as wu
# from willutil.homog.hgeom import *
h = sys.modules[__name__]

def torch_min(func, iters=4, history_size=10, max_iter=4, line_search_fn="strong_wolfe", **kw):
    import functools

    lbfgs = t.optim.LBFGS(
        kw["indep"],
        history_size=history_size,
        max_iter=max_iter,
        line_search_fn=line_search_fn,
    )
    closure = functools.partial(func, lbfgs=lbfgs, **kw)
    for iter in range(iters):
        loss = lbfgs.step(closure)
    return loss

def construct(rot, trans=None):
    rot = t.as_tensor(rot)
    x = t.zeros((rot.shape[:-2] + (4, 4)))
    x[..., :3, :3] = rot[..., :3, :3]
    if trans is not None:
        x[..., :3, 3] = t.as_tensor(trans)[..., :3]
    x[..., 3, 3] = 1
    return x

def mean_along(vecs, along=None):
    vecs = vec(vecs)
    assert vecs.ndim == 2
    if not along:
        along = vecs[0]
    along = vec(along)
    sign = t.sign(dot(along, vecs))
    flipped = (vecs.T * sign).T
    tot = t.sum(flipped, axis=0)
    return normalized(tot)

def com_flat(points, closeto=None, closefrac=0.5):
    if closeto != None:
        dist = norm(points - closeto)
        close = t.argsort(dist)[:closefrac * len(dist)]
        points = points[close]
    return t.mean(points, axis=-2)

def com(points, **kw):
    points = point(points)
    oshape = points.shape
    points = points.reshape(-1, oshape[-2], 4)
    com = com_flat(points)
    com = com.reshape(*oshape[:-2], 4)
    return com

def rog_flat(points):
    com = com_flat(points).reshape(-1, 1, 4)
    delta = t.linalg.norm(points - com, dim=2)
    rg = t.sqrt(t.mean(delta**2, dim=1))
    return rg

def rog(points, aboutaxis=None):
    points = point(points)
    oshape = points.shape
    points = points.reshape(-1, *oshape[-2:])
    if aboutaxis != None:
        aboutaxis = vec(aboutaxis)
        points = projperp(aboutaxis, points)
    rog = rog_flat(points)
    rog = rog.reshape(oshape[:-2])
    return rog

def proj(u, v):
    u = vec(u)
    v = point(v)
    return dot(u, v)[..., None] / norm2(u)[..., None] * u

def projperp(u, v):
    u = vec(u)
    v = point(v)
    return v - proj(u, v)

def axis_angle_cen(xforms, ident_match_tol=1e-8):
    # ic(xforms.dtype)
    origshape = xforms.shape[:-2]
    xforms = xforms.reshape(-1, 4, 4)
    axis, angle = axis_angle(xforms)
    not_ident = t.abs(angle) > ident_match_tol
    cen = t.tile(
        t.tensor([0, 0, 0, 1]),
        angle.shape,
    ).reshape(*angle.shape, 4)

    assert t.all(not_ident)
    xforms1 = xforms[not_ident]
    axis1 = axis[not_ident]
    #  sketchy magic points...
    p1, p2 = axis_ang_cen_magic_points_torch()
    p1 = p1.to(xforms.dtype)
    p2 = p2.to(xforms.dtype)
    tparallel = dot(axis, xforms[..., :, 3])[..., None] * axis
    q1 = xforms @ p1 - tparallel
    q2 = xforms @ p2 - tparallel
    n1 = normalized(q1 - p1).reshape(-1, 4)
    n2 = normalized(q2 - p2).reshape(-1, 4)
    c1 = (p1 + q1) / 2.0
    c2 = (p2 + q2) / 2.0

    isect, norm, status = intersect_planes(c1, n1, c2, n2)
    cen1 = isect[..., :]
    if len(cen) == len(cen1):
        cen = cen1
    else:
        cen = t.where(not_ident, cen1, cen)

    axis = axis.reshape(*origshape, 4)
    angle = angle.reshape(origshape)
    cen = cen.reshape(*origshape, 4)
    return axis, angle, cen

def rot(axis, angle, center=None, hel=None, squeeze=True):
    if center is None:
        center = t.tensor([0, 0, 0, 1], dtype=t.float)
    angle = t.as_tensor(angle)
    axis = vec(axis)
    center = point(center)
    if hel is None:
        hel = t.tensor([0], dtype=t.float)
    if axis.ndim == 1:
        axis = axis[
            None,
        ]
    if angle.ndim == 0:
        angle = angle[
            None,
        ]
    if center.ndim == 1:
        center = center[
            None,
        ]
    if hel.ndim == 0:
        hel = hel[
            None,
        ]
    rot = rot3(axis, angle, shape=(4, 4), squeeze=False)
    shape = angle.shape

    # assert 0
    x, y, z = center[..., 0], center[..., 1], center[..., 2]
    center = t.stack(
        [
            x - rot[..., 0, 0] * x - rot[..., 0, 1] * y - rot[..., 0, 2] * z,
            y - rot[..., 1, 0] * x - rot[..., 1, 1] * y - rot[..., 1, 2] * z,
            z - rot[..., 2, 0] * x - rot[..., 2, 1] * y - rot[..., 2, 2] * z,
            t.ones(*shape),
        ],
        axis=-1,
    )
    shift = axis * hel[..., None]
    center = center + shift
    r = t.cat(
        [
            rot[..., :3],
            center[
                ...,
                None,
            ],
        ],
        axis=-1,
    )
    if r.shape == (1, 4, 4):
        r = r.reshape(4, 4)
    return r

def rand_point(*a, **kw):
    return t.from_numpy(wu.hrandpoint(*a, **kw))

def rand_vec(*a, **kw):
    return t.from_numpy(wu.hrandvec(*a, **kw))

def rand_xform_small(*a, **kw):
    return t.from_numpy(wu.hrandsmall(*a, **kw))

def rand_xform(*a, **kw):
    return t.from_numpy(wu.hrand(*a, **kw))

def rand_quat(*a, **kw):
    return t.from_numpy(rand_quat(*a, **kw))

rand = rand_xform

def rot_to_quat(xform):
    raise NotImplementedError
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

    assert np.sum(case0) + np.sum(case1) + np.sum(case2) + np.sum(case3) == np.prod(xform.shape[:-2])

    return quat_to_upper_half(quat)

xform_to_quat = rot_to_quat

def is_valid_quat_rot(quat):
    assert quat.shape[-1] == 4
    return np.isclose(1, t.linalg.norm(quat, axis=-1))

def quat_to_upper_half(quat):
    ineg0 = quat[..., 0] < 0
    ineg1 = (quat[..., 0] == 0) * (quat[..., 1] < 0)
    ineg2 = (quat[..., 0] == 0) * (quat[..., 1] == 0) * (quat[..., 2] < 0)
    ineg3 = (quat[..., 0] == 0) * (quat[..., 1] == 0) * (quat[..., 2] == 0) * (quat[..., 3] < 0)
    # ic(ineg0.shape)
    # ic(ineg1.shape)
    # ic(ineg2.shape)
    # ic(ineg3.shape)
    ineg = ineg0 + ineg1 + ineg2 + ineg3
    quat2 = t.where(ineg, -quat, quat)
    return normalized(quat2)

def cart(h):
    return h[..., :, 3]

def cart3(h):
    return h[..., :3, 3]

def ori(h):
    h = h.clone()
    h[..., :3, 3] = 0
    return h

def ori3(h):
    h[..., :3, :3]

def homog(rot, trans=None, **kw):
    if trans is None:
        trans = t.as_tensor([0, 0, 0, 0], device=rot.device, **kw)
    trans = t.as_tensor(trans)

    if rot.shape == (3, 3):
        rot = t.cat([rot, t.tensor([[0.0, 0.0, 0.0]], device=rot.device)], axis=0)
        rot = t.cat([rot, t.tensor([[0], [0], [0], [1]], device=rot.device)], axis=1)

    assert rot.shape[-2:] == (4, 4)
    assert trans.shape[-1:] == (4, )

    h = t.cat([rot[:, :3], trans[:, None]], axis=1)
    return h

def quat_to_rot(quat):
    assert quat.shape[-1] == 4
    qr = quat[..., 0]
    qi = quat[..., 1]
    qj = quat[..., 2]
    qk = quat[..., 3]

    rot = t.cat([
        t.tensor([[
            1 - 2 * (qj**2 + qk**2),
            2 * (qi * qj - qk * qr),
            2 * (qi * qk + qj * qr),
        ]]),
        t.tensor([[
            2 * (qi * qj + qk * qr),
            1 - 2 * (qi**2 + qk**2),
            2 * (qj * qk - qi * qr),
        ]]),
        t.tensor([[
            2 * (qi * qk - qj * qr),
            2 * (qj * qk + qi * qr),
            1 - 2 * (qi**2 + qj**2),
        ]]),
    ])
    # ic(rot.shape)
    return rot

def quat_to_xform(quat, dtype="f8"):
    r = quat_to_rot(quat, dtype)
    r = t.cat([r])
    return r

def rot3(axis, angle, shape=(3, 3), squeeze=True):
    # axis = t.tensor(axis, dtype=dtype, requires_grad=requires_grad)
    # angle = angle * np.pi / 180.0 if degrees else angle
    # angle = t.tensor(angle, dtype=dtype, requires_grad=requires_grad)

    if axis.ndim == 1:
        axis = axis[
            None,
        ]
    if angle.ndim == 0:
        angle = angle[
            None,
        ]
    # if angle.ndim == 0
    if axis.shape and angle.shape and not is_broadcastable(axis.shape[:-1], angle.shape):
        raise ValueError(f"axis/angle not compatible: {axis.shape} {angle.shape}")
    zero = t.zeros(*angle.shape)
    axis = normalized(axis)
    a = t.cos(angle / 2.0)
    tmp = axis * -t.sin(angle / 2)[..., None]
    b, c, d = tmp[..., 0], tmp[..., 1], tmp[..., 2]
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    if shape == (3, 3):
        rot = t.stack(
            [
                t.stack([aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)], axis=-1),
                t.stack([2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)], axis=-1),
                t.stack([2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc], axis=-1),
            ],
            axis=-2,
        )
    elif shape == (4, 4):
        rot = t.stack(
            [
                t.stack([aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac), zero], axis=-1),
                t.stack([2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab), zero], axis=-1),
                t.stack([2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc, zero], axis=-1),
                t.stack([zero, zero, zero, zero + 1], axis=-1),
            ],
            axis=-2,
        )
    else:
        raise ValueError(f"rot3 shape must be (3,3) or (4,4), not {shape}")
    # ic('foo')
    # ic(axis.shape)
    # ic(angle.shape)
    # ic(rot.shape)
    if squeeze and rot.shape == (1, 3, 3):
        rot = rot.reshape(3, 3)
    if squeeze and rot.shape == (1, 4, 4):
        rot = rot.reshape(4, 4)
    return rot

def rms(a, b):
    assert a.shape == b.shape
    return t.sqrt(t.sum(t.square(a - b)) / len(a))

def xform(xform, stuff, homogout="auto", **kw):
    xform = t.as_tensor(xform).to(stuff.dtype)
    nothomog = stuff.shape[-1] == 3
    if stuff.shape[-1] == 3:
        stuff = point(stuff)
    result = _thxform_impl(xform, stuff, **kw)
    if homogout is False or homogout == "auto" and nothomog:
        result = result[..., :3]
    return result

_xform = xform

def xformpts(xform, stuff, **kw):
    return _xform(xform, stuff, is_points=True, **kw)

    # if result.shape[-1] == 4 and not wu.hvalid(result.cpu().detach().numpy(), **kw):
    #   # ic(result[:10])
    #   # is is a bad copout.. should make is check handle nans correctly
    #   if not stuff.shape[-2:] == (4, 1):
    #      raise ValueError(
    #         f'malformed homogeneous coords with shape {stuff.shape}, if points and shape is (...,4,4) try is_points=True'
    #      )

    return result

def rmsfit(mobile, target):
    """use kabsch method to get rmsd fit"""
    assert mobile.shape == target.shape
    assert mobile.ndim > 1
    assert mobile.shape[-1] in (3, 4)
    if len(mobile) < 3:
        raise ValueError("need at least 3 points to fit")
    if mobile.dtype != target.dtype:
        mobile = mobile.to(target.dtype)
    mobile = point(mobile)
    target = point(target)
    mobile_cen = t.mean(mobile, axis=0)
    target_cen = t.mean(target, axis=0)
    mobile = mobile - mobile_cen
    target = target - target_cen
    # ic(mobile.shape)
    # ic(target.shape[-1] in (3, 4))
    covariance = mobile.T[:3] @ target[:, :3]
    V, S, W = t.linalg.svd(covariance)
    if 0 > t.det(V) * t.det(W):
        S = t.tensor([S[0], S[1], -S[2]], dtype=S.dtype, device=S.device)
        # S[-1] = -S[-1]
        # ic(S - S1)
        V = t.cat([V[:, :-1], -V[:, -1, None]], dim=1)
        # V[:, -1] = -V[:, -1]
        # ic(V - V1)
        # assert 0
    rot_m2t = homog(V @ W).T
    trans_m2t = target_cen - rot_m2t @ mobile_cen
    xform_mobile_to_target = homog(rot_m2t, trans_m2t)

    mobile = mobile + mobile_cen
    target = target + target_cen
    mobile_fit_to_target = xform(xform_mobile_to_target, mobile)
    rms_ = rms(target, mobile_fit_to_target)

    return rms_, mobile_fit_to_target, xform_mobile_to_target

def randpoint(shape=(), cen=[0, 0, 0], std=1, dtype=None):
    dtype = dtype or t.float32
    cen = t.as_tensor(cen)
    if isinstance(shape, int):
        shape = (shape, )
    p = point(t.randn(*(shape) + (3, ), dtype=dtype) * std + cen)
    return p

def randvec(shape=(), std=1, dtype=None):
    dtype = dtype or t.float32
    if isinstance(shape, int):
        shape = (shape, )
    return vec(t.randn(*(shape + (3, ))) * std)

def randunit(shape=(), cen=[0, 0, 0], std=1):
    dtype = dtype or t.float32
    if isinstance(shape, int):
        shape = (shape, )
    v = normalized(t.randn(*(shape + (3, ))) * std)
    return v

def point(point, **kw):
    point = t.as_tensor(point)
    shape = point.shape[:-1]
    points = t.cat([point[..., :3], t.ones(shape + (1, ), device=point.device)], axis=-1)
    if points.dtype not in (t.float32, t.float64):
        points = points.to(t.float32)
    return points

def vec(vec):
    vec = t.as_tensor(vec)
    if vec.dtype not in (t.float32, t.float64):
        vec = vec.to(t.float32)
    if vec.shape[-1] == 4:
        if t.any(vec[..., 3] != 0):
            vec = t.cat([vec[..., :3], t.zeros(*vec.shape[:-1], 1, device=vec.device)], dim=-1)
        return vec
    elif vec.shape[-1] == 3:
        r = t.zeros(vec.shape[:-1] + (4, ), dtype=vec.dtype, device=vec.device)
        r[..., :3] = vec
        return r
    else:
        raise ValueError("vec must len 3 or 4")

def normalized(a):
    return t.nn.functional.normalize(t.as_tensor(a, dtype=float), dim=-1)
    # a = t.as_tensor(a)
    # if (not a.shape and len(a) == 3) or (a.shape and a.shape[-1] == 3):
    #    a, tmp = t.zeros(a.shape[:-1] + (4, ), dtype=a.type), a
    #    a[..., :3] = tmp
    # a2 = a[:]
    # a2[..., 3] = 0
    # return a2 / norm(a2)[..., None]

def norm(a):
    a = t.as_tensor(a)
    return t.sqrt(t.sum(a[..., :3] * a[..., :3], axis=-1))

def norm2(a):
    a = t.as_tensor(a)
    return t.sum(a[..., :3] * a[..., :3], axis=-1)

def axis_angle_hel(xforms):
    axis, angle = axis_angle(xforms)
    hel = dot(axis, xforms[..., :, 3])
    return axis, angle, hel

def axis_angle_cen_hel(xforms):
    axis, angle, cen = axis_angle_cen(xforms)
    hel = dot(axis, xforms[..., :, 3])
    return axis, angle, cen, hel

def axis_angle(xforms):
    axis_ = axis(xforms)
    angl = angle(xforms)
    return axis_, angl

def axis(xforms):
    if xforms.shape[-2:] == (4, 4):
        return normalized(
            t.stack(
                (
                    xforms[..., 2, 1] - xforms[..., 1, 2],
                    xforms[..., 0, 2] - xforms[..., 2, 0],
                    xforms[..., 1, 0] - xforms[..., 0, 1],
                    t.zeros(xforms.shape[:-2]),
                ),
                axis=-1,
            ))
    if xforms.shape[-2:] == (3, 3):
        return normalized(
            t.stack(
                (
                    xforms[..., 2, 1] - xforms[..., 1, 2],
                    xforms[..., 0, 2] - xforms[..., 2, 0],
                    xforms[..., 1, 0] - xforms[..., 0, 1],
                ),
                axis=-1,
            ))
    else:
        raise ValueError("wrong shape for xform/rotation matrix: " + str(xforms.shape))

def angle(xforms):
    tr = xforms[..., 0, 0] + xforms[..., 1, 1] + xforms[..., 2, 2]
    cos = (tr - 1.0) / 2.0
    angl = t.arccos(t.clip(cos, -1, 1))
    return angl

def point_line_dist2(point, cen, norm):
    point, cen, norm = h.point(point), h.point(cen), h.normalized(norm)
    point = point - cen
    perp = h.projperp(norm, point)
    return h.norm2(perp)

def dot(a, b, outerprod=False):
    if outerprod:
        shape1 = a.shape[:-1]
        shape2 = b.shape[:-1]
        a = a.reshape((1, ) * len(shape2) + shape1 + (-1, ))
        b = b.reshape(shape2 + (1, ) * len(shape1) + (-1, ))
    return t.sum(a[..., :3] * b[..., :3], axis=-1)

def point_in_plane(point, normal, pt):
    inplane = t.abs(dot(normal[..., :3], pt[..., :3] - point[..., :3]))
    return inplane < 0.00001

def ray_in_plane(point, normal, p1, n1):
    inplane1 = point_in_plane(point, normal, p1)
    inplane2 = point_in_plane(point, normal, p1 + n1)
    return inplane1 and inplane2

def intersect_planes(p1, n1, p2, n2):
    """
    intersect_Planes: find e 3D intersection of two planes
       Input:  two planes represented (point, normal) as (p1,n1), (p2,n2)
       Output: L = e intersection line (when it exists)
       Return: rays shape=(...,4,2), status
               0 = intersection returned
               1 = disjoint (no intersection)
               2 = e two planes coincide
    """
    """intersect two planes
   :param plane1: first plane represented by ray
   :type plane2: np.array shape=(..., 4, 2)
   :param plane1: second planes represented by rays
   :type plane2: np.array shape=(..., 4, 2)
   :return: line: np.array shape=(...,4,2), status: int (0 = intersection returned, 1 = no intersection, 2 = e two planes coincide)
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

    u = t.linalg.cross(n1[..., :3], n2[..., :3])
    abs_u = t.abs(u)
    planes_parallel = t.sum(abs_u, axis=-1) < 0.000001
    p2_in_plane1 = point_in_plane(p1, n1, p2)
    status = t.zeros(N)
    status[planes_parallel] = 1
    status[planes_parallel * p2_in_plane1] = 2
    d1 = -dot(n1, p1)
    d2 = -dot(n2, p2)

    amax = t.argmax(abs_u, axis=-1)
    sel = amax == 0, amax == 1, amax == 2
    perm = t.cat([
        t.where(sel[0])[0],
        t.where(sel[1])[0],
        t.where(sel[2])[0],
    ])
    perminv = t.empty_like(perm)
    perminv[perm] = t.arange(len(perm))
    breaks = np.cumsum([0, sum(sel[0]), sum(sel[1]), sum(sel[2])])
    n1 = n1[perm]
    n2 = n2[perm]
    d1 = d1[perm]
    d2 = d2[perm]
    up = u[perm]

    zeros = t.zeros(N)
    ones = t.ones(N)
    l = []

    s = slice(breaks[0], breaks[1])
    y = (d2[s] * n1[s, 2] - d1[s] * n2[s, 2]) / up[s, 0]
    z = (d1[s] * n2[s, 1] - d2[s] * n1[s, 1]) / up[s, 0]
    l.append(t.stack([zeros[s], y, z, ones[s]], axis=-1))

    s = slice(breaks[1], breaks[2])
    z = (d2[s] * n1[s, 0] - d1[s] * n2[s, 0]) / up[s, 1]
    x = (d1[s] * n2[s, 2] - d2[s] * n1[s, 2]) / up[s, 1]
    l.append(t.stack([x, zeros[s], z, ones[s]], axis=-1))

    s = slice(breaks[2], breaks[3])
    x = (d2[s] * n1[s, 1] - d1[s] * n2[s, 1]) / up[s, 2]
    y = (d1[s] * n2[s, 0] - d2[s] * n1[s, 0]) / up[s, 2]
    l.append(t.stack([x, y, zeros[s], ones[s]], axis=-1))

    isect_pt = t.cat(l)
    isect_pt = isect_pt[perminv]
    isect_pt = isect_pt.reshape(origshape)

    isect_dirn = normalized(t.cat([u, t.zeros(N, 1)], axis=-1))
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
    return t.from_numpy(wu.homog.hgeom._axis_ang_cen_magic_points_numpy).float()

def diff(x, y, lever=10.0):
    shape1 = x.shape[:-2]
    shape2 = y.shape[:-2]
    a = x.reshape(shape1 + (1, ) * len(shape1) + (4, 4))
    b = y.reshape((1, ) * len(shape2) + shape2 + (4, 4))

    axyz = a[..., :3, :3] * lever + a[..., :3, 3, None]
    bxyz = b[..., :3, :3] * lever + b[..., :3, 3, None]

    diff = t.norm(axyz - bxyz, dim=-1)
    diff = t.mean(diff, dim=-1)

    return diff

# def cross(u, v):
#    return t.linalg.cross(u[..., :3], v[..., :3])

# def frame(u, v, w, cen=None):
#    assert u.shape == v.shape == w.shape
#    if not cen: cen = u
#    assert cen.shape == u.shape
#    stubs = t.empty(u.shape[:-1] + (4, 4), device=u.device)
#    stubs[..., :, 0] = normalized(u - v)
#    stubs[..., :, 2] = normalized(cross(stubs[..., :, 0], w - v))
#    stubs[..., :, 1] = cross(stubs[..., :, 2], stubs[..., :, 0])
#    stubs[..., :, 3] = cen[..., :]
#    return stubs

def Qs2Rs(Qs):
    Rs = t.zeros((*Qs.shape[:-1], 3, 3), device=Qs.device)

    Rs[..., 0, 0] = (Qs[..., 0] * Qs[..., 0] + Qs[..., 1] * Qs[..., 1] - Qs[..., 2] * Qs[..., 2] -
                     Qs[..., 3] * Qs[..., 3])
    Rs[..., 0, 1] = 2 * Qs[..., 1] * Qs[..., 2] - 2 * Qs[..., 0] * Qs[..., 3]
    Rs[..., 0, 2] = 2 * Qs[..., 1] * Qs[..., 3] + 2 * Qs[..., 0] * Qs[..., 2]
    Rs[..., 1, 0] = 2 * Qs[..., 1] * Qs[..., 2] + 2 * Qs[..., 0] * Qs[..., 3]
    Rs[..., 1, 1] = (Qs[..., 0] * Qs[..., 0] - Qs[..., 1] * Qs[..., 1] + Qs[..., 2] * Qs[..., 2] -
                     Qs[..., 3] * Qs[..., 3])
    Rs[..., 1, 2] = 2 * Qs[..., 2] * Qs[..., 3] - 2 * Qs[..., 0] * Qs[..., 1]
    Rs[..., 2, 0] = 2 * Qs[..., 1] * Qs[..., 3] - 2 * Qs[..., 0] * Qs[..., 2]
    Rs[..., 2, 1] = 2 * Qs[..., 2] * Qs[..., 3] + 2 * Qs[..., 0] * Qs[..., 1]
    Rs[..., 2, 2] = (Qs[..., 0] * Qs[..., 0] - Qs[..., 1] * Qs[..., 1] - Qs[..., 2] * Qs[..., 2] +
                     Qs[..., 3] * Qs[..., 3])

    return Rs

# ============================================================
def normQ(Q):
    """normalize a quaternions"""
    return Q / t.linalg.norm(Q, keepdim=True, dim=-1)

def Q2R(Q):
    Qs = t.cat((t.ones((len(Q), 1), device=Q.device, dtype=Q.dtype), Q), dim=-1)
    Qs = normQ(Qs)
    return Qs2Rs(Qs[None, :]).squeeze(0)

def _thxform_impl(x, stuff, outerprod="auto", flat=False, is_points="auto", improper_ok=False):
    if is_points == "auto":
        is_points = not valid44(stuff, improper_ok=improper_ok)
        if is_points:
            if stuff.shape[-1] != 4 and stuff.shape[-2:] == (4, 1):
                raise ValueError(f"hxform cant understand shape {stuff.shape}")

    if not is_points:
        if outerprod == "auto":
            outerprod = x.shape[:-2] != stuff.shape[:-2]
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
        if flat:
            result = result.reshape(-1, 4, 4)
    else:
        if outerprod == "auto":
            outerprod = x.shape[:-2] != stuff.shape[:-1]

        if stuff.shape[-1] != 1:
            stuff = stuff[..., None]
        if outerprod:
            shape1 = x.shape[:-2]
            shape2 = stuff.shape[:-2]
            # ic(x.shape, stuff.shape, shape1, shape2)
            a = x.reshape(shape1 + (1, ) * len(shape2) + (4, 4))

            b = stuff.reshape((1, ) * len(shape1) + shape2 + (4, 1))
            result = a @ b
        else:
            # try to match first N dimensions, outer prod the rest
            shape1 = x.shape[:-2]
            shape2 = stuff.shape[:-2]
            sameshape = tuple()
            for i, (s1, s2) in enumerate(zip(shape1, shape2)):
                # ic(s1, s2)
                if s1 == s2:
                    shape1 = shape1[1:]
                    shape2 = shape2[1:]
                    sameshape = sameshape + (s1, )
                else:
                    break
            newshape1 = sameshape + shape1 + (1, ) * len(shape2) + (4, 4)
            newshape2 = sameshape + (1, ) * len(shape1) + shape2 + (4, 1)
            # ic(shape1, shape2, newshape1, newshape2)
            a = x.reshape(newshape1)
            b = stuff.reshape(newshape2)
            result = a @ b

        result = result.squeeze(axis=-1)

        if flat:
            result = result.reshape(-1, 4)

        # assert 0
    # ic('result', result.shape)
    return result

def valid(stuff, is_points=None, strict=False, **kw):
    if stuff.shape[-2:] == (4, 4) and not is_points == True:
        return valid44(stuff, **kw)
    if stuff.shape[-2:] == (4, 2) and not is_points == True:
        return is_valid_rays(stuff)
    elif stuff.shape[-1] == 4 and strict:
        return t.allclose(stuff[..., 3], 0) or t.allclose(stuff[..., 3], 1)
    elif stuff.shape[-1] == 4:
        return t.all(t.logical_or(t.isclose(stuff[..., 3], 0), t.isclose(stuff[..., 3], 1)))
    elif stuff.shape[-1] == 3:
        return True
    return False

def valid_norm(x):
    normok = t.allclose(1, t.linalg.norm(x[..., :3, :3], axis=-1))
    normok &= t.allclose(1, t.linalg.norm(x[..., :3, :3], axis=-2))
    return t.all(normok)

def valid44(x, improper_ok=False, **kw):
    if x.shape[-2:] != (4, 4):
        return False
    det = t.linalg.det(x[..., :3, :3])
    if improper_ok:
        det = t.abs(det)
    detok = t.allclose(det, t.tensor(1.0))

    return all([t.allclose(x[..., 3, 3], t.tensor(1.0)), t.allclose(x[..., 3, :3], t.tensor(0.0)), detok])
