import itertools as it
import numpy as np
from willutil import homog as hm, Bunch

class RelXformInfo(Bunch):
    pass

def rel_xform_info(frame1, frame2):
    # rel = np.linalg.inv(frame1) @ frame2
    rel = frame2 @ np.linalg.inv(frame1)
    rot = rel[:3, :3]
    # axs, ang = hm.axis_angle_of(rel)
    axs, ang, cen = hm.axis_ang_cen_of(rel)

    framecen = (frame2[:, 3] + frame1[:, 3]) / 2
    framecen = framecen - cen
    framecen = hm.proj(axs, framecen)
    framecen = framecen + cen

    inplane = hm.proj_perp(axs, cen - frame1[:, 3])
    # inplane2 = hm.proj_perp(axs, cen - frame2[:, 3])
    rad = np.sqrt(np.sum(inplane**2))
    if np.isnan(rad):
        print('isnan rad')
        print('xrel')
        print(rel)
        print('det', np.linalg.det(rel))
        print('axs ang', axs, ang)
        print('cen', cen)
        print('inplane', inplane)
        assert 0
    hel = np.sum(axs * rel[:, 3])
    return RelXformInfo(
        xrel=rel,
        axs=axs,
        ang=ang,
        cen=cen,
        rad=rad,
        hel=hel,
        framecen=framecen,
        frames=np.array([frame1, frame2]),
    )

def cyclic_sym_err(pair, angle):
    hel_err = pair.hel
    errrad = min(10000, max(pair.rad, 1.0))
    ang_err = errrad * (angle - pair.ang)
    err = np.sqrt(hel_err**2 + ang_err**2)
    return err

def symops_from_frames(frames, point_angles):
    assert len(frames) > 1
    assert frames.shape[-2:] == (4, 4)
    pairs = dict()
    for i, frame1 in enumerate(frames):
        for j in range(i + 1, len(frames)):
            frame2 = frames[j]
            pair = rel_xform_info(frame1, frame2)
            pair.err = dict()
            for n, angs in point_angles.items():
                err = min(cyclic_sym_err(pair, a) for a in angs)
                pair.err[n] = err
                # print('symops_from_frames', n, a, err)
            pairs[i, j] = pair
    return pairs

class SymOpsImfo(Bunch):
    pass

def symops_info(
    symops,
    max_nan=0.0,
    remove_outliers_sd=3,
    **kw,
):
    cen1, cen2, axis1, axis2 = list(), list(), list(), list()
    for op1, op2 in it.combinations(symops.values(), 2):
        if op1 is op2: continue
        cen1.append(op1.cen)
        cen2.append(op2.cen)
        axis1.append(op1.axs)
        axis2.append(op2.axs)
    cen1 = np.stack(cen1)
    cen2 = np.stack(cen2)
    axis1 = np.stack(axis1)
    axis2 = np.stack(axis2)

    not_parallel = np.abs(np.sum(axis1 * axis2, axis=-1)) < 0.99
    p1np = cen1[not_parallel]
    p2np = cen2[not_parallel]
    a1np = axis1[not_parallel]
    a2np = axis2[not_parallel]

    # print('cen1', cen1.shape, 'isnan', np.sum(np.isnan(cen1)))
    # print('cen2', cen2.shape, 'isnan', np.sum(np.isnan(cen2)))
    # print('axis1', axis1.shape, 'isnan', np.sum(np.isnan(axis1)))
    # print('axis2', axis2.shape, 'isnan', np.sum(np.isnan(axis2)))

    p, q = hm.line_line_closest_points_pa(p1np, a1np, p2np, a2np)
    # print('p', p.shape, 'isnan', np.sum(np.isnan(p)))
    # print('q', q.shape, 'isnan', np.sum(np.isnan(q)))
    tot_nan = np.sum(np.isnan(p))
    assert tot_nan / len(p) <= max_nan  # some are parallel

    p = p[~np.isnan(p)].reshape(-1, 4)
    q = q[~np.isnan(q)].reshape(-1, 4)
    isect = (p + q) / 2

    # print(hm.angle_degrees(axis1, axis2)[:10])
    # print('p', p[:10])
    # print('q', q[:10])
    # print('isect', isect[:10])
    # print(p.shape, q.shape)

    cen = np.mean(isect, axis=0)
    # print(cen.shape, cen)

    if remove_outliers_sd is not None:
        norm = np.linalg.norm(p - cen, axis=-1)
        meannorm = np.mean(norm)
        sdnorm = np.std(norm)
        not_outlier = norm - meannorm < sdnorm * remove_outliers_sd
        # print('norm', norm.shape, np.mean(norm), np.min(norm), np.max(norm), np.sum(not_outlier),
        # np.sum(not_outlier) / len(not_outlier))
        # print(cen)
        cen = np.mean(isect[not_outlier], axis=0)
        # print(cen)

    return SymOpsImfo(
        symops=symops,
        mean_center=cen,
        cen1=cen1,
        cen2=cen2,
        axis1=axis1,
        axis2=axis2,
        iscet=isect,
        isect1=p,
        iscet2=q,
    )

def align_axes(symops_info):
    pass

# def symfit(frames, point_ang, symangles):
# pairs = symops_from_frames()
