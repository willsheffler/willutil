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
    ang_err = pair.rad * (angle - pair.ang)
    # print('cyclic_sym_err', ang_err, angle, pair.ang, pair.rad)
    return np.sqrt(hel_err**2 + ang_err**2)

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

def symfit(frames, point_ang, symangles):
    pairs = symops_from_frames()
