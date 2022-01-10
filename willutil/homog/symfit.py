import numpy as np
from willutil import homog as hm, Bunch

def rel_xform_info(frame1, frame2):
    # rel = np.linalg.inv(frame1) @ frame2
    rel = frame2 @ np.linalg.inv(frame1)
    rot = rel[:3, :3]
    # axs, ang = hm.axis_angle_of(rel)
    axs, ang, cen = hm.axis_ang_cen_of(rel)
    inplane = hm.proj_perp(axs, cen - frame1[:, 3])
    # inplane2 = hm.proj_perp(axs, cen - frame2[:, 3])
    rad = np.sqrt(np.sum(inplane**2))
    if np.isnan(rad):
        print(rel)
        print(np.linalg.det(rel))
        print('arosti', axs, ang)
        assert 0
    hel = np.sum(axs * rel[:, 3])
    return Bunch(
        xrel=rel,
        axs=axs,
        ang=ang,
        cen=cen,
        rad=rad,
        hel=hel,
    )

def cyclic_sym_err(pair, angle):
    hel_err = pair.hel
    ang_err = pair.rad * (angle - pair.ang)
    print(ang_err, angle, pair.ang, pair.rad)
    return np.sqrt(hel_err**2 + ang_err**2)

def symops_from_frames(frames, angles):
    assert len(frames) > 1
    assert frames.shape[-2:] == (4, 4)
    pairs = dict()
    for i, frame1 in enumerate(frames):
        for j in range(i):
            frame2 = frames[j]
            pair = rel_xform_info(frame1, frame2)
            pair.err = dict()
            for n, a in angles.items():
                pair.err[n] = cyclic_sym_err(pair, a)
            pairs[i, j] = pair
    return pairs

def symfit(frames, point_ang, symangles):
    pairs = symops_from_frames()
