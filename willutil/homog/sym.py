from willutil.homog.hgeom import *
from willutil.homog.symframes import *

m = -1

ambuguous_axes = dict(tet=[], oct=[(2, 4)], icos=[])

tetrahedral_axes = {
    2: hnormalized([1, 0, 0]),
    3: hnormalized([1, 1, 1]),
    7: hnormalized([1, 1, m])  # other c3
}
octahedral_axes = {
    2: hnormalized([1, 1, 0]),
    3: hnormalized([1, 1, 1]),
    4: hnormalized([1, 0, 0])
}
icosahedral_axes = {
    2: hnormalized([1, 0, 0]),
    3: hnormalized([0.934172, 0.000000, 0.356822]),
    5: hnormalized([0.850651, 0.525731, 0.000000])
}

tetrahedral_axes_all = {
    2:
    hnormalized([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        # [m, 0, 0],
        # [0, m, 0],
        # [0, 0, m],
    ]),
    3:
    hnormalized([
        [1, 1, 1],
        [1, m, m],
        [m, m, 1],
        [m, 1, m],
        # [m, m, m],
        # [m, 1, 1],
        # [1, 1, m],
        # [1, m, 1],
    ]),
    7:
    hnormalized([
        [m, 1, 1],
        [1, m, 1],
        [1, 1, m],
        [m, m, -1],
    ]),
}
octahedral_axes_all = {
    2:
    hnormalized([
        [1, 1, 0],
        [0, 1, 1],
        [1, 0, 1],
        [m, 1, 0],
        [0, m, 1],
        [m, 0, 1],
        # [1, m, 0],
        # [0, 1, m],
        # [1, 0, m],
        # [m, m, 0],
        # [0, m, m],
        # [m, 0, m],
    ]),
    3:
    hnormalized([
        [1, 1, 1],
        [m, 1, 1],
        [1, m, 1],
        [1, 1, m],
        # [m, 1, m],
        # [m, m, 1],
        # [1, m, m],
        # [m, m, m],
    ]),
    4:
    hnormalized([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        # [m, 0, 0],
        # [0, m, 0],
        # [0, 0, m],
    ]),
}

def _icosahedral_axes_all():
    a2 = icosahedral_frames @ icosahedral_axes[2]
    a3 = icosahedral_frames @ icosahedral_axes[3]
    a5 = icosahedral_frames @ icosahedral_axes[5]

    # six decimals enough to account for numerical errors
    a2 = a2[np.unique(np.around(a2, decimals=6), axis=0, return_index=True)[1]]
    a3 = a3[np.unique(np.around(a3, decimals=6), axis=0, return_index=True)[1]]
    a5 = a5[np.unique(np.around(a5, decimals=6), axis=0, return_index=True)[1]]

    a2 = np.stack([a for i, a in enumerate(a2) if np.all(np.sum(a * a2[:i], axis=-1) > -0.999)])
    a3 = np.stack([a for i, a in enumerate(a3) if np.all(np.sum(a * a3[:i], axis=-1) > -0.999)])
    a5 = np.stack([a for i, a in enumerate(a5) if np.all(np.sum(a * a5[:i], axis=-1) > -0.999)])

    assert len(a2) == 15  # 30
    assert len(a3) == 10  # 20
    assert len(a5) == 6  #12
    icosahedral_axes_all = {
        2: hnormalized(a2),
        3: hnormalized(a3),
        5: hnormalized(a5),
    }
    return icosahedral_axes_all

icosahedral_axes_all = _icosahedral_axes_all()

def _d_axes(nfold):
    return {2: hnormalized([1, 0, 0]), nfold: hnormalized([0, 0, 1])}

def _d_frames(nfold):
    cx = hrot([0, 0, 1], np.pi * 2 / nfold)
    c2 = hrot([1, 0, 0], np.pi)
    frames = list()
    for i2 in range(2):
        rot1 = [np.eye(4), c2][i2]
        for ix in range(nfold):
            rot2 = hrot([0, 0, 1], np.pi * 2 * ix / nfold)
            frames.append(rot1 @ rot2)
    return np.array(frames)

def _d_axes_all(nfold):
    frames = _d_frames(nfold)
    a2 = frames @ [1, 0, 0, 0]
    an = frames @ [0, 0, 1, 0]

    # six decimals enough to account for numerical errors
    a2 = a2[np.unique(np.around(a2, decimals=6), axis=0, return_index=True)[1]]
    an = an[np.unique(np.around(an, decimals=6), axis=0, return_index=True)[1]]

    a2 = np.stack([a for i, a in enumerate(a2) if np.all(np.sum(a * a2[:i], axis=-1) > -0.999)])
    an = np.stack([a for i, a in enumerate(an) if np.all(np.sum(a * an[:i], axis=-1) > -0.999)])

    assert len(a2) == nfold  # 30
    assert len(an) == 1  # 20
    axes_all = {
        2: hnormalized(a2),
        nfold: hnormalized(an),
    }
    return axes_all

symaxes = dict(
    tet=tetrahedral_axes,
    oct=octahedral_axes,
    icos=icosahedral_axes,
)
for i in range(2, 10):
    symaxes['d%i' % i] = _d_axes(i)
symaxes_all = dict(
    tet=tetrahedral_axes_all,
    oct=octahedral_axes_all,
    icos=icosahedral_axes_all,
    d3=_d_axes_all(3),
    d5=_d_axes_all(5),
)

tetrahedral_angles = {(i, j): angle(
    tetrahedral_axes[i],
    tetrahedral_axes[j],
) for i, j in [
    (2, 3),
    (3, 7),
]}
octahedral_angles = {(i, j): angle(
    octahedral_axes[i],
    octahedral_axes[j],
) for i, j in [
    (2, 3),
    (2, 4),
    (3, 4),
]}
icosahedral_angles = {(i, j): angle(
    icosahedral_axes[i],
    icosahedral_axes[j],
) for i, j in [
    (2, 3),
    (2, 5),
    (3, 5),
]}
nfold_axis_angles = dict(
    tet=tetrahedral_angles,
    oct=octahedral_angles,
    icos=icosahedral_angles,
)
sym_point_angles = dict(tet={
    2: [np.pi],
    3: [np.pi * 2 / 3]
}, oct={
    2: [np.pi],
    3: [np.pi * 2 / 3],
    4: [np.pi / 2]
}, icos={
    2: [np.pi],
    3: [np.pi * 2 / 3],
    5: [np.pi * 2 / 5, np.pi * 4 / 5]
}, d3={
    2: [np.pi],
    3: [np.pi * 2 / 3],
})
for i in range(3, 10):
    sym_point_angles['d%i' % i] = {
        2: [np.pi],
        i: [np.pi * 2 / j for j in range(1, i) if j % 2 == 1]
    }

sym_frames = dict(
    tet=tetrahedral_frames,
    oct=octahedral_frames,
    icos=icosahedral_frames,
)
for i in range(3, 10):
    sym_frames['d%i' % i] = _d_frames(i)
