from willutil.homog.hgeom import *
from willutil.homog.symframes import *

tetrahedral_axes = {
    2: hnormalized([1, 0, 0]),
    3: hnormalized([1, 1, 1]),
    7: hnormalized([1, 1, -1])  # other c3
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
# yapf: disable
tetrahedral_axes_all = {
    2: hnormalized([
        [ 1,  0,  0],
        [ 0,  1,  0],
        [ 0,  0,  1],
        [-1,  0,  0],
        [ 0, -1,  0],
        [ 0,  0, -1],
    ]),
    3: hnormalized([
        [ 1,  1,  1],
        [ 1, -1, -1],
        [-1, -1,  1],
        [-1,  1, -1],
    ]),
    7: hnormalized([
        [-1,  1,  1],
        [ 1, -1,  1],
        [ 1,  1, -1],
        [-1, -1,- 1],
    ]),
}
octahedral_axes_all = {
    2:
    hnormalized([
        [ 1,  1,  0],
        [ 0,  1,  1],
        [ 1,  0,  1],
        [-1,  1,  0],
        [ 0, -1,  1],
        [-1,  0,  1],
        [ 1, -1,  0],
        [ 0,  1, -1],
        [ 1,  0, -1],
        [-1, -1,  0],
        [ 0, -1, -1],
        [-1,  0, -1],
    ]),
    3:
    hnormalized([
        [ 1,  1,  1],
        [-1,  1,  1],
        [ 1, -1,  1],
        [ 1,  1, -1],
        [-1,  1, -1],
        [-1, -1,  1],
        [ 1, -1, -1],
        [-1, -1, -1],
    ]),
    4:
    hnormalized([
        [ 1,  0,  0],
        [ 0,  1,  0],
        [ 0,  0,  1],
        [-1,  0,  0],
        [ 0, -1,  0],
        [ 0,  0, -1],
    ]),
}
# yapf: enable

def _icosahedral_axes_all():
    a2 = icosahedral_frames @ icosahedral_axes[2]
    a3 = icosahedral_frames @ icosahedral_axes[3]
    a5 = icosahedral_frames @ icosahedral_axes[5]

    # six decimals enough to account for numerical errors
    a2 = a2[np.unique(np.around(a2, decimals=6), axis=0, return_index=True)[1]]
    a3 = a3[np.unique(np.around(a3, decimals=6), axis=0, return_index=True)[1]]
    a5 = a5[np.unique(np.around(a5, decimals=6), axis=0, return_index=True)[1]]
    assert len(a2) == 30
    assert len(a3) == 20
    assert len(a5) == 12
    icosahedral_axes_all = {2: a2, 3: a3, 5: a5}
    return icosahedral_axes_all

icosahedral_axes_all = _icosahedral_axes_all()

symaxes = dict(tet=tetrahedral_axes, oct=octahedral_axes, icos=icosahedral_axes)
symaxes_all = dict(tet=tetrahedral_axes_all, oct=octahedral_axes_all, icos=icosahedral_axes_all)

tetrahedral_angles = {(i, j): angle(tetrahedral_axes[i], tetrahedral_axes[j]) for i, j in [
    (2, 3),
    (3, 7),
]}
octahedral_angles = {(i, j): angle(octahedral_axes[i], octahedral_axes[j]) for i, j in [
    (2, 3),
    (2, 4),
    (3, 4),
]}
icosahedral_angles = {(i, j): angle(icosahedral_axes[i], icosahedral_axes[j]) for i, j in [
    (2, 3),
    (2, 5),
    (3, 5),
]}

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
})
sym_frames = dict(tet=tetrahedral_frames, oct=octahedral_frames, icos=icosahedral_frames)
