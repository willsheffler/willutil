from willutil import Bunch
from willutil.homog.hgeom import *
from willutil.sym.symframes import *

m = -1

ambiguous_axes = dict(
    tet=[],
    oct=[(2, 4)],
    icos=[],
)

tetrahedral_axes = {
    2: hnormalized([1, 0, 0]),
    3: hnormalized([1, 1, 1]),
    '2b': hnormalized([1, 1, m])  # other c3
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
    '2b':
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
    for ix in range(nfold):
        rot2 = hrot([0, 0, 1], np.pi * 2 * ix / nfold)
        for i2 in range(2):
            rot1 = [np.eye(4), c2][i2]
            frames.append(rot1 @ rot2)
    return np.array(frames)

def _d_axes_all(nfold):
    ang = 2 * np.pi / nfold
    frames = _d_frames(nfold)
    a2A = frames @ [1, 0, 0, 0]
    anA = frames @ [0, 0, 1, 0]
    if nfold % 2 == 0:
        a2A = np.concatenate([a2A, frames @ [np.cos(ang / 2), np.sin(ang / 2), 0, 0]])

    # six decimals enough to account for numerical errors
    a2B = a2A[np.unique(np.around(a2A, decimals=6), axis=0, return_index=True)[1]]
    anB = anA[np.unique(np.around(anA, decimals=6), axis=0, return_index=True)[1]]
    a2B = np.flip(a2B, axis=0)

    a2 = np.stack([a for i, a in enumerate(a2B) if np.all(np.sum(a * a2B[:i], axis=-1) > -0.999)])
    an = np.stack([a for i, a in enumerate(anB) if np.all(np.sum(a * anB[:i], axis=-1) > -0.999)])

    # if nfold == 4:
    #     print(np.around(a2A, decimals=3))
    #     print()
    #     print(np.around(a2B, decimals=3))
    #     print()
    #     print(np.around(a2, decimals=3))
    #     print()

    assert len(an) == 1, f'nfold {nfold}'
    assert len(a2) == nfold, f'nfold {nfold}'

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

symaxes_all = dict(
    tet=tetrahedral_axes_all,
    oct=octahedral_axes_all,
    icos=icosahedral_axes_all,
)

tetrahedral_angles = {(i, j): angle(
    tetrahedral_axes[i],
    tetrahedral_axes[j],
) for i, j in [
    (2, 3),
    (3, '2b'),
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

sym_frames = dict(
    tet=tetrahedral_frames,
    oct=octahedral_frames,
    icos=icosahedral_frames,
)
minsymang = dict(
    tet=angle(tetrahedral_axes[2], tetrahedral_axes[3]) / 2,
    oct=angle(octahedral_axes[2], octahedral_axes[3]) / 2,
    icos=angle(icosahedral_axes[2], icosahedral_axes[3]) / 2,
    d2=np.pi / 4,
)
for icyc in range(3, 33):
    sym = 'd%i' % icyc
    symaxes[sym] = _d_axes(icyc)
    sym_frames[sym] = _d_frames(icyc)
    ceil = int(np.ceil(icyc / 2))
    sym_point_angles[sym] = {
        2: [np.pi],
        icyc: [np.pi * 2 * j / icyc for j in range(1, ceil)],
    }
    minsymang[sym] = np.pi / icyc / 2
    symaxes_all[sym] = _d_axes_all(icyc)
    if icyc % 2 == 0:
        ambiguous_axes[sym] = [(2, icyc)]

sym_frames['d2'] = np.stack([
    np.eye(4),
    hrot([1, 0, 0], np.pi),
    hrot([0, 1, 0], np.pi),
    hrot([0, 0, 1], np.pi),
])

symaxes['d2'] = {
    '2a': np.array([1, 0, 0, 0]),
    '2b': np.array([0, 1, 0, 0]),
    '2c': np.array([0, 0, 1, 0]),
}
symaxes_all['d2'] = {
    2: np.array([
        np.array([1, 0, 0, 0]),
        np.array([0, 1, 0, 0]),
        np.array([0, 0, 1, 0]),
    ])
}

sym_point_angles['d2'] = {2: [np.pi]}

def sym_nfold_map(nfold):
    if isinstance(nfold, str):
        return int(nfold[:-1])
    return nfold

def get_syminfo(sym):
    sym = sym.lower()
    try:
        ambig = list()
        if sym in ambiguous_axes: ambig = ambiguous_axes[sym]
        nfoldmap = {k: sym_nfold_map(k) for k in symaxes[sym]}
        assert sym_frames[sym].shape[-2:] == (4, 4)
        return Bunch(
            frames=sym_frames[sym],
            axes=symaxes[sym],
            axesall=symaxes_all[sym],
            point_angles=sym_point_angles[sym],
            ambiguous_axes=ambig,
            nfoldmap=nfoldmap,
        )

    except KeyError as e:
        # raise ValueError(f'sim.py: dont know symmetry "{sym}"')
        print(f'sym.py: dont know symmetry "{sym}"')
        raise e

_sym_permute_axes_choices = dict(
    d2=np.array([
        np.eye(4),  #           x y z
        hrot([1, 0, 0], 90),  # x z y
        hrot([0, 0, 1], 90),  # y z x
        hrot([1, 0, 0], 90) @ hrot([0, 0, 1], 90),  # y x z
        hrot([0, 1, 0], 90),  # z y x
        hrot([1, 0, 0], 90) @ hrot([0, 1, 0], 90),  # z y x        
    ]),
    d3=np.array([
        np.eye(4),
        hrot([0, 0, 1], 180),
    ]),
)

def sym_permute_axes_choices(sym):
    if sym in _sym_permute_axes_choices:
        return _sym_permute_axes_choices[sym]
    else:
        return np.eye(4).reshape(1, 4, 4)
