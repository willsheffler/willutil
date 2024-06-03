import numpy as np
import willutil as wu


def prune_radial_outliers(xyz, nprune=10):
    npoints = len(xyz) - nprune
    for i in range(nprune):
        com = wu.hcom(xyz)
        r = wu.hnorm(xyz - com)
        w = np.argsort(r)
        xyz = xyz[w[:-1]]
    return xyz


def point_cloud(npoints=100, std=10, outliers=0):
    xyz = wu.hrandpoint(npoints + outliers, std=10)
    xyz = prune_radial_outliers(xyz, outliers)
    assert len(xyz) == npoints
    xyz = xyz[np.argsort(xyz[:, 0])]
    xyz -= wu.hvec(wu.hcom(xyz))
    return xyz
