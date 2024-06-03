import sys

sys.path.append("/home/sheffler/src/willutil")
import numpy as np
import willutil as wu


def main():
    # generate 3 point clouds that are not quite C3 symmetrical
    # there is deviation in the axis and angle of rotation, as well
    # as noise in the positions of the points
    point_spread_sd = 1.0
    axisA = wu.hnormalized([1.0, 2.0, 3.0, 0])
    axisB = wu.hnormalized([1.1, 2.1, 3.1, 0])
    cenA = np.array([17.0, 28.0, 9.0, 1.0])
    cenB = np.array([17.2, 28.5, 9.7, 1.0])
    rotA = wu.hrot(axis=axisA, angle=119, center=cenA)
    rotB = wu.hrot(axis=axisB, angle=242, center=cenB)
    xyz1 = wu.tests.point_cloud(100, std=30, outliers=0)
    xyz2 = wu.hxform(rotA, xyz1 + point_spread_sd * wu.hrandvec(100))
    xyz3 = wu.hxform(rotB, xyz1 + point_spread_sd * wu.hrandvec(100))

    # compute rms fit for each pair of point clouds
    rms12, fit12, xform12 = wu.hrmsfit(xyz1, xyz2)
    rms23, fit12, xform23 = wu.hrmsfit(xyz2, xyz3)
    rms31, fit31, xform31 = wu.hrmsfit(xyz3, xyz1)
    print("rms AB/BC/CA", rms12, rms23, rms31)

    # get the axis, angle and 'center' of rotation for the
    # best-fit xforms between point clouds
    axis12, ang12, cen12 = wu.haxis_ang_cen_of(xform12)
    axis23, ang23, cen23 = wu.haxis_ang_cen_of(xform23)
    axis31, ang31, cen31 = wu.haxis_ang_cen_of(xform31)

    # can guess axis and center
    axis_guess = (axis12 + axis23 + axis31) / 3
    ang_guess = (ang12 + ang23 + ang31) / 3
    cen_guess = (cen12 + cen23 + cen31) / 3
    print("rotation axis vectors:", axisA, axisB, axis_guess, sep="\n   ")
    print("rotation angles", 119, 241, ang_guess * 180 / np.pi)

    # cen guess above probably won't match original cen, because the center of
    # rotation is really a line defined by the symmetry axis vector and a 'center'
    # point that can be anywhere along this line.
    # Can check how close our cen_guess is to the actual sym axis:
    dist1 = wu.homog.line_line_distance_pa(cenA, axisA, cen_guess, axis_guess)
    dist2 = wu.homog.line_line_distance_pa(cenA, axisB, cen_guess, axis_guess)
    print("distances from guess to actual rotation axis (line)", dist1, dist2)


if __name__ == "__main__":
    main()
