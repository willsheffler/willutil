import os
import numpy as np
import willutil as wu


def main():
    make_ori_samples("/home/sheffler/tmp/gr00.pdb", anglespacing=20, maxangle=45)


def euler_ori_samples(maxangle=45, nsamp=16, nsamp_short=4):
    ang = np.radians(maxangle)
    # nsamp_short = int(nsamp * ang / np.pi)
    xr = wu.hrot([1, 0, 0], np.linspace(-np.pi, np.pi, nsamp))
    yr = wu.hrot([0, 1, 0], np.linspace(-ang, ang, nsamp_short))
    zr = wu.hrot([0, 0, 1], np.linspace(-ang, ang, nsamp_short))
    orientation_samples = wu.hxform(wu.hxform(xr, yr), zr)
    orientation_samples = orientation_samples.reshape(-1, 4, 4)
    return orientation_samples


def karney_ori_samples(maxangle=45, anglespacing=20):
    quat, weight = wu.sampling.orientations.quaternion_set_with_covering_radius_degrees(anglespacing)
    orientation_samples = wu.homog.quat_to_xform(quat)
    newx = wu.hxform(orientation_samples, [1, 0, 0])
    ang = wu.hangle(newx, [1, 0, 0])
    ok = ang < np.radians(maxangle)
    orientation_samples = orientation_samples[ok]
    return orientation_samples


def random_ori_samples(maxangle, nsamp):
    maxangle = np.radians(maxangle)
    xforms = list()
    for i in range(nsamp):
        while True:
            x = wu.hrand(cart_sd=0)
            if wu.hangle(wu.hxform(x, [1, 0, 0]), [1, 0, 0]) < maxangle:
                break
        xforms.append(x)
    return np.stack(xforms)


def make_ori_samples(pdbfile, anglespacing, maxangle):
    orientation_samples_karney = karney_ori_samples(maxangle, anglespacing)
    orientation_samples_euler = euler_ori_samples(maxangle, nsamp=18, nsamp_short=4)
    orientation_samples_rand = random_ori_samples(maxangle, 300)
    # wu.showme(orientation_samples_karney)
    # wu.showme(orientation_samples_euler)
    # wu.showme(orientation_samples_rand)

    # assert 0

    pdb = wu.readpdb(pdbfile)
    for iori, xori in enumerate(orientation_samples_rand):
        newfname = os.path.basename(pdbfile).replace(".pdb", "") + f"_{iori:04}.pdb"
        pdb.dump_pdb(newfname, xform=xori)


if __name__ == "__main__":
    main()
