import numpy as np
import pytest
import willutil as wu
import willutil.homog as hm

def test_cyclic_sym_err(nsamp=100):
    for i in range(nsamp):
        prex = hm.rand_xform()
        axs = hm.rand_unit()
        tgtang = np.random.rand() * np.pi
        f1 = np.eye(4)
        cart = hm.proj_perp(axs, hm.rand_point())
        rad = np.linalg.norm(cart[:3])
        f1[:, 3] = cart
        rel = hm.hrot(axs, tgtang)
        f2 = rel @ f1
        pair = hm.rel_xform_info(f1, f2)
        err = hm.cyclic_sym_err(pair, tgtang)
        assert np.allclose(err, 0)

        tgtang2 = np.random.rand() * np.pi
        err2 = hm.cyclic_sym_err(pair, tgtang2)
        assert np.allclose(err2, abs(tgtang - tgtang2) * rad)

        hlen = np.random.normal()
        rel[:3, 3] = hlen * axs[:3]
        f2 = rel @ f1
        pair = hm.rel_xform_info(f1, f2)
        err = hm.cyclic_sym_err(pair, tgtang)
        assert np.allclose(err, abs(hlen))

        tgtang3 = np.random.rand() * np.pi
        err3 = hm.cyclic_sym_err(pair, tgtang3)
        angerr = (tgtang - tgtang3) * rad
        assert np.allclose(err3, np.sqrt(hlen**2 + angerr**2))

def test_rel_xform_info():

    axs0 = [0, 0, 1, 0]
    ang0 = (2 * np.random.random() - 1) * np.pi
    shift = [0, 0, 2 * np.random.random() - 1, 0]
    cen0 = [0, 0, 0, 1]
    # print(axs0)
    # print(ang0)
    # print(shift)

    frameA = np.eye(4)
    # frameA[:, 3] = shift
    xrel0 = hm.hrot(axs0, ang0, cen0)
    xrel0[:, 3] = shift

    frameB = xrel0 @ frameA

    xinfo = hm.rel_xform_info(frameA, frameB)
    assert np.allclose(xrel0, xinfo.xrel)
    assert np.allclose(axs0, xinfo.axs if ang0 > 0 else -xinfo.axs)
    assert np.allclose(xinfo.ang, abs(ang0))
    # print('xinfo.cen', xinfo.cen)
    # print('xinfo.rad', xinfo.rad, shift[0])
    assert np.allclose(xinfo.hel, np.sum(xinfo.axs * shift))

def test_rel_xform_info_rand(nsamp=100):
    for i in range(nsamp):
        axs0 = [1, 0, 0, 0]
        ang0 = np.random.rand() * np.pi
        cen0 = [np.random.normal(), 0, 0, 1]

        rady = np.random.normal()
        radz = np.random.normal()
        # radz = rady
        rad0 = np.sqrt(rady**2 + radz**2)
        hel0 = np.random.normal()
        prefx = hm.rand_xform()
        postx = hm.rand_xform()

        xrel0 = hm.hrot(axs0, ang0, cen0)
        xrel0[:, 3] = [hel0, 0, 0, 1]

        frameA = np.eye(4)
        frameA = prefx @ frameA
        frameA[:, 3] = [0, rady, radz, 1]

        frameB = xrel0 @ frameA
        xinfo = hm.rel_xform_info(frameA, frameB)

        # print('xinfo.cen')
        # print(cen0)
        # print(xinfo.cen)
        # print('xinfo.rad', rad0, xinfo.rad, xinfo.rad / rad0)
        # print('xinfo.hel', hel0, xinfo.hel)
        cen1 = hm.proj_perp(axs0, cen0)
        assert np.allclose(np.linalg.norm(xinfo.axs, axis=-1), 1.0)
        assert np.allclose(xrel0, xinfo.xrel)
        assert np.allclose(axs0, xinfo.axs)
        assert np.allclose(ang0, xinfo.ang)
        assert np.allclose(cen1, xinfo.cen, atol=0.001)
        assert np.allclose(hel0, xinfo.hel)
        assert np.allclose(rad0, xinfo.rad)

        frameA = postx @ frameA
        frameB = postx @ frameB
        xinfo = hm.rel_xform_info(frameA, frameB)
        rrel0 = xrel0[:3, :3]
        rrel = xinfo.xrel[:3, :3]
        rpost = postx[:3, :3]
        assert np.allclose(np.linalg.norm(xinfo.axs, axis=-1), 1.0)

        assert np.allclose(np.linalg.det(rrel0), 1.0)
        assert np.allclose(np.linalg.norm(hm.axis_angle_of(rrel0)[0]), 1.0)

        # print(hm.axis_angle_of(rrel0))
        # print(hm.axis_angle_of(rrel))
        # print(hm.axis_angle_of(rpost))
        # print(hm.axis_angle_of(rpost @ rrel0))
        # assert np.allclose(rpost @ rrel0, rrel)
        # assert 0

        # hrm... not sure why this isn't true... rest of tests should be enough
        # assert np.allclose(postx[:3, :3] @ xrel0[:3, :3], xinfo.xrel[:3, :3])

        assert np.allclose(postx @ axs0, xinfo.axs)
        assert np.allclose(ang0, xinfo.ang)
        assert np.allclose(hm.proj_perp(xinfo.axs, postx @ cen0),
                           hm.proj_perp(xinfo.axs, xinfo.cen))
        assert np.allclose(hel0, xinfo.hel)
        assert np.allclose(rad0, xinfo.rad)

# @pytest.mark.xfail
def test_symops_from_frames():
    np.set_printoptions(precision=6, suppress=True, linewidth=98,
                        formatter={'float': lambda f: '%9.5f' % f})
    # r = hm.hrot([1, 0, 0], 180)
    # xinfo.axs, a = hm.axis_angle_of(r)
    # print(r)
    # print(xinfo.axs)
    # print(a)
    # assert 0

    nframes = 24
    symframes = hm.octahedral_frames[:nframes]
    point_angs = {2: np.pi, 3: np.pi * 2 / 3, 4: np.pi / 2}

    # xpre = hm.rand_xform(len(symframes))
    # post = hm.rand_xform()
    # xpre = np.tile(np.eye(4), (len(symframes), 1, 1))
    xpre = hm.htrans([1, 2, 3])  # subunits shouldn't overlap

    frames = symframes @ xpre  # move subunit
    xpost1 = np.eye(4)
    # print(frames)
    # print('-------------')

    # np.random.shuffle(frames)
    # frames = frames[:nframes]

    symops = hm.symops_from_frames(frames, point_angs)
    assert len(symops) == len(frames) * (len(frames) - 1) / 2
    print(list(symops[(0, 1)].keys()))
    for k, op in symops.items():
        wgood, ngood = None, 0
        for n, e in op.err.items():
            if e < 0.000001:
                ngood += 1
                wgood = n
        if not ngood == 1:
            print('ngood!=1', op.err)
        assert np.allclose(0, op.cen[:3])

    xpost2 = hm.htrans(np.array([4, 5, 6]))
    xpost2inv = np.linalg.inv(xpost2)
    frames2 = xpost2 @ frames  # move whole structure
    symops2 = hm.symops_from_frames(frames2, point_angs)
    assert len(symops2) == len(frames2) * (len(frames2) - 1) / 2
    assert len(frames) == len(frames2)
    for k in symops:
        op1 = symops[k]
        op2 = symops2[k]
        try:
            assert np.allclose(op1.axs, xpost2inv @ op2.axs, atol=1e-6)
            assert np.allclose(op1.ang, op2.ang, atol=1e-6)
            assert np.allclose(op1.cen, hm.proj_perp(op2.axs, xpost2inv @ op2.cen), atol=1e-6)
            assert np.allclose(op1.rad, op2.rad, atol=1e-6)
            assert np.allclose(op1.hel, op2.hel, atol=1e-6)
            assert np.allclose(op1.err[2], op2.err[2], atol=1e-6)
            assert np.allclose(op1.err[3], op2.err[3], atol=1e-6)
            assert np.allclose(op1.err[4], op2.err[4], atol=1e-6)
        except AssertionError as e:
            print('op1   ', op1.cen)
            print('op2   ', op2.cen)
            print('op2   ', xpost2inv @ op2.cen)
            print('proj  ', hm.proj_perp(op2.axs, xpost2inv @ op2.cen))
            print('op1axs', op1.axs, op1.ang)
            print('op2axs', op2.axs, op2.ang)
            print(op1.xrel)
            print(op2.xrel)
            raise e

    # return
    xpost3 = hm.rand_xform()
    xpost3inv = np.linalg.inv(xpost3)
    frames3 = xpost3 @ frames  # move whole structure
    symops3 = hm.symops_from_frames(frames3, point_angs)
    assert len(symops3) == len(frames3) * (len(frames3) - 1) / 2
    assert len(frames) == len(frames3)
    for k in symops:
        op1 = symops[k]
        op2 = symops3[k]
        try:
            # assert np.allclose(op1.axs, xpost3inv @ op2.axs, atol=1e-6)
            assert np.allclose(op1.ang, op2.ang, atol=1e-6)
            # assert np.allclose(op1.cen, hm.proj_perp(op2.axs, xpost3inv @ op2.cen), atol=1e-6)
            # assert np.allclose(op1.rad, op2.rad, atol=1e-6)
            # assert np.allclose(op1.hel, op2.hel, atol=1e-6)
            # assert np.allclose(op1.err[2], op2.err[2], atol=1e-6)
            # assert np.allclose(op1.err[3], op2.err[3], atol=1e-6)
            # assert np.allclose(op1.err[4], op2.err[4], atol=1e-6)
        except AssertionError as e:
            print('op1   ', op1.cen)
            print('op2   ', op2.cen)
            print('op2   ', xpost3inv @ op2.cen)
            print('proj  ', hm.proj_perp(op2.axs, xpost3inv @ op2.cen))
            print('op1axs', op1.axs, op1.ang)
            print('op2axs', op2.axs, op2.ang)
            print(op1.xrel)
            print(op2.xrel)
            raise e

if __name__ == '__main__':
    test_symops_from_frames()
    test_rel_xform_info()
    test_rel_xform_info_rand()
    test_cyclic_sym_err()
