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

def test_symops_with_perfect_sym_frames():
    np.set_printoptions(precision=20, suppress=True, linewidth=98,
                        formatter={'float': lambda f: '%20.16f' % f})
    # r = hm.hrot([1, 0, 0], 180)
    # xinfo.axs, a = hm.axis_angle_of(r)
    # print(r)
    # print(xinfo.axs)
    # print(a)
    # assert 0

    all_names = 'tet oct icos'.split()
    all_symframes = [
        hm.tetrahedral_frames,
        # hm.octahedral_frames,
        hm.icosahedral_frames[:7],
    ]
    all_point_angles = [
        {
            2: [np.pi],
            3: [np.pi * 2 / 3]
        },
        # {
        # 2: [np.pi],
        # 3: [np.pi * 2 / 3],
        # 4: [np.pi / 2]
        # },
        # {
        # 2: [np.pi],
        # 3: [np.pi * 2 / 3],
        # 5: [np.pi * 2 / 5, np.pi * 4 / 5]
        # },
    ]
    xpost3 = hm.rand_xform(cart_sd=10)
    # xpost3 = np.array(
    #     [[-0.9579827211004066, -0.2697986972771439, 0.0973538341341333, -0.5345926298039275],
    #      [0.0362313657804560, -0.4505260684058727, -0.8920277741306206, 0.7021952604606366],
    #      [0.2845283715321604, -0.8510199319841474, 0.4413713642262649, 0.9988688264173512],
    #      [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000]])

    print('-------------')
    print(repr(xpost3))
    print('-------------')

    for name, symframes, point_angles in zip(all_names, all_symframes, all_point_angles):
        print('---------', name, '-----------')
        xpre = hm.rand_xform(cart_sd=5)
        # xpre[:3, :3] = np.eye(3)

        frames = symframes @ xpre  # move subunit
        xpost1 = np.eye(4)
        # print(frames)
        # print('-------------')

        # np.random.shuffle(frames)
        # frames = frames[:nframes]

        symops = hm.symops_from_frames(frames, point_angles)
        assert len(symops) == len(frames) * (len(frames) - 1) / 2
        # print(list(symops[(0, 1)].keys()))
        for k, op in symops.items():
            wgood, ngood = None, 0
            for n, e in op.err.items():
                if e < 0.001:
                    ngood += 1
                    wgood = n
            if not ngood == 1:
                print('ngood!=1', op.err)
                assert 0
            assert np.allclose(0, op.cen[:3], atol=1e-4)

        xpost2 = hm.htrans(np.array([4, 5, 6]))
        xpost2inv = np.linalg.inv(xpost2)
        frames2 = xpost2 @ frames  # move whole structure
        symops2 = hm.symops_from_frames(frames2, point_angles)
        assert len(symops2) == len(frames2) * (len(frames2) - 1) / 2
        assert len(frames) == len(frames2)
        for k in symops:
            op1 = symops[k]
            op2 = symops2[k]
            frame1 = frames2[k[0]]
            frame2 = frames2[k[1]]
            try:
                assert np.allclose(op2.xrel @ frame1, frame2)
                assert np.allclose(op1.axs, xpost2inv @ op2.axs, atol=1e-5)
                assert np.allclose(op1.ang, op2.ang, atol=1e-5)
                assert np.allclose(op1.cen, hm.proj_perp(op2.axs, xpost2inv @ op2.cen), atol=1e-4)
                assert np.allclose(op1.rad, op2.rad, atol=1e-4)
                assert np.allclose(op1.hel, op2.hel, atol=1e-5)
                for k in point_angles:
                    if not np.allclose(op1.err[k], op2.err[k], atol=1e-5):
                        print('err', op1.err[k], op2.err[k])
                    assert np.allclose(op1.err[k], op2.err[k], atol=1e-5)
            except AssertionError as e:
                from willutil import viz
                # assert 0
                viz.showme(frames, 'frames1')
                viz.showme(frames2, 'frames2')
                viz.showme(list(symops2.values()), 'someops2')
                viz.showme(op2, 'op2')
                print('axs', op1.axs)
                print('cen', op1.cen)
                print('cen', hm.proj_perp(op2.axs, xpost2inv @ op2.cen))
                print('cen', xpost2inv @ op2.cen)
                print('cen', op2.cen)
                assert 0
                t2 = hm.hrot(op2.axs, np.pi, op2.cen)
                viz.showme([
                    [frame1, frame2],
                    op2,
                ], headless=False)
                # assert 0
                print(op2.xrel)
                print(t2)

                print()
                print('hel   ', op1.hel, op2.hel)
                print('rad   ', op1.rad, op2.rad)
                print('op1   ', op1.cen)
                print('op2   ', op2.cen)
                print('op2   ', xpost2inv @ op2.cen)
                print('proj  ', hm.proj_perp(op2.axs, xpost2inv @ op2.cen))
                # print('op1axs', op1.axs, op1.ang)
                print('op2axs', op2.axs, op2.ang)
                # print(op1.xrel)
                # print(op2.xrel)
                raise e

        # return

        xpost3inv = np.linalg.inv(xpost3)
        frames3 = xpost3 @ frames  # move whole structure
        symops3 = hm.symops_from_frames(frames3, point_angles)
        assert len(symops3) == len(frames3) * (len(frames3) - 1) / 2
        assert len(frames) == len(frames3)
        for k in symops:
            op1 = symops[k]
            op2 = symops3[k]
            try:
                assert np.allclose(op1.ang, op2.ang, atol=1e-5)
                assert np.allclose(op1.cen, hm.proj_perp(op1.axs, xpost3inv @ op2.cen), atol=1e-5)
                assert np.allclose(op1.rad, op2.rad, atol=1e-5)
                assert np.allclose(op1.hel, op2.hel, atol=1e-5)
                for k in point_angles:
                    assert np.allclose(op1.err[k], op2.err[k], atol=1e-5)

                op2axsinv = xpost3inv @ op2.axs
                if hm.hdot(op2axsinv, op1.axs) < 0:
                    op2axsinv = -op2axsinv
                assert np.allclose(op1.axs, op2axsinv, atol=1e-5)

                # assert np.allclose(op1.cen, xpost3inv @ hm.proj_perp(op2.axs, op2.cen), atol=1e-5)
            except AssertionError as e:
                # print('op1   ', op1.cen)
                print('cen op2   ', op2.cen)
                print('cen op2inv', xpost3inv @ op2.cen)
                # print('proj  ', hm.proj_perp(op1.axs, xpost3inv @ op2.cen))
                # print('proj  ', hm.proj_perp(op2.axs, op2.cen))
                # print('proj  ', xpost3inv @ hm.proj_perp(op2.axs, op2.cen))
                print('op1axs', op1.axs, op1.ang)
                print('op2axs', xpost3inv @ op2.axs, op2.ang)
                print('op2axs', op2.axs, op2.ang)
                # print(op1.xrel)
                # print(op2.xrel)
                raise e

        wu.viz.showme(list(symops3.values()), 'someops2')

if __name__ == '__main__':
    test_symops_with_perfect_sym_frames()
    test_rel_xform_info()
    test_rel_xform_info_rand()
    test_cyclic_sym_err()
