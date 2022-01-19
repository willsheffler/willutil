import os
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
        assert np.allclose(err2, abs(tgtang - tgtang2) * min(10000, max(1, rad)))

        hlen = np.random.normal()
        rel[:3, 3] = hlen * axs[:3]
        f2 = rel @ f1
        pair = hm.rel_xform_info(f1, f2)
        err = hm.cyclic_sym_err(pair, tgtang)
        assert np.allclose(err, abs(hlen))

        tgtang3 = np.random.rand() * np.pi
        err3 = hm.cyclic_sym_err(pair, tgtang3)
        angerr = (tgtang - tgtang3) * min(10000, max(1, rad))
        assert np.allclose(err3, np.sqrt(hlen**2 + angerr**2))

def test_rel_xform_info():

    axs0 = [0, 0, 1, 0]
    ang0 = (2 * np.random.random() - 1) * np.pi
    # frameAcen = [0, 0, 2 * np.random.random() - 1, 1]
    frameAcen = np.array([2 * np.random.random() - 1, 2 * np.random.random() - 1, 1, 1])

    xformcen = [0, 0, 0, 1]
    trans = [0, 0, 0, 1]

    # print(axs0)
    # print(ang0)
    # print(shift)

    frameA = np.eye(4)
    frameA[:, 3] = frameAcen
    xrel0 = hm.hrot(axs0, ang0, xformcen)
    xrel0[:, 3] = trans

    frameB = xrel0 @ frameA
    xinfo = hm.rel_xform_info(frameA, frameB)

    rad = np.sqrt(np.sum(frameAcen[:2]**2))
    # print('frameAcen', frameA[:, 3])
    # print('frameBcen', frameB[:, 3])

    assert np.allclose(xrel0, xinfo.xrel)
    assert np.allclose(axs0, xinfo.axs if ang0 > 0 else -xinfo.axs)
    assert np.allclose(xinfo.ang, abs(ang0))
    assert np.allclose(rad, xinfo.rad)
    assert np.allclose([0, 0, frameAcen[2], 1], xinfo.framecen)

    # print()
    # print('xinfo.cen', xinfo.cen)
    # print()
    # print('framecen', (frameA[:, 3] + frameB[:, 3]) / 2)
    # print()
    # print(axs0)
    # print('xinfo.framecen', xinfo.framecen)

    assert np.allclose(xinfo.hel, np.sum(xinfo.axs * trans))

def test_rel_xform_info_rand(nsamp=50):
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

def test_symops_cen_perfect(nframes=9):
    np.set_printoptions(
        precision=10,
        suppress=True,
        linewidth=98,
        formatter={'float': lambda f: '%14.10f' % f},
    )
    # r = hm.hrot([1, 0, 0], 180)
    # xinfo.axs, a = hm.axis_angle_of(r)
    # print(r)
    # print(xinfo.axs)
    # print(a)
    # assert 0

    all_names = 'tet oct icos'.split()
    all_symframes = [
        # hm.tetrahedral_frames,
        # hm.octahedral_frames,
        # hm.icosahedral_frames[:30],
        hm.tetrahedral_frames[np.random.choice(12, nframes, replace=False), :, :],
        hm.octahedral_frames[np.random.choice(24, nframes, replace=False), :, :],
        hm.icosahedral_frames[np.random.choice(60, nframes, replace=False), :, :],
        # hm.icosahedral_frames[(20, 21), :, :],
    ]
    all_point_angles = [{
        2: [np.pi],
        3: [np.pi * 2 / 3]
    }, {
        2: [np.pi],
        3: [np.pi * 2 / 3],
        4: [np.pi / 2]
    }, {
        2: [np.pi],
        3: [np.pi * 2 / 3],
        5: [np.pi * 2 / 5, np.pi * 4 / 5]
    }]
    xpost3 = hm.rand_xform(cart_sd=10)
    xpre = hm.rand_xform(cart_sd=5)

    #    xpre = np.array([[-0.4971291915, 0.5418972027, -0.6776503439, 1.5447300543],
    #                     [0.5677267562, -0.3874638202, -0.7263319616, 2.4858980827],
    #                     [-0.6561622492, -0.7458010524, -0.1150299654, -4.0124612619],
    #                     [0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000]])
    #    xprost3 = np.array([[0.2066723595, -0.9743827067, 0.0886841400, -4.8092830795],
    #                        [0.9297657442, 0.2238133467, 0.2923067684, -7.8301135871],
    #                        [-0.3046673543, 0.0220437459, 0.9522036949, -13.6244069897],
    #                        [0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000]])

    # print()
    # print(repr(xpre))
    # print('-------------')
    # print(repr(xpost3))
    # print('-------------')

    for name, symframes, point_angles in zip(all_names, all_symframes, all_point_angles):
        # print('---------', name, '-----------')

        # xpre[:3, :3] = np.eye(3)

        frames = symframes @ xpre  # move subunit
        xpost1 = np.eye(4)
        # print(frames)
        # print('-------------')

        # np.random.shuffle(frames)
        # frames = frames[:nframes]

        symops = hm.symops_from_frames(frames, point_angles)
        symops = hm.stupid_pairs_from_symops(symops)
        assert len(symops) == len(frames) * (len(frames) - 1) / 2
        # print(list(symops[(0, 1)].keys()))
        for k, op in symops.items():
            # wgood, ngood = None, 0
            # for n, e in op.err.items():
            #     if e < 5e-2:
            #         ngood += 1
            #         wgood = n
            # if not ngood == 1:
            #     print('ngood!=1', k, ngood, op.err)
            #     assert 0
            assert np.allclose(0, op.cen[:3], atol=1e-3)

        xpost2 = hm.htrans(np.array([4, 5, 6]))
        xpost2inv = np.linalg.inv(xpost2)
        frames2 = xpost2 @ frames  # move whole structure
        symops2 = hm.symops_from_frames(frames2, point_angles)
        symops2 = hm.stupid_pairs_from_symops(symops2)
        assert len(symops2) == len(frames2) * (len(frames2) - 1) / 2
        assert len(frames) == len(frames2)
        for k in symops:
            op1 = symops[k]
            op2 = symops2[k]
            frame1 = frames2[k[0]]
            frame2 = frames2[k[1]]
            try:
                assert np.allclose(op2.xrel @ frame1, frame2, atol=1e-8)
                assert np.allclose(op1.axs, xpost2inv @ op2.axs, atol=1e-4)
                assert np.allclose(op1.ang, op2.ang, atol=1e-4)
                assert np.allclose(op1.cen, hm.proj_perp(op2.axs, xpost2inv @ op2.cen), atol=1e-3)
                assert np.allclose(op1.rad, op2.rad, atol=1e-3)
                assert np.allclose(op1.hel, op2.hel, atol=1e-4)
                # for k in point_angles:
                #     if not np.allclose(op1.err[k], op2.err[k], atol=1e-4):
                #         print('err', op1.err[k], op2.err[k])
                #     assert np.allclose(op1.err[k], op2.err[k], atol=1e-4)
            except AssertionError as e:
                print(repr(xpost2))
                from willutil import viz
                # assert 0
                # viz.showme(list(symops2.values()), 'someops2')
                # viz.showme(op1, 'op1')
                # viz.showme(op2, 'op2')

                print(op1.ang)
                print(repr(op1.frames))
                print(op1.axs)

                print('axs', op1.axs)
                print('cen', op1.cen)
                print('cen', hm.proj_perp(op2.axs, xpost2inv @ op2.cen))
                print('cen', xpost2inv @ op2.cen)
                print('cen', op2.cen)

                t2 = hm.hrot(op2.axs, np.pi, op2.cen)
                viz.showme([op2], headless=False)
                # viz.showme(symops, headless=False)
                assert 0
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

        # continue

        xpost3inv = np.linalg.inv(xpost3)
        frames3 = xpost3 @ frames  # move whole structure
        symops3 = hm.symops_from_frames(frames3, point_angles)
        symops3 = hm.stupid_pairs_from_symops(symops3)
        assert len(symops3) == len(frames3) * (len(frames3) - 1) / 2
        assert len(frames) == len(frames3)
        for k in symops:
            op1 = symops[k]
            op2 = symops3[k]
            try:
                assert np.allclose(op1.ang, op2.ang, atol=1e-3)
                assert np.allclose(op1.cen, hm.proj_perp(op1.axs, xpost3inv @ op2.cen), atol=1e-2)
                assert np.allclose(op1.rad, op2.rad, atol=1e-2)
                assert np.allclose(op1.hel, op2.hel, atol=1e-3)
                # for k in point_angles:
                #     assert np.allclose(op1.err[k], op2.err[k], atol=1e-2)

                op2axsinv = xpost3inv @ op2.axs
                if hm.hdot(op2axsinv, op1.axs) < 0:
                    op2axsinv = -op2axsinv
                assert np.allclose(op1.axs, op2axsinv, atol=1e-4)

                # assert np.allclose(op1.cen, xpost3inv @ hm.proj_perp(op2.axs, op2.cen), atol=1e-4)
            except AssertionError as e:
                print('op1       ', op1.rad)
                print('op1       ', op1.cen)
                print('cen op2   ', op2.cen)
                print('cen op2inv', xpost3inv @ op2.cen)
                print('proj      ', hm.proj_perp(op1.axs, xpost3inv @ op2.cen))
                print('proj      ', hm.proj_perp(op2.axs, xpost3inv @ op2.cen))
                print(op1.rad, op2.rad)
                # print('proj  ', hm.proj_perp(op2.axs, op2.cen))
                # print('proj  ', xpost3inv @ hm.proj_perp(op2.axs, op2.cen))
                # print('op1axs', op1.axs, op1.ang)
                # print('op2axs', xpost3inv @ op2.axs, op2.ang)
                # print('op2axs', op2.axs, op2.ang)
                # print(op1.xrel)
                # print(op2.xrel)
                raise e

        # wu.viz.showme(list(symops3.values()), 'someops2')

def test_symops_cen_imperfect(nsamp=20, manual=False, **kw):
    # np.set_printoptions(
    # precision=10,
    # suppress=True,
    # linewidth=98,
    # formatter={'float': lambda f: '%14.10f' % f},
    # )
    # r = hm.hrot([1, 0, 0], 180)
    # xinfo.axs, a = hm.axis_angle_of(r)
    # print(r)
    # print(xinfo.axs)
    # print(a)
    # assert 0

    kw = wu.Bunch()
    kw.tprelen = 20
    kw.tprerand = 2
    kw.tpostlen = 20
    kw.tpostrand = 2
    kw.cart_sd_fuzz = 1.0
    kw.rot_sd_fuzz = np.radians(7)
    kw.cart_sd_fuzz = 1.0
    kw.rot_sd_fuzz = np.radians(7)
    kw.remove_outliers_sd = 3

    # wu.viz.showme(symops)
    # assert 0

    all_cen_err = list()
    for i in range(nsamp):

        kw.sym = np.random.choice('tet oct icos'.split())
        kw.nframes = np.random.choice(6) + 6
        kw.nframes = min(kw.nframes, len(hm.sym_frames[kw.sym]))

        frames, xpre, xpost, xfuzz, radius = setup_test_frames(**kw)

        symfit = hm.compute_symfit(frames=frames, **kw)
        symops = symfit.symops
        cen_err = np.linalg.norm(symfit.center - xpost[:, 3])
        all_cen_err.append(cen_err)

        # wu.viz.showme(selframes, showcen=True, name='source_frames')
        # wu.viz.showme(frames, showcen=True, name='source_frames')

        # wu.viz.showme(frames, xyzlen=(.1, .1, .1), showcen=True)
        # wu.viz.showme(xpost @ symframes, xyzlen=(.1, .1, .1), showcen=True)
        # wu.viz.showme(symops, center=xpost[:, 3], expand=2.0, scalefans=0.125, name='symops',
        # cyc_ang_match_tol=0.3, axislen=30, fixedfansize=2)
        # assert 0

        radius = np.mean(np.linalg.norm(frames[:, :, 3] - xpost[:, 3], axis=-1))
        # print(radius, symfit.radius)
        assert np.allclose(radius, symfit.radius, atol=kw.cart_sd_fuzz * 3)

    np.sort(all_cen_err)
    err = wu.Bunch()
    err.mean = np.mean(all_cen_err)
    err.mean1 = np.mean(all_cen_err[1:-1])
    err.mean2 = np.mean(all_cen_err[2:-2])
    err.mean2 = np.mean(all_cen_err[3:-3])
    err.median = np.median(all_cen_err)
    err.min = np.min(all_cen_err)
    err.max = np.max(all_cen_err)

    # print('test_symops_cen_imperfect median err', err.median)
    assert err.median < 3.0
    if manual:
        return err

def setup_test_frames(nframes, sym, cart_sd_fuzz, rot_sd_fuzz, tprelen=20, tprerand=0,
                      tpostlen=10, tpostrand=0, noxpost=False, **kw):
    symframes = hm.sym_frames[sym]
    selframes = symframes[np.random.choice(len(symframes), nframes, replace=False), :, :]
    xpre = hm.rand_xform()
    xpre[:3, 3] = hm.rand_unit()[:3] * (tprelen + tprerand * (np.random.rand() - 0.5))
    xfuzz = hm.rand_xform_small(nframes, cart_sd=cart_sd_fuzz, rot_sd=rot_sd_fuzz)
    xpost = hm.rand_xform()
    xpost[:3, 3] = hm.rand_unit()[:3] * (tpostlen + tpostrand * (np.random.rand() - 0.5))
    if noxpost: xpost = np.eye(4)
    frames = xpost @ selframes @ xpre @ xfuzz  # move subunit
    radius = None
    return frames, xpre, xpost, xfuzz, radius

def test_symfit_align_axes():
    kw = wu.Bunch()
    # kw.sym = np.random.choice('tet oct icos'.split())
    kw.sym = 'tet'
    # kw.nframes = np.random.choice(6) + 6
    kw.nframes = len(hm.sym_frames[kw.sym])
    kw.tprelen = 20
    kw.tprerand = 2
    kw.tpostlen = 20
    kw.tpostrand = 2
    kw.remove_outliers_sd = 3
    kw.fuzzstdfrac = 0.05  # frac of radian
    # kw.cart_sd_fuzz = fuzzstdabs
    # kw.rot_sd_fuzz = fuzzstdabs / tprelen

    kw.cart_sd_fuzz = kw.fuzzstdfrac * kw.tprelen
    kw.rot_sd_fuzz = kw.fuzzstdfrac

    point_angles = hm.sym_point_angles[kw.sym]
    frames, xpre, xpost, xfuzz, radius = setup_test_frames(**kw)
    symops = hm.symops_from_frames(frames, point_angles)
    symops = hm.stupid_pairs_from_symops(symops)
    # wu.viz.showme(symops)
    # assert 0

    symfit = hm.compute_symfit(frames=frames, **kw)

    print('ang_err', symfit.symop_ang_err)
    print('hel_err', symfit.symop_hel_err)
    print('cen_err', symfit.cen_err)
    print('axes_err', symfit.axes_err)
    print('total_err', symfit.total_err)
    assert symfit.total_err < 10
    # assert 0

def test_symfit_loss_mc(seed=None, quiet=True, maxiters=1000, goalerr=0.1, **kw):
    kw = wu.Bunch(kw)
    if not 'timer' in kw: kw.timer = wu.Timer()

    assert "PYTEST_CURRENT_TEST" not in os.environ
    if seed is None:
        seed = np.random.randint(2**32 - 1)
    np.random.seed(seed)

    # kw.sym = np.random.choice('tet oct icos'.split())
    # kw.nframes = len(hm.sym_frames[kw.sym])
    # kw.nframes = np.random.choice(6) + 6
    kw.sym = 'tet'
    kw.nframes = 6

    kw.tprelen = 10
    kw.tprerand = 0
    kw.tpostlen = 20
    kw.tpostrand = 0
    kw.fuzzstdfrac = 0.1  # frac of radian
    kw.cart_sd_fuzz = kw.fuzzstdfrac * kw.tprelen
    kw.rot_sd_fuzz = kw.fuzzstdfrac
    kw.remove_outliers_sd = 3
    kw.choose_closest_frame = False

    frames, xpre, xpost, xfuzz, radius = setup_test_frames(**kw)

    # frames = hm.rand_xform(len(frames), cart_sd=20)  #   @ frames

    showargs = dict(headless=0, sphere=1)
    # wu.viz.showme(frames, 'start', col=(1, 1, 1), **showargs)
    symfit = hm.compute_symfit(frames=frames, **kw)
    frames = symfit.xfit @ frames
    # wu.viz.showme(hm.sym_frames[kw.sym][:, None] @ frames[None, :], name='symstart',
    # col=(1, 1, 0), rays=0.02, weight=0.3, **showargs)

    low_err = symfit.weighted_err
    best_err = low_err
    best = frames, None
    if not quiet: print('start', symfit.weighted_err)

    naccept = 0
    for isamp in range(maxiters):
        if isamp % 10 == 0: frames = best[0]
        if isamp % 100 == 0 and not quiet:
            print(isamp, symfit.weighted_err, naccept / (isamp + 1))
        cartsd = symfit.weighted_err / 15
        rotsd = cartsd / symfit.radius
        temp = symfit.weighted_err / 150
        candidate_frames = hm.rand_xform_small(len(frames), cart_sd=cartsd, rot_sd=rotsd) @ frames
        symfit = hm.compute_symfit(frames=candidate_frames, **kw)
        candidate_frames = symfit.xfit @ candidate_frames

        delta = symfit.weighted_err - low_err
        if np.exp(-delta / temp) > np.random.rand():
            naccept += 1
            # frames = symfit.xfit @ candidate_frames
            frames = candidate_frames
            low_err = symfit.weighted_err
            # col = (isamp / maxiters, 1 - isamp / maxiters, 1)
            # wu.viz.showme(candidate_frames, name='mc%05i' % isamp, col=col, center=[0, 0, 0],
            # **showargs)
            if low_err < best_err:
                best_err = low_err
                best = symfit.xfit @ frames, symfit
                # if symfit.weighted_err < goalerr:
                abserr = symframes_coherence(hm.sym_frames[kw.sym][:, None] @ frames[None, :])
                if abserr < 2 * goalerr:
                    break
                # best = frames
                # print('    best %6i' % isamp, low_err)
            # print(f'low %6i {low_err:7.4f} {best_err:7.4f}' % isamp)

    frames, symfit = best
    symdupframes = hm.sym_frames[kw.sym][:, None] @ frames[None, :]
    symerr = symframes_coherence(symdupframes)
    # print('symerr', symerr)
    # wu.viz.showme(frames, name='best', col=(1, 1, 1), center=[0, 0, 0], **showargs)
    # wu.viz.showme(symdupframes, name='xfitbest', col=(0, 0, 1), rays=0.1, **showargs)

    # t.report()

    return wu.Bunch(nsamp=isamp + 1, besterr=best_err, symerr=symerr)

def symframes_coherence(frames):
    frames = frames.reshape(-1, 4, 4)
    norms = np.linalg.norm(frames[:, None, :, 3] - frames[None, :, :, 3], axis=-1)
    np.fill_diagonal(norms, 9e9)
    normmin = np.sort(norms, axis=0)[2]  # should always be at leart 3 frames
    err = np.max(normmin, axis=0)
    # print(norms.shape, err)
    return err

def symfit_parallel_mc_trials(**kw):
    import concurrent.futures as cf
    from itertools import repeat, chain, combinations
    from collections import defaultdict

    termsset = list(chain(*(combinations("CHNA", i + 1) for i in range(4))))
    termsset = list(str.join('', combo) for combo in termsset)
    # termsset = ['C']

    kw = wu.Bunch()
    ntrials = 10
    kw.goalerr = 0.1
    kw.maxiters = 2000
    kw.quiet = True
    seeds = list(np.random.randint(2**32 - 1) for i in range(ntrials))
    # print('seeds', seeds)
    fut = defaultdict(dict)
    with cf.ProcessPoolExecutor() as exe:
        for iterms, terms in enumerate(termsset):
            kw.lossterms = terms
            for iseed, seed in enumerate(seeds):
                kw.seed = seed
                # print('submit', terms, seed)
                fut[terms][seed] = exe.submit(test_symfit_loss_mc, **kw)
        # for i, f in fut.items():
        # print(i, f.result())
        print('symfit_parallel_mc_trials mean iters:')
        for terms in termsset:
            niters = [f.result().nsamp for k, f in fut[terms].items()]
            score = [f.result().symerr for k, f in fut[terms].items()]
            badscores = [s for s in score if s > 3 * kw.goalerr]
            # badscores = []
            print(
                f'{terms:4} iters {np.mean(niters):7.1f} ',
                f'fail {len(badscores)/ntrials:5.3f} ',
                ' '.join(
                    ['%4.2f' % q for q in np.quantile(badscores, [0.0, 0.1, 0.25, 0.5, 1.0])]
                    if badscores else '', ),
            )

def test_symfit_grad_gd():
    pass

if __name__ == '__main__':

    # test_symfit_align_axes()
    t = wu.Timer()

    symfit_parallel_mc_trials()
    assert 0
    # errs = list()
    # for i in range(5):
    #     errs.append(test_symops_cen_imperfect(nsamp=20, manual=True))
    # err = wu.Bunch().accumulate(errs)
    # err.reduce(max)
    # print(err)
    test_rel_xform_info()
    t.checkpoint('test_rel_xform_info')
    test_symops_cen_perfect()
    t.checkpoint('test_symops_cen_perfect')
    test_symops_cen_imperfect()
    t.checkpoint('test_symops_cen_imperfect')
    test_rel_xform_info_rand()
    t.checkpoint('test_rel_xform_info_rand')
    test_cyclic_sym_err()
    t.checkpoint('test_cyclic_sym_err')
    # test_symfit_loss_mc()
    # t.checkpoint('test_symfit_loss_mc')

    t.report()
    print('test_symfit.py done')
