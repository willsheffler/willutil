import itertools as it
import numpy as np
import willutil as wu
from willutil import homog as hm, Bunch

class SymFitError(Exception):
    pass

def _checkpoint(kw, label):
    if 'timer' in kw: kw['timer'].checkpoint(label)

class RelXformInfo(Bunch):
    pass

class SymOps(Bunch):
    pass

class SymFit(Bunch):
    pass

def rel_xform_info(frame1, frame2, **kw):
    # rel = np.linalg.inv(frame1) @ frame2
    rel = frame2 @ np.linalg.inv(frame1)
    # rot = rel[:3, :3]
    # axs, ang = hm.axis_angle_of(rel)
    axs, ang, cen = hm.axis_ang_cen_of(rel)

    framecen = (frame2[:, 3] + frame1[:, 3]) / 2
    framecen = framecen - cen
    framecen = hm.proj(axs, framecen)
    framecen = framecen + cen

    inplane = hm.proj_perp(axs, cen - frame1[:, 3])
    # inplane2 = hm.proj_perp(axs, cen - frame2[:, 3])
    rad = np.sqrt(np.sum(inplane**2))
    if np.isnan(rad):
        print('isnan rad')
        print('xrel')
        print(rel)
        print('det', np.linalg.det(rel))
        print('axs ang', axs, ang)
        print('cen', cen)
        print('inplane', inplane)
        assert 0
    hel = np.sum(axs * rel[:, 3])
    return RelXformInfo(
        xrel=rel,
        axs=axs,
        ang=ang,
        cen=cen,
        rad=rad,
        hel=hel,
        framecen=framecen,
        frames=np.array([frame1, frame2]),
    )

def xform_update_symop(symop, xform, srad):
    frame1 = xform @ symop.frames[0]
    frame2 = xform @ symop.frames[1]
    result = rel_xform_info(frame1, frame2)
    for k, v in symop.items():
        if k not in result:
            result[k] = symop[k]
    scen = xform[:, 3]
    p = hm.proj(result.axs, -result.cen) + result.cen
    d = np.linalg.norm(p[:3])
    e = 0
    if d < srad:
        e = np.sqrt(srad**2 - d**2)
    result.closest_to_cen = p
    result.isect_sphere = p + result.axs * e
    # wu.viz.showme([p, np.array([0, 0, 0, 1]), result])
    # assert 0

    return result

def cyclic_sym_err(pair, angle):
    hel_err = pair.hel
    errrad = min(10000, max(pair.rad, 1.0))
    ang_err = errrad * (angle - pair.ang)
    err = np.sqrt(hel_err**2 + ang_err**2)
    return err

def symops_from_frames(*, sym, frames, **kw):
    kw = wu.Bunch(kw)
    assert len(frames) > 1
    assert frames.shape[-2:] == (4, 4)
    pairs = dict()
    pairlist = list()
    keys, frame1, frame2 = list(), list(), list()
    for i, f1 in enumerate(frames):
        for j in range(i + 1, len(frames)):
            f2 = frames[j]
            keys.append((i, j))
            frame1.append(f1)
            frame2.append(f2)

    frame1 = np.stack(frame1)
    frame2 = np.stack(frame2)
    xrel = frame2 @ np.linalg.inv(frame1)
    axs, ang, cen = hm.axis_ang_cen_of(xrel)
    framecen = (frame2[:, :, 3] + frame1[:, :, 3]) / 2 - cen
    framecen = hm.proj(axs, framecen) + cen
    inplane = hm.proj_perp(axs, cen - frame1[:, :, 3])
    rad = np.sqrt(np.sum(inplane**2, axis=-1))
    hel = np.sum(axs * xrel[:, :, 3], axis=-1)
    assert (len(frame1) == len(frame2) == len(xrel) == len(axs) == len(ang) == len(cen) ==
            len(framecen) == len(rad) == len(hel))
    errrad = np.minimum(10000, np.maximum(rad, 1.0))
    angdelta = dict()
    err = dict()
    point_angles = hm.sym_point_angles[sym]
    for n, tgtangs in point_angles.items():
        dabs = [np.abs(ang - atgt) for atgt in tgtangs]
        d = [atgt - ang for atgt in tgtangs]
        if len(dabs) == 1:
            dabs = dabs[0]
            d = d[0]
            atgt = tgtangs[0]
        elif len(dabs) == 2:
            d = np.where(dabs[0] < dabs[1], d[0], d[1])
            atgt = np.where(dabs[0] < dabs[1], tgtangs[0], tgtangs[1])
            dabs = np.where(dabs[0] < dabs[1], dabs[0], dabs[1])
        else:
            assert 0, 'too many point angle choices'
        ang_err = errrad * dabs
        err[n] = np.sqrt(hel**2 + ang_err**2)
        angdelta[n] = d

    errvals = np.stack(list(err.values()))
    w = np.argmin(errvals, axis=0)
    nfold = np.array(list(err.keys()))[w].astype('i4')
    angdelta = np.array([angdelta[nf][i] for i, nf in enumerate(nfold)])
    nfold_err = np.min(errvals, axis=0)

    # pair.nfold, pair.nfold_err = None, 9e9
    # for n, angs in point_angles.items():
    #     err = min(cyclic_sym_err(pair, a) for a in angs)
    #     pair.err[n] = err
    #     if err < pair.nfold_err:
    #         pair.nfold, pair.nfold_err = n, err
    #     # print('symops_from_frames', n, a, err)

    nfold = disambiguate_axes(sym, axs, nfold)

    result = SymOps(
        key=keys,
        frame1=frame1,
        frame2=frame2,
        xrel=xrel,
        axs=axs,
        ang=ang,
        cen=cen,
        rad=rad,
        hel=hel,
        framecen=framecen,
        nfold=nfold,
        nfold_err=nfold_err,
        angdelta=angdelta,
    )

    return result

def disambiguate_axes(sym, axis, nfold):
    if not sym in hm.ambuguous_axes: return nfold
    if not hm.ambuguous_axes[sym]: return nfold
    assert len(hm.ambuguous_axes[sym]) == 1

    nfold = nfold[:]
    ambignfold, maybenfold = hm.ambuguous_axes[sym][0]
    # print(ambignfold, maybenfold)
    ambigaxis = axis[nfold == ambignfold]
    maybeaxis = axis[nfold == maybenfold]
    # print('ambigaxis', ambigaxis.shape)
    # print('maybeaxis', maybeaxis.shape)
    # maybeaxis = (hm.sym_frames[sym][None, :] @ maybeaxis[:, None, :, None]).reshape(-1, 4)
    # print(maybeaxis.shape)

    nambig = len(ambigaxis)
    nmaybe = len(maybeaxis)
    # if nambig == 0:
    #     # tgtnum = int((nmaybe + nambig) * 2 / 3)
    #     nfold[np.random.rand(len(nfold)) < .25] = ambignfold
    #     return nfold
    # if nmaybe == 0:
    #     # tgtnum = int((nmaybe + nambig) * 2 / 3)
    #     nfold[np.random.rand(len(nfold)) < .25] = maybenfold
    #     return nfold

    dot = np.abs(hm.hdot(ambigaxis[None, :], maybeaxis[:, None]))
    try:
        maxdot = np.max(dot, axis=0)
        # print(maxdot.shape)
    except Exception as e:
        raise SymFitError(f'missing axes: {nfold}')

    # tgtnum = int((nmaybe + nambig) / 3)
    # idx = np.argsort(maxdot)[-tgtnum:]
    # maybe_so = np.zeros_like(maxdot, dtype='b')
    # for i in idx:
    # maybe_so[i] = True

    # print(nambig, nmaybe, dot.shape, maxdot.shape)
    # print(maxdot)
    # assert 0
    # print(nfold)
    maybe_so = maxdot > np.cos(np.pi / 12)  # theoretically pi/8 ro 22.5 deg

    nfold[nfold == ambignfold] = np.where(maybe_so, maybenfold, ambignfold)

    # assert 0

    return nfold

def stupid_pairs_from_symops(symops):
    # assert 0, 'no more stupid_pairs_from_symops'
    pairs = dict()
    for i, k in enumerate(symops.key):
        pairs[k] = RelXformInfo(
            xrel=symops.xrel[i],
            axs=symops.axs[i],
            ang=symops.ang[i],
            cen=symops.cen[i],
            rad=symops.rad[i],
            hel=symops.hel[i],
            framecen=symops.framecen[i],
            frames=np.array([symops.frame1[i], symops.frame2[i]]),
            nfold=symops.nfold[i],
            nfold_err=symops.nfold_err[i],
        )
    return pairs

def compute_symfit(
    *,
    sym,
    frames,
    max_nan=0.9,  # totally arbitrary, downstream check for lacking info maybe better
    remove_outliers_sd=3,
    # lossterms='CHNA',
    lossterms=None,
    **kw,
):
    kw = wu.Bunch(kw)
    point_angles = hm.sym_point_angles[sym]
    minsymang = dict(
        tet=hm.angle(hm.tetrahedral_axes[2], hm.tetrahedral_axes[3]) / 2,
        oct=hm.angle(hm.octahedral_axes[2], hm.octahedral_axes[3]) / 2,
        icos=hm.angle(hm.icosahedral_axes[2], hm.icosahedral_axes[3]) / 2,
        d3=np.pi / 6,
        d5=np.pi / 10,
    )
    symops = hm.symops_from_frames(sym=sym, frames=frames, **kw)
    # symops = stupid_pairs_from_symops(symops)
    # if len'nfolds',(symops) <= len(hm.symaxes[sym]):
    # raise SymFitError('not enough symops/monomers')
    # symops = None
    _checkpoint(kw, 'symops_from_frames')
    # cen1, cen2, opaxs1, opaxs2 = list(n), list(), list(), list()
    #cen1A, cen2A, axs1A, axs2A = list(), list(), list(), list()
    #for iop1, op1 in enumerate(symops.values()):
    #    for iop2, op2 in enumerate(symops.values()):
    #        if iop1 < iop2:
    #            cen1A.append(op1.cen)
    #            cen2A.append(op2.cen)
    #            axs1A.append(op1.axs)
    #            axs2A.append(op2.axs)
    #cen1A = np.stack(cen1A)
    # cen2A = np.stack(cen2A)
    # axs1A = np.stack(axs1A)
    # axs2A = np.stack(axs2A)
    # cen1A, cen2A, axs1A, axs2A = list(), list(), list(), list()
    # for op1, op2 in it.combinations(symops.values(), 2):
    #     if op1 is op2: continue
    #     cen1A.append(op1.cen)
    #     cen2A.append(op2.cen)
    #     axs1A.append(op1.axs)
    #     axs2A.append(op2.axs)
    # cen1A = np.stack(cen1A)
    # cen2A = np.stack(cen2A)
    # axs1A = np.stack(axs1A)
    # axs2A = np.stack(axs2A)

    cen1, cen2, axs1, axs2 = list(), list(), list(), list()
    nops = len(symops.axs)
    # assert nops == len(symops)
    for i in range(nops):
        cen1.append(np.tile(symops.cen[i], nops - i - 1).reshape(-1, 4))
        axs1.append(np.tile(symops.axs[i], nops - i - 1).reshape(-1, 4))
        cen2.append(symops.cen[i + 1:])
        axs2.append(symops.axs[i + 1:])
        # print(cen1[-1].shape)
        # print(cen2[-1].shape)
        assert cen1[-1].shape == cen2[-1].shape
    cen1 = np.concatenate(cen1)
    cen2 = np.concatenate(cen2)
    axs1 = np.concatenate(axs1)
    axs2 = np.concatenate(axs2)
    # assert cen1.shape == cen1A.shape
    # assert cen2.shape == cen2A.shape
    # assert np.allclose(cen1, cen1A)
    # assert np.allclose(cen2, cen2A)
    # assert np.allclose(axs1, axs1A)
    # assert np.allclose(axs2, axs2A)

    # cen1, cen2, axs1, axs2 = cen1A, cen2A, axs1A, axs2A
    # assert 0

    not_same_symaxis = hm.line_angle(axs1, axs2) > minsymang[sym]
    p1np = cen1[not_same_symaxis]
    p2np = cen2[not_same_symaxis]
    a1np = axs1[not_same_symaxis]
    a2np = axs2[not_same_symaxis]
    _checkpoint(kw, 'make symop pair arrays')
    # print('cen1', cen1.shape, 'isnan', np.sum(np.isnan(cen1)))
    # print('cen2', cen2.shape, 'isnan', np.sum(np.isnan(cen2)))
    # print('axs1', axs1.shape, 'isnan', np.sum(np.isnan(axs1)))
    # print('axs2', axs2.shape, 'isnan', np.sum(np.isnan(axs2)))

    p, q = hm.line_line_closest_points_pa(p1np, a1np, p2np, a2np)
    _checkpoint(kw, 'compute symax intersects')
    # print('p', p.shape, 'isnan', np.sum(np.isnan(p)))
    # print('q', q.shape, 'isnan', np.sum(np.isnan(q)))
    tot_nan = np.sum(np.isnan(p)) / 4
    if tot_nan / len(p) > max_nan:
        print('nfolds', symops.nfold)
        raise SymFitError(
            f'{tot_nan/len(p)*100:7.3f}% of symops are parallel or cant be intersected')

    p = p[~np.isnan(p)].reshape(-1, 4)
    q = q[~np.isnan(q)].reshape(-1, 4)
    isect = (p + q) / 2

    # print(hm.angle_degrees(axs1, axs2)[:10])
    # print('p', p[:10])
    # print('q', q[:10])
    # print('isect', isect[:10])
    # print(p.shape, q.shape)

    center = np.mean(isect, axis=0)
    # print(center.shape, center)

    if remove_outliers_sd is not None:
        norm = np.linalg.norm(p - center, axis=-1)
        meannorm = np.mean(norm)
        sdnorm = np.std(norm)
        not_outlier = norm - meannorm < sdnorm * remove_outliers_sd
        # print('norm', norm.shape, np.mean(norm), np.min(norm), np.max(norm), np.sum(not_outlier),
        # np.sum(not_outlier) / len(not_outlier))
        # print(center)
        center = np.mean(isect[not_outlier], axis=0)
        # print(center)

    radii = np.linalg.norm(frames[:, :, 3] - center, axis=-1)
    radius = np.mean(radii)

    # rad_err = np.sqrt(np.mean(radii**2))
    # rad_err = np.sqrt(np.mean(radii**2) / radius)
    rad_err = np.sqrt(np.mean(radii**2)) / radius

    cen_err = np.sqrt((np.sum((center - p)**2) + np.sum((center - q)**2)) / (len(q) + len(q)))

    op_hel_err = np.sqrt(np.mean(symops.hel**2))
    op_ang_err = np.sqrt(np.mean(symops.nfold_err**2))
    _checkpoint(kw, 'post intersect stuff')

    xfit, axesfiterr = hm.symops_align_axes(sym, frames, symops, symops, center, radius, **kw)
    _checkpoint(kw, 'align axes')

    loss = dict()
    loss['C'] = 1.0 * cen_err**2
    loss['H'] = 0.7 * op_hel_err**2
    loss['N'] = 1.2 * op_ang_err**2
    loss['A'] = 1.5 * axesfiterr**2
    # loss['R'] = 1.0 * rad_err**2
    # loss['Q'] = 1.0 * quad_err**2
    # loss['M'] = 1.0 * missing_axs_err**2
    # loss['S'] = 1.0 * skew_axs_err**2
    total_err = np.sqrt(np.sum(list(loss.values())))
    weighted_err = total_err
    if lossterms:
        weighted_err = np.sqrt(sum(loss[c] for c in lossterms))

    # CA   iters   861.7  fail 0.000
    # CHA  iters   884.7  fail 0.000
    # CHNA iters   470.2  fail 0.000
    # C    iters  2000.0  fail 1.000  1.56 2.38 3.17 3.92 5.96
    # H    iters  2000.0  fail 1.000  1.22 2.57 3.14 4.04 6.29
    # A    iters  2000.0  fail 1.000  2.23 2.73 3.16 4.13 8.09
    # CH   iters  2000.0  fail 1.000  1.71 2.51 3.44 4.10 6.58
    # CN   iters  1614.5  fail 0.810  0.30 0.53 1.60 3.46 5.56
    # CHN  iters  1790.7  fail 0.890  0.30 0.54 1.71 4.01 6.61
    # HN   iters  1633.5  fail 0.760  0.31 0.56 1.28 4.12 6.35
    # N    iters  1590.0  fail 0.570  0.33 0.87 1.84 3.68 6.56

    # A7 N4 H4 C4
    # NA   iters  1251.3  fail 0.040  0.31 0.96 1.95 3.00 4.05
    # HA   iters  1131.2  fail 0.020  3.67 3.67 3.68 3.69 3.72
    # HNA  iters   830.2  fail 0.020  2.55 2.69 2.90 3.25 3.95
    # CA   iters   863.9  fail 0.020  0.96 0.97 0.99 1.02 1.09
    # CHA  iters   798.1  fail 0.020  0.32 0.37 0.44 0.57 0.81
    # CHNA iters   480.1  fail 0.020  0.31 0.31 0.31 0.32 0.33
    # CNA  iters   512.2  fail 0.000

    return SymFit(
        sym=sym,
        nframes=len(frames),
        frames=frames,
        symops=symops,
        center=center,
        opcen1=cen1,
        opcen2=cen2,
        opaxs1=axs1,
        opaxs2=axs2,
        iscet=isect,
        isect1=p,
        iscet2=q,
        radius=radius,
        xfit=xfit,
        cen_err=cen_err,
        symop_hel_err=op_hel_err,
        symop_ang_err=op_ang_err,
        axes_err=axesfiterr,
        total_err=total_err,
        weighted_err=weighted_err,
    )

def best_axes_fit(sym, xsamp, nfolds, tgtaxes, tofitaxes, **kw):
    xsamp = xsamp[:, None]
    randtgtaxes = [(xsamp @ ax.reshape(1, -1, 4, 1)).squeeze(-1) for ax in tgtaxes]

    err = list()
    for i, (nf, tgt, fit) in enumerate(zip(nfolds, randtgtaxes, tofitaxes)):
        n = np.newaxis
        dotall = hm.hdot(fit[n, n, :], tgt[:, :, n])
        if sym != 'tet' or nf != 2:
            dotall = np.abs(dotall)
        # angall = np.arccos(dotall)
        # angmatch = np.min(angall, axis=1)
        # angerr = np.mean(angmatch**2, axis=1)
        dotmatch = np.max(dotall, axis=1)
        angerr = np.mean(np.arccos(dotmatch)**2, axis=1)
        err.append(angerr)
    err = np.mean(np.stack(err), axis=0)
    bestx = xsamp[np.argmin(err)].squeeze()
    err = np.sqrt(np.min(err))
    return bestx, err

def symops_align_axes(
    sym,
    frames,
    opary,
    symops,
    center,
    radius,
    choose_closest_frame=False,
    align_ang_delta_thresh=0.001,
    **kw,
):

    nfolds = list(hm.symaxes[sym].keys())
    if 7 in nfolds: nfolds.remove(7)  # what to do about T33?
    pang = hm.sym_point_angles[sym]
    # xtocen = np.eye(4)
    # xtocen[:, 3] = -center

    _checkpoint(kw, 'symops_align_axes xtocen')
    # recenter frames without modification
    # for k in symops:
    # symops[k] = xform_update_symop(symops[k], xtocen, radius)
    # fitaxis = hm.hnormalized(symops[k].framecen)
    # fitaxis = hm.hnormalized(symops[k].isect_sphere)
    # fitaxis = hm.hnormalized(symops[k].axs)
    # symops[k].fitaxis = fitaxis

    # fitaxis = hm.hnormalized(symops[k].framecen)
    # fitaxis = hm.hnormalized(symops[k].isect_sphere)
    # opary.fitaxis = opary.isect_sphere
    # opary.fitaxis = opary.framecen
    opary.fitaxis = opary.axs
    # for i, a in enumerate(opary.fitaxis):
    #     cyccen = opary.framecen - center
    #     if np.sum(cyccen * a) < 0:
    #         opary.fitaxis[i] = -a

    # wu.viz.showme(symops)
    _checkpoint(kw, 'symops_align_axes xform xform_update_symop')

    # nfold_axes = [{k: v for k, v in symops.items() if v.nfold == nf} for nf in nfolds]
    # allopaxes = np.array([op.fitaxis for op in symops.values()])
    # opnfold = np.array([op.nfold for op in symops.values()], dtype='i4')
    # sopaxes = [allopaxes[opnfold == nf] for nf in nfolds]
    # tgtaxes = [hm.symaxes_all[sym][nf] for nf in nfolds]
    allopaxes = opary.fitaxis
    opnfold = opary.nfold
    sopaxes = [allopaxes[opnfold == nf] for nf in nfolds]
    tgtaxes = [hm.symaxes_all[sym][nf] for nf in nfolds]

    _checkpoint(kw, 'symops_align_axes build arrays')
    # axes = symin
    # print('origtgtaxes', [a.shape for a in origtgtaxes])
    # print('tgtdaxes   ', [a.shape for a in tgtdaxes])
    # print('sopaxes     ', [a.shape for a in sopaxes])
    nsamp = 20
    xsamp = hm.rand_xform(nsamp, cart_sd=0)
    xfit, angerr = best_axes_fit(sym, xsamp, nfolds, tgtaxes, sopaxes)
    best = 9e9, np.eye(4)
    for i in range(20):
        _checkpoint(kw, 'symops_align_axes make xsamp pre')
        xsamp = hm.rand_xform_small(nsamp, rot_sd=angerr / 2, cart_sd=0) @ xfit
        _checkpoint(kw, 'symops_align_axes make xsamp')
        xfit, angerr = best_axes_fit(sym, xsamp, nfolds, tgtaxes, sopaxes, **kw)
        _checkpoint(kw, 'symops_align_axes best_axes_fit')
        delta = angerr - best[0]
        if delta < 0:
            best = angerr, xfit
            if delta > -align_ang_delta_thresh:
                # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                break
        xfit = best[1]
        # if i % 1 == 0: print(angerr)
    angerr, xfit = best
    _checkpoint(kw, 'symops_align_axes check rand axes')

    xfit[:, 3] = center
    xfit = np.linalg.inv(xfit)
    axesfiterr = angerr * radius

    if choose_closest_frame:
        which_frame = np.argmin(np.sum((hm.sym_frames[sym] @ xfit - np.eye(4))**2, axis=(-1, -2)))
        xfit = hm.sym_frames[sym][which_frame] @ xfit
        # print(which_frame)

    # if sym == 'tet':
    #     fit = xfit @ frames
    #     cens = hm.hnormalized(fit[:, :, 3])
    #     upper = np.any(hm.angle(cens, [1, 1, 1]) > 1.91 / 2)
    #     if not upper:
    #         # rotate 3fold around -1 -1 -1 to 3fold around 1 1 1
    #         # this prevents mismatch with canonical tetrahedral 3fold position
    #         # tetframes @ frames can form octahedra
    #         xfit = hm.hrot([1, 1, -1], np.pi * 2 / 3) @ xfit
    #         fit = xfit @ frames
    #         cens = hm.hnormalized(fit[:, :, 3])
    #         upper = np.any(hm.angle(cens, [1, 1, 1]) > 1.91 / 2)
    #         # upper = np.any(np.all(cens > 0.0, axis=-1))
    #         if not upper:
    #             xfit = hm.hrot([1, 1, -1], np.pi * 2 / 3) @ xfit
    #     # [1, 1, 1],
    #     # [m, 1, 1],
    #     # [1, m, 1],
    #     # [1, 1, m],

    #     # lower = np.any(np.all(cens < +0.1, axis=-1))
    #     # if upper and lower:
    #     # assert 0
    #     # axesfiterr = 9e9
    #     # assert 0, 'can this happen and be reasonable?'

    # print(xfit.shape, axesfiterr * 180 / np.pi)

    # for i, ax in enumerate(sopaxes):
    #     col = [0, 0, 0]
    #     col[i] = 1
    #     wu.viz.showme(ax, col=col, usefitaxis=True, name='sopA')
    #     # wu.viz.showme(tgtaxes[i], col=col, usefitaxis=True, name='tgtA')
    #     ax = hm.hxform(xfit, ax)
    #     wu.viz.showme(ax, col=col, usefitaxis=True, name='nfoldB')
    return xfit, axesfiterr

def symfit_gradient(symfit):
    # sym=sym,
    # frames=frames,
    # symops=symops,
    # center=center,
    # opcen1=cen1,
    # opcen2=cen2,
    # opaxs1=axs1,
    # opaxs2=axs2,
    # iscet=isect,
    # isect1=p,
    # iscet2=q,
    # radius=radius,
    # xfit=xfit,
    # cen_err=cen_err,
    # symop_hel_err=op_hel_err,
    # symop_ang_err=op_ang_err,
    # axes_err=axesfiterr,
    # total_err=total_err,
    # weighted_err=weighted_err,

    # result = SymOps(
    #     key=keys,
    #     frame1=frame1,
    #     frame2=frame2,
    #     xrel=xrel,
    #     axs=axs,
    #     ang=ang,
    #     cen=cen,
    #     rad=rad,
    #     hel=hel,
    #     framecen=framecen,
    #     nfold=nfold,
    #     nfold_err=nfold_err,
    sop = symfit.symops

    print(list(symfit.symops.keys()))
    for i in range(len(sop.key)):
        print(
            sop.nfold[i],
            np.round(np.degrees(sop.ang[i] + sop.angdelta[i])),
        )
    cenforce = np.zeros
    frametorq = np.zeros(shape=(symfit.nframes, 4))

    for key, torq in zip(sop.key, optorq):
        print(key, torq)
        frametorq[key[0]] -= torq
        frametorq[key[1]] += torq

    frameforce = np.zeros((symfit.nframes, 4))
    for key, force in zip(sop.key, sop.hel):
        frameforce[key[0]] += force
        frameforce[key[1]] -= force
    # SHOULD I ADD CART ROTATION FORCE HERE TOO?????

    # wu.viz.showme(sy,mfit.symops)

    # assert 0
