import itertools as it
import numpy as np
import willutil as wu
from willutil import homog as hm, Bunch

class SymFItError(Exception):
    pass

def _checkpoint(kw, label):
    if 'timer' in kw: kw['timer'].checkpoint(label)

class RelXformInfo(Bunch):
    pass

class SymOpsInfo(Bunch):
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

def symops_from_frames(frames, point_angles, **kw):
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
    err = dict()
    for n, angs in point_angles.items():
        d = [np.abs(a - ang) for a in angs]
        if len(d) == 1: d = d[0]
        elif len(d) == 2: d = np.where(d[0] < d[1], d[0], d[1])
        else: assert 0
        ang_err = errrad * d
        err[n] = np.sqrt(hel**2 + ang_err**2)
    errvals = np.stack(list(err.values()))
    w = np.argmin(errvals, axis=0)
    best_nfold = np.array(list(err.keys()))[w].astype('i4')
    best_nfold_err = np.min(errvals, axis=0)
    # pair.best_nfold, pair.best_nfold_err = None, 9e9
    # for n, angs in point_angles.items():
    #     err = min(cyclic_sym_err(pair, a) for a in angs)
    #     pair.err[n] = err
    #     if err < pair.best_nfold_err:
    #         pair.best_nfold, pair.best_nfold_err = n, err
    #     # print('symops_from_frames', n, a, err)

    result = SymOpsInfo(
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
        best_nfold=best_nfold,
        best_nfold_err=best_nfold_err,
    )

    return result
    # return stupid_pairs_from_symops(result)s

    #  pairs = dict()
    #  for i, k in enumerate(keys):
    # pairs[k] = RelXformInfo(
    # xrel=xrel[i],
    # axs=axs[i],
    # ang=ang[i],
    # cen=cen[i],
    # rad=rad[i],
    # hel=hel[i],
    # framecen=framecen[i],
    # frames=np.array([frame1[i], frame2[i]]),
    # best_nfold=best_nfold[i],
    # best_nfold_err=best_nfold_err[i],
    # )

    # for i, pair in enumerate(pairlist):
    # assert np.allclose(pair.frames[0], frame1[i])
    # assert np.allclose(pair.frames[1], frame2[i])
    # assert np.allclose(pair.xrel, xrel[i])
    # assert np.allclose(pair.axs, axs[i])
    # assert np.allclose(pair.ang, ang[i])
    # assert np.allclose(pair.cen, cen[i])
    # assert np.allclose(pair.framecen, framecen[i])
    # assert np.allclose(pair.rad, rad[i])
    # assert np.allclose(pair.hel, hel[i])

    # return pairs

def stupid_pairs_from_symops(symopinfo):
    # assert 0, 'no more stupid_pairs_from_symops'
    pairs = dict()
    for i, k in enumerate(symopinfo.key):
        pairs[k] = RelXformInfo(
            xrel=symopinfo.xrel[i],
            axs=symopinfo.axs[i],
            ang=symopinfo.ang[i],
            cen=symopinfo.cen[i],
            rad=symopinfo.rad[i],
            hel=symopinfo.hel[i],
            framecen=symopinfo.framecen[i],
            frames=np.array([symopinfo.frame1[i], symopinfo.frame2[i]]),
            best_nfold=symopinfo.best_nfold[i],
            best_nfold_err=symopinfo.best_nfold_err[i],
        )
    return pairs

def compute_symfit(
    sym,
    frames,
    max_nan=0.333,  # totally arbitrary, downstream check for lacking info maybe better
    remove_outliers_sd=3,
    **kw,
):
    kw = wu.Bunch(kw)
    point_angles = hm.sym_point_angles[sym]
    minsymang = dict(
        tet=hm.angle(hm.tetrahedral_axes[2], hm.tetrahedral_axes[3]) / 2,
        oct=hm.angle(hm.octahedral_axes[2], hm.octahedral_axes[3]) / 2,
        icos=hm.angle(hm.icosahedral_axes[2], hm.icosahedral_axes[3]) / 2,
    )

    symops_ary = hm.symops_from_frames(frames, point_angles, **kw)
    # symops = stupid_pairs_from_symops(symops_ary)
    # if len(symops) <= len(hm.symaxes[sym]):
    # raise SymFItError('not enough symops/monomers')
    symops = None
    _checkpoint(kw, 'symops_from_frames')
    # cen1, cen2, axis1, axis2 = list(n), list(), list(), list()
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
    nops = len(symops_ary.axs)
    # assert nops == len(symops)
    for i in range(nops):
        cen1.append(np.tile(symops_ary.cen[i], nops - i - 1).reshape(-1, 4))
        axs1.append(np.tile(symops_ary.axs[i], nops - i - 1).reshape(-1, 4))
        cen2.append(symops_ary.cen[i + 1:])
        axs2.append(symops_ary.axs[i + 1:])
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
        raise SymFItError(
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

    radius = np.mean(np.linalg.norm(frames[:, :, 3] - center, axis=-1))
    cen_err = np.sqrt((np.sum((center - p)**2) + np.sum((center - q)**2)) / (len(q) + len(q)))

    op_hel_err = np.sqrt(np.mean(symops_ary.hel**2))
    op_ang_err = np.sqrt(np.mean(symops_ary.best_nfold_err**2))
    _checkpoint(kw, 'post intersect stuff')

    xfit, axesfiterr = hm.symops_align_axes(sym, symops_ary, symops, center, radius, **kw)
    _checkpoint(kw, 'align axes')

    C = cen_err**2
    H = op_hel_err**2
    N = op_ang_err**2
    A = axesfiterr**2

    err = np.sqrt(C + H + N + A)

    # err = np.sqrt(H + N + A)
    # err = np.sqrt(C + N + A)
    # err = np.sqrt(C + H + A)
    # err = np.sqrt(C + H + N)

    # err = np.sqrt(H + N) # bad
    # err = np.sqrt(C + A)

    # err = np.sqrt(C) # bad
    # err = np.sqrt(H) # bad
    # err = np.sqrt(N)  # bad
    # err = np.sqrt(A) # bad

    return SymOpsInfo(
        sym=sym,
        symops=symops,
        center=center,
        opcen1=cen1,
        opcen2=cen2,
        axis1=axs1,
        axis2=axs2,
        iscet=isect,
        isect1=p,
        iscet2=q,
        radius=radius,
        xfit=xfit,
        cen_err=cen_err,
        symop_hel_err=op_hel_err,
        symop_ang_err=op_ang_err,
        axes_err=axesfiterr,
        total_err=err,
    )

def best_axes_fit(xsamp, nfolds, tgtaxes, tofitaxes, **kw):
    xsamp = xsamp[:, None]
    randtgtaxes = [(xsamp @ ax.reshape(1, -1, 4, 1)).squeeze(-1) for ax in tgtaxes]

    err = list()
    for i, (nf, tgt, fit) in enumerate(zip(nfolds, randtgtaxes, tofitaxes)):
        n = np.newaxis
        dotall = np.abs(hm.hdot(fit[n, n, :], tgt[:, :, n]))
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

def symops_align_axes(sym, opary, symops, center, radius, choose_closest_frame=False,
                      align_ang_delta_thresh=0.001, **kw):

    nfolds = list(hm.symaxes[sym].keys())
    if 7 in nfolds: nfolds.remove(7)  # what to do about T33?
    pang = hm.sym_point_angles[sym]
    xtocen = np.eye(4)
    xtocen[:, 3] = -center

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

    # wu.viz.showme(symops)
    _checkpoint(kw, 'symops_align_axes xform xform_update_symop')

    # nfold_axes = [{k: v for k, v in symops.items() if v.best_nfold == nf} for nf in nfolds]
    # allopaxes = np.array([op.fitaxis for op in symops.values()])
    # opnfold = np.array([op.best_nfold for op in symops.values()], dtype='i4')
    # sopaxes = [allopaxes[opnfold == nf] for nf in nfolds]
    # tgtaxes = [hm.symaxes_all[sym][nf] for nf in nfolds]
    allopaxes = opary.fitaxis
    opnfold = opary.best_nfold
    sopaxes = [allopaxes[opnfold == nf] for nf in nfolds]
    tgtaxes = [hm.symaxes_all[sym][nf] for nf in nfolds]

    _checkpoint(kw, 'symops_align_axes build arrays')
    # axes = symin
    # print('origtgtaxes', [a.shape for a in origtgtaxes])
    # print('tgtdaxes   ', [a.shape for a in tgtdaxes])
    # print('sopaxes     ', [a.shape for a in sopaxes])
    nsamp = 20
    xsamp = hm.rand_xform(nsamp, cart_sd=0)
    xfit, angerr = best_axes_fit(xsamp, nfolds, tgtaxes, sopaxes)
    best = 9e9, np.eye(4)
    for i in range(20):
        _checkpoint(kw, 'symops_align_axes make xsamp pre')
        xsamp = hm.rand_xform_small(nsamp, rot_sd=angerr / 2, cart_sd=0) @ xfit
        _checkpoint(kw, 'symops_align_axes make xsamp')
        xfit, angerr = best_axes_fit(xsamp, nfolds, tgtaxes, sopaxes, **kw)
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

    xfit[:, 3] = -center
    xfit = np.linalg.inv(xfit)
    axesfiterr = angerr * radius

    if choose_closest_frame:
        which_frame = np.argmin(np.sum((hm.sym_frames[sym] @ xfit - np.eye(4))**2, axis=(-1, -2)))
        xfit = hm.sym_frames[sym][which_frame] @ xfit
        # print(which_frame)

    # print(xfit.shape, axesfiterr * 180 / np.pi)

    # for i, ax in enumerate(sopaxes):
    #     col = [0, 0, 0]
    #     col[i] = 1
    #     wu.viz.showme(ax, col=col, usefitaxis=True, name='sopA')
    #     # wu.viz.showme(tgtaxes[i], col=col, usefitaxis=True, name='tgtA')
    #     ax = hm.hxform(xfit, ax)
    #     wu.viz.showme(ax, col=col, usefitaxis=True, name='nfoldB')
    return xfit, axesfiterr

# def symfit(frames, point_ang, symangles):
# pairs = symops_from_frames()
