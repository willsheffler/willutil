import sys, os, tempfile, numpy as np, time
from collections import defaultdict
import willutil as wu
from logging import info
from functools import singledispatch

from deferred_import import deferred_import

# pymol = deferred_import('pymol')
# cgo = deferred_import('pymol.cgo')
# cmd = deferred_import('pymol.cmd')
import pymol
from pymol import cgo, cmd
from willutil import homog as hm
from willutil.viz.pymol_cgo import *

try:
    from pyrosetta.rosetta.core.pose import Pose
except ImportError:

    class DummyPose:
        pass

    Pose = DummyPose

@singledispatch
def pymol_load(toshow, state=None, name=None, **kw):
    raise NotImplementedError("pymol_load: don't know how to show " + str(type(toshow)))

_nsymops = 0

@pymol_load.register(wu.homog.RelXformInfo)
def _(
    toshow,
    state,
    col='bycx',
    name='xrel',
    showframes=True,
    center=np.array([0, 0, 0, 1]),
    scalefans=0.25,
    fixedfansize=None,
    expand=1.0,
    fuzz=0,
    make_cgo_only=False,
    cyc_ang_match_tol=0.1,
    axislen=10,
    **kw,
):
    global _nsymops
    _nsymops += 1

    ang = toshow.ang
    if np.isclose(ang, np.pi * 4 / 5, atol=1e-4):
        ang /= 2
    if col == 'bycx':
        if np.isclose(ang, np.pi * 2 / 2, atol=cyc_ang_match_tol): col = [1, 1, 0]
        elif np.isclose(ang, np.pi * 2 / 3, atol=cyc_ang_match_tol): col = [0, 1, 1]
        elif np.isclose(ang, np.pi * 2 / 4, atol=cyc_ang_match_tol): col = [1, 0, 1]
        elif np.isclose(ang, np.pi * 2 / 5, atol=cyc_ang_match_tol): col = [1, 0, 1]
        elif np.isclose(ang, np.pi * 2 / 6, atol=cyc_ang_match_tol): col = [1, 0, 1]
        else: col = [1, 1, 1]
    elif col == 'random':
        # col = np.random.rand(3) / 2 + 0.5
        col = (1, 1, 1)
    # cen = (toshow.framecen - center) * expand + center
    cen = toshow.framecen

    mycgo = list()

    if showframes:
        state, cgo = pymol_visualize_xforms(toshow.frames, state, make_cgo_only=True, **kw)
        mycgo += cgo

    cen1 = toshow.frames[0, :, 3]
    cen2 = toshow.frames[1, :, 3]

    if abs(toshow.ang) < 1e-6:
        mycgo += cgo_cyl(cen1, cen2, 0.01, col=(1, 1, 1))
        mycgo += cgo_sphere(cen=(cen1 + cen2) / 2, rad=0.1, col=(1, 1, 1))
    else:
        mycgo += cgo_cyl(
            cen + toshow.axs * axislen / 2,
            cen - toshow.axs * axislen / 2,
            0.03,
            col=col,
        )
        mycgo += cgo_cyl(
            cen + toshow.axs * toshow.hel / 2,
            cen - toshow.axs * toshow.hel / 2,
            0.15,
            col=col,
        )
        shift = fuzz * (np.random.rand() - 0.5)
        mycgo += cgo_fan(
            toshow.axs,
            cen + toshow.axs * shift,
            fixedfansize if fixedfansize else toshow.rad * scalefans,
            arc=ang,
            col=col,
            startpoint=cen1,
        )

    if make_cgo_only:
        return state, mycgo
    pymol.cmd.load_cgo(mycgo, 'symops%i' % _nsymops)
    return state

@pymol_load.register(Pose)
def _(toshow, state=None, name=None, **kw):
    name = name or "rif_thing"
    state["seenit"][name] += 1
    name += "_%i" % state["seenit"][name]
    pymol_load_pose(toshow, name, **kw)
    state["last_obj"] = name
    return state

@pymol_load.register(dict)
def _(toshow, state=None, name='stuff_in_dict', **kw):
    if "pose" in toshow:
        state = pymol_load(toshow["pose"], state, **kw)
        pymol_xform(toshow["position"], state["last_obj"])
    else:
        mycgo = list()
        for k, v in toshow.items():
            state, cgo = pymol_load(v, state, name=k, make_cgo_only=True, **kw)
            mycgo += cgo
        pymol.cmd.load_cgo(mycgo, name)
    return state

@pymol_load.register(list)
def _(toshow, state=None, name=None, **kw):
    for t in toshow:
        print('    ##############', type(t), '################')
        state = pymol_load(t, state, **kw)
    return state

@pymol_load.register(np.ndarray)
def _(toshow, state, **kw):
    # showaxes()
    shape = toshow.shape
    if shape[-2:] == (3, 4):
        return show_ndarray_n_ca_c(toshow, state, **kw)
    elif shape[-2:] == (4, 2):
        return show_ndarray_lines(toshow, state, **kw)
    elif len(shape) > 2 and shape[-2:] == (4, 4):
        return pymol_visualize_xforms(toshow, state, **kw)
    elif shape == (4, ) or len(shape) == 2 and shape[-1] == 4:
        return show_ndarray_point_or_vec(toshow, state, **kw)
    else:
        raise NotImplementedError

_nxforms = 0

def get_different_colors(ncol, niter=100):
    maxmincoldis, best = 0, None
    for i in range(niter):
        colors = np.random.rand(ncol, 3)
        cdis2 = np.linalg.norm(colors[None] - colors[:, None], axis=-1)
        np.fill_diagonal(cdis2, 4.0)
        if np.min(cdis2) > maxmincoldis:
            maxmincoldis, best = np.min(cdis2), colors
    return best

def pymol_visualize_xforms(
    xforms,
    state,
    name=None,
    randpos=0.0,
    xyzlen=[5 / 4, 1, 4 / 5],
    scale=1.0,
    weight=1.0,
    showcen=False,
    make_cgo_only=False,
    **kw,
):
    global _nxforms
    _nxforms += 1

    if xforms.shape == (4, 4):
        xforms = xforms.reshape(1, 4, 4)
    name = name or "xforms"
    state["seenit"][name] += 1
    name += "_%i" % state["seenit"][name]
    # mycgo = [cgo.BEGIN]
    mycgo = list()

    c0 = [0, 0, 0, 1]
    x0 = [xyzlen[0], 0, 0, 1]
    y0 = [0, xyzlen[1], 0, 1]
    z0 = [0, 0, xyzlen[2], 1]
    if randpos > 0:
        rr = wu.homog.rand_xform()
        rr[:3, 3] *= randpos
        c0 = rr @ c0
        x0 = rr @ x0
        y0 = rr @ y0
        z0 = rr @ z0

    for xform in xforms:
        xform[:3, 3] *= scale
        cen = xform @ c0
        x = xform @ x0
        y = xform @ y0
        z = xform @ z0
        mycgo.extend(cgo_cyl(cen, x, 0.05 * weight, [1, 0, 0]))
        mycgo.extend(cgo_cyl(cen, y, 0.05 * weight, [0, 1, 0]))
        mycgo.extend(cgo_cyl(cen, z, 0.05 * weight, [0, 0, 1]))

    if showcen:
        cen = np.mean(xforms[:, :, 3], axis=0)
        mycgo.extend(cgo_sphere(cen=cen, rad=1, col=(1, 1, 1)))
    # mycgo.append(cgo.END)
    # print(mycgo)
    if make_cgo_only:
        return state, mycgo
    pymol.cmd.load_cgo(mycgo, 'xforms%i' % _nxforms)
    return state

def show_ndarray_lines(toshow, state=None, name=None, colors=None, **kw):
    name = name or "worms_thing"
    state["seenit"][name] += 1
    name += "_%i" % state["seenit"][name]

    assert toshow.shape[-2:] == (4, 2)
    toshow = toshow.reshape(-1, 4, 2)

    for i, ray in enumerate(toshow):
        color = colors[i] if colors else (1, 1, 1)
        showline(ray[:3, 1] * 100, ray[:3, 0], col=color)
        showsphere(ray[:3, 0], col=color)
    return state

def show_ndarray_point_or_vec(toshow, state=None, name=None, colors=None, **kw):
    name = name or "points"
    state["seenit"][name] += 1
    name += "_%i" % state["seenit"][name]
    if colors == 'rand':
        colors = get_different_colors(len(toshow))

    assert toshow.shape[-1] == 4
    for i, p_or_v in enumerate(toshow):
        color = (1, 1, 1) if colors is None else colors
        if isinstance(color[0], (list, tuple, np.ndarray)):
            color = color[i]
        if p_or_v[3] > 0.999:
            showsphere(p_or_v, 1.0, col=color)
        elif np.abs(p_or_v[3]) < 0.001:
            showvecfrompoint(p_or_v * 20, p_or_v, col=color)
        else:
            raise NotImplementedError

    return state

def show_ndarray_n_ca_c(toshow, state=None, name=None, **kw):
    name = name or "worms_thing"
    state["seenit"][name] += 1
    name += "_%i" % state["seenit"][name]

    tmpdir = tempfile.mkdtemp()
    fname = tmpdir + "/" + name + ".pdb"
    assert toshow.shape[-2:] == (3, 4)
    with open(fname, "w") as out:
        for i, a1 in enumerate(toshow.reshape(-1, 3, 4)):
            for j, a in enumerate(a1):
                line = format_atom(
                    atomi=3 * i + j,
                    resn="GLY",
                    resi=i,
                    atomn=(" N  ", " CA ", " C  ")[j],
                    x=a[0],
                    y=a[1],
                    z=a[2],
                )
                out.write(line)
    pymol.cmd.load(fname)
    return state

showme_state = dict(launched=0, seenit=defaultdict(lambda: -1))

def showme_pymol(what, name='noname', headless=False, block=False, **kw):
    if "PYTEST_CURRENT_TEST" in os.environ and not headless:
        print("NOT RUNNING PYMOL IN UNIT TEST")
        return

    pymol.pymol_argv = ["pymol"]
    if headless:
        pymol.pymol_argv = ["pymol", "-c"]
    if not showme_state["launched"]:
        pymol.finish_launching()
        showme_state["launched"] = 1

    print('############## showme_pymol', type(what), '##############')
    result = pymol_load(what, showme_state, name=name, **kw)
    # # pymol.cmd.set('internal_gui_width', '20')

    while block:
        time.sleep(1)
    return result

def showme(*args, how="pymol", **kw):
    if how == "pymol":
        return showme_pymol(*args, **kw)
    else:
        raise NotImplementedError('showme how="%s" not implemented' % how)

_atom_record_format = (
    "ATOM  {atomi:5d} {atomn:^4}{idx:^1}{resn:3s} {chain:1}{resi:4d}{insert:1s}   "
    "{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{b:6.2f}\n")

def format_atom(
    atomi=0,
    atomn="ATOM",
    idx=" ",
    resn="RES",
    chain="A",
    resi=0,
    insert=" ",
    x=0,
    y=0,
    z=0,
    occ=0,
    b=0,
):
    return _atom_record_format.format(**locals())

def is_rosetta_pose(toshow):
    return isinstance(toshow, Pose)

def pymol_load_pose(pose, name):
    tmpdir = tempfile.mkdtemp()
    fname = tmpdir + "/" + name + ".pdb"
    pose.dump_pdb(fname)
    pymol.cmd.load(fname)

def pymol_xform(name, xform):
    assert name in pymol.cmd.get_object_list()
    pymol.cmd.transform_object(name, xform.flatten())