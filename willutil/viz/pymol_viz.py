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

@pymol_load.register(wu.homog.RelXformInfo)
def _(toshow, state, col='bycx', name='xrel', showframes=True, **kw):
    ang = toshow.ang
    if np.isclose(ang, np.pi * 4 / 5, atol=1e-4):
        ang /= 2
    if col == 'bycx':
        if np.isclose(ang, np.pi * 2 / 2, atol=0.01): col = [1, 1, 0]
        elif np.isclose(ang, np.pi * 2 / 3, atol=0.01): col = [0, 1, 1]
        elif np.isclose(ang, np.pi * 2 / 4, atol=0.01): col = [1, 0, 1]
        elif np.isclose(ang, np.pi * 2 / 5, atol=0.01): col = [1, 0, 1]
        elif np.isclose(ang, np.pi * 2 / 6, atol=0.01): col = [1, 0, 1]
        else: col = [1, 1, 1]
    elif col == 'random':
        # col = np.random.rand(3) / 2 + 0.5
        col = (1, 1, 1)
    cen = toshow.framecen

    if showframes:
        pymol_visualize_xforms(toshow.frames, state, **kw)

    cen1 = toshow.frames[0, :, 3]
    cen2 = toshow.frames[1, :, 3]

    if abs(toshow.ang) < 1e-6:
        showcyl(cen1, cen2, 0.01, col=(1, 1, 1))
        showsphere(cen=(cen1 + cen2) / 2, rad=0.1, col=(1, 1, 1))

    else:

        showcyl(
            cen + toshow.axs * 20,
            cen - toshow.axs * 20,
            0.03,
            col=col,
        )
        showcyl(
            cen + toshow.axs * toshow.hel / 2,
            cen - toshow.axs * toshow.hel / 2,
            0.15,
            col=col,
        )
        shift = 1.0 * (np.random.rand() - 0.5)
        showfan(
            toshow.axs,
            cen + toshow.axs * shift,
            toshow.rad / 4,
            arc=ang,
            col=col,
            startpoint=cen1,
        )

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
def _(toshow, state=None, name=None, **kw):
    if "pose" in toshow:
        state = pymol_load(toshow["pose"], state, **kw)
        pymol_xform(toshow["position"], state["last_obj"])
    else:
        for k, v in toshow.items():
            state = pymol_load(v, state, name=k, **kw)
    return state

@pymol_load.register(list)
def _(toshow, state=None, name=None, **kw):
    for t in toshow:
        print('    ##############', type(t), '################')
        state = pymol_load(t, state, **kw)
    return state

@pymol_load.register(np.ndarray)
def _(toshow, state, **kw):
    showaxes()
    if toshow.shape[-2:] == (3, 4):
        return show_ndarray_n_ca_c(toshow, state, **kw)
    elif toshow.shape[-2:] == (4, 2):
        return show_ndarray_lines(toshow, state, **kw)
    elif toshow.shape[-2:] == (4, 4):
        return pymol_visualize_xforms(toshow, state, **kw)

def pymol_visualize_xforms(
    xforms,
    state,
    name=None,
    randpos=0.0,
    xyzlen=[5 / 4, 1, 4 / 5],
    scale=1.0,
    **kw,
):
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
        mycgo.extend(cgo_cyl(cen, x, 0.05, [1, 0, 0]))
        mycgo.extend(cgo_cyl(cen, y, 0.05, [0, 1, 0]))
        mycgo.extend(cgo_cyl(cen, z, 0.05, [0, 0, 1]))
    # mycgo.append(cgo.END)
    # print(mycgo)
    pymol.cmd.load_cgo(mycgo, name)
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

    print(pymol.cmd)

    pymol.pymol_argv = ["pymol"]
    if headless:
        pymol.pymol_argv = ["pymol", "-c"]
    if not showme_state["launched"]:
        pymol.finish_launching()
        showme_state["launched"] = 1

    print('############## showme_pymol', type(what), '##############')
    result = pymol_load(what, showme_state, **kw)
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
