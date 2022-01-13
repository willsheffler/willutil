import sys, os, tempfile, numpy as np, time
from collections import defaultdict
import willutil as wu
from logging import info
from functools import singledispatch
from deferred_import import deferred_import

# pymol = deferred_import('pymol')
import pymol
import pymol.cgo
import pymol.cmd

try:
    import DUMMY_DOESNT_EXIST
    from pyrosetta.rosetta.core.pose import Pose
except ImportError:
    # from unittest.mock import MagicMock
    # Pose = MagicMock()
    class DummyPose:
        pass

    Pose = DummyPose

@singledispatch
def pymol_load(toshow, state=None, name=None, **kw):
    raise NotImplementedError("pymol_load: don't know how to show " + str(type(toshow)))

@pymol_load.register(wu.homog.RelXformInfo)
def _(toshow, state, name='noname', col='bycx', **kw):
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
        col = np.random.rand(3) / 2 + 0.5
    cen = toshow.framecen

    showcyl(
        cen + toshow.axs * 20,
        cen - toshow.axs * 20,
        0.03,
        col=col,
    )
    showcyl(
        cen + toshow.axs * toshow.hel / 2,
        cen - toshow.axs * toshow.hel / 2,
        0.1,
        col=col,
    )
    delta = toshow.axs * (toshow.hel + 0.01) / 2
    shift = 0  # 0.5 * (np.random.rand() - 0.5)
    showfan(toshow.axs, cen + toshow.axs * shift, toshow.rad / 2, ang, col=col)

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
        c = xform @ c0
        x = xform @ x0
        y = xform @ y0
        z = xform @ z0
        mycgo.extend(cgo_cyl(c, x, 0.05, [1, 0, 0]))
        mycgo.extend(cgo_cyl(c, y, 0.05, [0, 1, 0]))
        mycgo.extend(cgo_cyl(c, z, 0.05, [0, 0, 1]))
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

    pymol.pymol_argv = ["pymol"]
    if headless:
        pymol.pymol_argv = ["pymol", "-c"]
    if not showme_state["launched"]:
        pymol.finish_launching()
        showme_state["launched"] = 1

    print('############## showme_pymol', type(what), '##############')
    r = pymol_load(what, showme_state, **kw)
    # # pymol.cmd.set('internal_gui_width', '20')

    while block:
        time.sleep(1)
    return r

def showme(*args, how="pymol", **kw):
    if how == "pymol":
        return showme_pymol(*args, **kw)
    else:
        raise NotImplementedError('showme how="%s" not implemented' % how)

numcom = 0
numvec = 0
numray = 0
numline = 0
numseg = 0

def showcom(sel="all"):
    global numcom
    c = com(sel)
    print("Center of mass: ", c)
    cgo = [
        pymol.cgo.COLOR,
        1.0,
        1.0,
        1.0,
        cgo.SPHERE,
        c[0],
        c[1],
        c[2],
        1.0,
    ]  # white sphere with 3A radius
    pymol.cmd.load_cgo(cgo, "com%i" % numcom)
    numcom += 1

def cgo_sphere(c, r=1, col=(1, 1, 1)):
    # white sphere with 3A radius
    return [cgo.COLOR, col[0], col[1], col[2], cgo.SPHERE, c[0], c[1], c[2], r]

def showsphere(c, r=1, col=(1, 1, 1), lbl=''):
    v = pymol.cmd.get_view()
    if not lbl:
        global numvec
        lbl = "sphere%i" % numvec
        numvec += 1
    mycgo = cgo_sphere(c=c, r=r, col=col)
    pymol.cmd.load_cgo(mycgo, lbl)
    pymol.cmd.set_view(v)

def showvecfrompoint(a, c, col=(1, 1, 1), lbl=''):
    if not lbl:
        global numray
        lbl = "ray%i" % numray
        numray += 1
    pymol.cmd.delete(lbl)
    v = pymol.cmd.get_view()
    OBJ = [
        cgo.BEGIN,
        cgo.LINES,
        cgo.COLOR,
        col[0],
        col[1],
        col[2],
        cgo.VERTEX,
        c[0],
        c[1],
        c[2],
        cgo.VERTEX,
        c[0] + a[0],
        c[1] + a[1],
        c[2] + a[2],
        cgo.END,
    ]
    pymol.cmd.load_cgo(OBJ, lbl)
    # pymol.cmd.load_cgo([cgo.COLOR, col[0],col[1],col[2],
    #             cgo.SPHERE,   c[0],       c[1],       c[2],    0.08,
    #             cgo.CYLINDER, c[0],       c[1],       c[2],
    #                       c[0] + a[0], c[1] + a[1], c[2] + a[2], 0.02,
    #               col[0],col[1],col[2],col[0],col[1],col[2],], lbl)
    pymol.cmd.set_view(v)

def cgo_segment(c1, c2, col=(1, 1, 1)):
    OBJ = [
        cgo.BEGIN,
        cgo.LINES,
        cgo.COLOR,
        col[0],
        col[1],
        col[2],
        cgo.VERTEX,
        c1[0],
        c1[1],
        c1[2],
        cgo.VERTEX,
        c2[0],
        c2[1],
        c2[2],
        cgo.END,
    ]
    # pymol.cmd.load_cgo([cgo.COLOR, col[0],col[1],col[2],
    #             cgo.CYLINDER, c1[0],     c1[1],     c1[2],
    #                           c2[0],     c2[1],     c2[2], 0.02,
    #               col[0],col[1],col[2],col[0],col[1],col[2],], lbl)
    return OBJ

def showsegment(c1, c2, col=(1, 1, 1), lbl=''):
    if not lbl:
        global numseg
        lbl = "seg%i" % numseg
        numseg += 1
    pymol.cmd.delete(lbl)
    v = pymol.cmd.get_view()
    pymol.cmd.load_cgo(cgo_segment(c1=c1, c2=c2, col=col), lbl)
    # pymol.cmd.load_cgo([cgo.COLOR, col[0],col[1],col[2],
    #             cgo.CYLINDER, c1[0],     c1[1],     c1[2],
    #                           c2[0],     c2[1],     c2[2], 0.02,
    #               col[0],col[1],col[2],col[0],col[1],col[2],], lbl)
    pymol.cmd.set_view(v)

def cgo_cyl(c1, c2, r, col=(1, 1, 1), col2=None):
    col2 = col2 or col
    return [  # cgo.COLOR, col[0],col[1],col[2],
        pymol.cgo.CYLINDER, c1[0], c1[1], c1[2], c2[0], c2[1], c2[2], r, col[0], col[1], col[2],
        col2[0], col2[1], col2[2]
    ]

def showcyl(c1, c2, r, col=(1, 1, 1), col2=None, lbl=''):
    if not lbl:
        global numseg
        lbl = "seg%i" % numseg
        numseg += 1
    pymol.cmd.delete(lbl)
    v = pymol.cmd.get_view()
    pymol.cmd.load_cgo(cgo_cyl(c1=c1, c2=c2, r=r, col=col, col2=col2), lbl)
    pymol.cmd.set_view(v)

def showline(a, c, col=(1, 1, 1), lbl=''):
    if not lbl:
        global numline
        lbl = "line%i" % numline
        numline += 1
    pymol.cmd.delete(lbl)
    v = pymol.cmd.get_view()
    OBJ = [
        pymol.cgo.BEGIN,
        pymol.cgo.LINES,
        pymol.cgo.COLOR,
        col[0],
        col[1],
        col[2],
        pymol.cgo.VERTEX,
        c[0] - a[0],
        c[1] - a[1],
        c[2] - a[2],
        pymol.cgo.VERTEX,
        c[0] + a[0],
        c[1] + a[1],
        c[2] + a[2],
        pymol.cgo.END,
    ]
    pymol.cmd.load_cgo(OBJ, lbl)
    pymol.cmd.set_view(v)

def cgo_lineabs(a, c, col=(1, 1, 1)):
    return [
        pymol.cgo.BEGIN,
        pymol.cgo.LINES,
        pymol.cgo.COLOR,
        col[0],
        col[1],
        col[2],
        pymol.cgo.VERTEX,
        c[0],
        c[1],
        c[2],
        pymol.cgo.VERTEX,
        a[0],
        a[1],
        a[2],
        pymol.cgo.END,
    ]

def showlineabs(a, c, col=(1, 1, 1), lbl=''):
    if not lbl:
        global numline
        lbl = "line%i" % numline
        numline += 1
    pymol.cmd.delete(lbl)
    v = pymol.cmd.get_view()
    cgo = cgo_lineabs(a, c, col)
    pymol.cmd.load_cgo(cgo, lbl)
    pymol.cmd.set_view(v)

def cgo_fan(axis, cen, rad, arc, col=(1, 1, 1), col2=None):
    ntri = 10
    if arc > 10: arc = np.radians(arc)
    col2 = col2 or col
    rot = wu.homog.hrot(axis, arc / ntri, cen)
    obj = []
    pt1 = np.array([1, 2, 3, 0])
    # pt1 = [0, 0, 0, 0]
    # pt1[np.argmin(axis[:3])] = 1
    axis = axis[:]
    if wu.homog.hdot(pt1, axis) < 0:
        pt1 = -pt1

    pt1 = wu.homog.proj_perp(axis, pt1)
    pt1 = cen + wu.homog.hnormalized(pt1) * rad
    for i in range(ntri):
        # yapf: disable
        print(pt1)
        pt2 = rot @ pt1
        # obj += [
        #     pymol.cgo.TRIANGLE,
        #        cen[0],cen[1],cen[2],
        #        pt1[0],pt1[1],pt1[2],
        #        pt2[0],pt2[1],pt2[2],
        #        # normal-x1, normal-y1, normal-z1,
        #        # axis[0]-cen[0],axis[1]-cen[1],axis[2]-cen[2],
        #        # axis[0]-pt1[0],axis[1]-pt1[1],axis[2]-pt1[2],
        #        # axis[0]-pt2[0],axis[1]-pt2[1],axis[2]-pt2[2],
        #        axis[0],axis[1],axis[2],
        #        axis[0],axis[1],axis[2],
        #        axis[0],axis[1],axis[2],
        #        col[0],col[1],col[2],
        #        col[0],col[1],col[2],
        #        col[0],col[1],col[2],
        # ]


        obj += [
               pymol.cgo.BEGIN,
               pymol.cgo.TRIANGLES,
               pymol.cgo.COLOR,    col[0],  col[1],  col[2],
               pymol.cgo.ALPHA, 1,
               pymol.cgo.NORMAL,  axis[0], axis[1], axis[2],
               pymol.cgo.VERTEX,  cen [0], cen [1], cen [2],
               pymol.cgo.NORMAL,  axis[0], axis[1], axis[2],
               pymol.cgo.VERTEX,  pt1 [0], pt1 [1], pt1 [2],
               pymol.cgo.NORMAL,  axis[0], axis[1], axis[2],
               pymol.cgo.VERTEX,  pt2 [0], pt2 [1], pt2 [2],
               pymol.cgo.END,
            ]



        pt1 = pt2
        # yapf: enable
    return obj

def showfan(axis, cen, rad, arc, col=(1, 1, 1), lbl=''):
    if not lbl:
        global numseg
        lbl = "seg%i" % numseg
        numseg += 1
    pymol.cmd.delete(lbl)
    v = pymol.cmd.get_view()
    pymol.cmd.load_cgo(cgo_fan(axis=axis, cen=cen, rad=rad, arc=arc, col=col), lbl)
    pymol.cmd.set_view(v)

def showaxes():
    v = pymol.cmd.get_view()
    obj = [
        pymol.cgo.BEGIN, pymol.cgo.LINES, pymol.cgo.COLOR, 1.0, 0.0, 0.0, pymol.cgo.VERTEX, 0.0,
        0.0, 0.0, pymol.cgo.VERTEX, 20.0, 0.0, 0.0, pymol.cgo.COLOR, 0.0, 1.0, 0.0,
        pymol.cgo.VERTEX, 0.0, 0.0, 0.0, pymol.cgo.VERTEX, 0.0, 20.0, 0.0, pymol.cgo.COLOR, 0.0,
        0.0, 1.0, pymol.cgo.VERTEX, 0.0, 0.0, 0.0, pymol.cgo.VERTEX, 00, 0.0, 20.0, pymol.cgo.END
    ]
    pymol.cmd.load_cgo(obj, "axes")

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
