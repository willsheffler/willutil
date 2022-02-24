import numpy as np
import willutil as wu
import pymol
from pymol import cgo, cmd

_numcom = 0
_numvec = 0
_numray = 0
_numline = 0
_numseg = 0

def showcom(sel="all"):
    global _numcom
    c = com(sel)
    print("Center of mass: ", c)
    mycgo = [
        cgo.COLOR,
        1.0,
        1.0,
        1.0,
        cgo.SPHERE,
        cen[0],
        cen[1],
        cen[2],
        1.0,
    ]  # white sphere with 3A radius
    pymol.cmd.load_cgo(mycgo, "com%i" % _numcom)
    _numcom += 1

def cgo_sphere(cen, rad=1, col=(1, 1, 1)):
    # white sphere with 3A radius
    return [cgo.COLOR, col[0], col[1], col[2], cgo.SPHERE, cen[0], cen[1], cen[2], rad]

def showsphere(cen, rad=1, col=(1, 1, 1), lbl=''):
    v = pymol.cmd.get_view()
    if not lbl:
        global _numvec
        lbl = "sphere%i" % _numvec
        _numvec += 1
    mycgo = cgo_sphere(cen=cen, rad=rad, col=col)
    pymol.cmd.load_cgo(mycgo, lbl)
    pymol.cmd.set_view(v)

def cgo_vecfrompoint(axis, cen, col=(1, 1, 1), lbl=''):
    OBJ = [
        cgo.BEGIN,
        cgo.LINES,
        cgo.COLOR,
        col[0],
        col[1],
        col[2],
        cgo.VERTEX,
        cen[0],
        cen[1],
        cen[2],
        cgo.VERTEX,
        cen[0] + axis[0],
        cen[1] + axis[1],
        cen[2] + axis[2],
        cgo.END,
    ]
    return OBJ

def showvecfrompoint(axis, cen, col=(1, 1, 1), lbl=''):
    if not lbl:
        global _numray
        lbl = "ray%i" % _numray
        _numray += 1
    pymol.cmd.delete(lbl)
    v = pymol.cmd.get_view()
    mycgo = cgo_vecfrompoints(axis, cen, col)
    pymol.cmd.load_cgo(OBJ, lbl)
    # pymol.cmd.load_cgo([cgo.COLOR, col[0],col[1],col[2],
    #             cgo.SPHERE,   cen[0],       cen[1],       cen[2],    0.08,
    #             cgo.CYLINDER, cen[0],       cen[1],       cen[2],
    #                       cen[0] +axis[0], cen[1] +axis[1], cen[2] +axis[2], 0.02,
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
        global _numseg
        lbl = "seg%i" % _numseg
        _numseg += 1
    pymol.cmd.delete(lbl)
    v = pymol.cmd.get_view()
    pymol.cmd.load_cgo(cgo_segment(c1=c1, c2=c2, col=col), lbl)
    # pymol.cmd.load_cgo([cgo.COLOR, col[0],col[1],col[2],
    #             cgo.CYLINDER, c1[0],     c1[1],     c1[2],
    #                           c2[0],     c2[1],     c2[2], 0.02,
    #               col[0],col[1],col[2],col[0],col[1],col[2],], lbl)
    pymol.cmd.set_view(v)

def cgo_cyl(c1, c2, rad, col=(1, 1, 1), col2=None):
    col2 = col2 or col
    return [  # cgo.COLOR, col[0],col[1],col[2],
        cgo.CYLINDER,
        c1[0],
        c1[1],
        c1[2],
        c2[0],
        c2[1],
        c2[2],
        rad,
        col[0],
        col[1],
        col[2],
        col2[0],
        col2[1],
        col2[2],
    ]

def showcyl(c1, c2, rad, col=(1, 1, 1), col2=None, lbl=''):
    if not lbl:
        global _numseg
        lbl = "seg%i" % _numseg
        _numseg += 1
    pymol.cmd.delete(lbl)
    v = pymol.cmd.get_view()
    pymol.cmd.load_cgo(cgo_cyl(c1=c1, c2=c2, rad=rad, col=col, col2=col2), lbl)
    pymol.cmd.set_view(v)

def showline(axis, cen, col=(1, 1, 1), lbl=''):
    if not lbl:
        global _numline
        lbl = "line%i" % _numline
        _numline += 1
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
        cen[0] - axis[0],
        cen[1] - axis[1],
        cen[2] - axis[2],
        cgo.VERTEX,
        cen[0] + axis[0],
        cen[1] + axis[1],
        cen[2] + axis[2],
        cgo.END,
    ]
    pymol.cmd.load_cgo(OBJ, lbl)
    pymol.cmd.set_view(v)

def cgo_lineabs(axis, cen, col=(1, 1, 1)):
    return [
        cgo.BEGIN,
        cgo.LINES,
        cgo.COLOR,
        col[0],
        col[1],
        col[2],
        cgo.VERTEX,
        cen[0],
        cen[1],
        cen[2],
        cgo.VERTEX,
        axis[0],
        axis[1],
        axis[2],
        cgo.END,
    ]

def showlineabs(axis, cen, col=(1, 1, 1), lbl=''):
    if not lbl:
        global _numline
        lbl = "line%i" % _numline
        _numline += 1
    pymol.cmd.delete(lbl)
    v = pymol.cmd.get_view()
    mycgo = cgo_lineabs(axis, cen, col)
    pymol.cmd.load_cgo(mycgo, lbl)
    pymol.cmd.set_view(v)

def cgo_fan(
        axis,
        cen,
        rad,
        arc,
        col=(1, 1, 1),
        col2=None,
        startpoint=[1, 2, 3, 1],
):
    ntri = 10
    if arc > 10: arc = np.radians(arc)
    col2 = col2 or col
    rot = wu.homog.hrot(axis, arc / (ntri + 0), cen)

    dirn = startpoint - cen

    dirn = wu.homog.proj_perp(axis, dirn)
    pt1 = cen + wu.homog.hnormalized(dirn) * rad
    obj = []
    for i in range(ntri):
        # yapf: disable
        # print(pt1)
        pt2 = rot @ pt1
        obj += [
               cgo.BEGIN,
               cgo.TRIANGLES,
               cgo.COLOR,    col[0],  col[1],  col[2],
               cgo.ALPHA, 1,
               cgo.NORMAL,  axis[0], axis[1], axis[2],
               cgo.VERTEX,  cen [0], cen [1], cen [2],
               cgo.NORMAL,  axis[0], axis[1], axis[2],
               cgo.VERTEX,  pt1 [0], pt1 [1], pt1 [2],
               cgo.NORMAL,  axis[0], axis[1], axis[2],
               cgo.VERTEX,  pt2 [0], pt2 [1], pt2 [2],
               cgo.END,
            ]



        pt1 = pt2
        # yapf: enable
    return obj

def showfan(axis, cen, rad, arc, col=(1, 1, 1), lbl='', **kw):
    if not lbl:
        global _numseg
        lbl = "seg%i" % _numseg
        _numseg += 1
    pymol.cmd.delete(lbl)
    v = pymol.cmd.get_view()
    pymol.cmd.load_cgo(cgo_fan(axis=axis, cen=cen, rad=rad, arc=arc, col=col, **kw), lbl)
    pymol.cmd.set_view(v)

def showaxes():
    v = pymol.cmd.get_view()
    obj = [
        cgo.BEGIN, cgo.LINES, cgo.COLOR, 1.0, 0.0, 0.0, cgo.VERTEX, 0.0, 0.0, 0.0, cgo.VERTEX,
        20.0, 0.0, 0.0, cgo.COLOR, 0.0, 1.0, 0.0, cgo.VERTEX, 0.0, 0.0, 0.0, cgo.VERTEX, 0.0,
        20.0, 0.0, cgo.COLOR, 0.0, 0.0, 1.0, cgo.VERTEX, 0.0, 0.0, 0.0, cgo.VERTEX, 00, 0.0, 20.0,
        cgo.END
    ]
    pymol.cmd.load_cgo(obj, "axes")

def cgo_cyl_arrow(c1, c2, r, col=(1, 1, 1), col2=None, arrowlen=4.0):
    if not col2:
        col2 = col
    CGO = []
    c1.round0()
    c2.round0()
    CGO.extend(cgo_cyl(c1, c2 + randnorm() * 0.0001, r=r, col=col, col2=col2))
    dirn = (c2 - c1).normalized()
    perp = projperp(dirn, Vec(0.2340790923, 0.96794275, 0.52037438472304783)).normalized()
    arrow1 = c2 - dirn * arrowlen + perp * 2.0
    arrow2 = c2 - dirn * arrowlen - perp * 2.0
    # -dirn to shift to sphere surf
    CGO.extend(cgo_cyl(c2 - dirn * 3.0, arrow1 - dirn * 3.0, r=r, col=col2))
    # -dirn to shift to sphere surf
    CGO.extend(cgo_cyl(c2 - dirn * 3.0, arrow2 - dirn * 3.0, r=r, col=col2))
    return CGO

def showcube(lb=[-10, -10, -10], ub=[10, 10, 10], r=0.1, xform=np.eye(4)):
    cmd.delete('CUBE')
    v = cmd.get_view()
    a = [
        wu.homog.hxform(xform, [ub[0], ub[1], ub[2]]),
        wu.homog.hxform(xform, [ub[0], ub[1], lb[2]]),
        wu.homog.hxform(xform, [ub[0], lb[1], lb[2]]),
        wu.homog.hxform(xform, [ub[0], lb[1], ub[2]]),
        wu.homog.hxform(xform, [lb[0], ub[1], ub[2]]),
        wu.homog.hxform(xform, [lb[0], ub[1], lb[2]]),
        wu.homog.hxform(xform, [lb[0], lb[1], lb[2]]),
        wu.homog.hxform(xform, [lb[0], lb[1], ub[2]]),
        wu.homog.hxform(xform, [lb[0], ub[1], ub[2]]),
        wu.homog.hxform(xform, [lb[0], ub[1], lb[2]]),
        wu.homog.hxform(xform, [lb[0], lb[1], ub[2]]),
        wu.homog.hxform(xform, [lb[0], lb[1], lb[2]]),
    ]
    b = [
        wu.homog.hxform(xform, [ub[0], ub[1], lb[2]]),
        wu.homog.hxform(xform, [ub[0], lb[1], lb[2]]),
        wu.homog.hxform(xform, [ub[0], lb[1], ub[2]]),
        wu.homog.hxform(xform, [ub[0], ub[1], ub[2]]),
        wu.homog.hxform(xform, [lb[0], ub[1], lb[2]]),
        wu.homog.hxform(xform, [lb[0], lb[1], lb[2]]),
        wu.homog.hxform(xform, [lb[0], lb[1], ub[2]]),
        wu.homog.hxform(xform, [lb[0], ub[1], ub[2]]),
        wu.homog.hxform(xform, [ub[0], ub[1], ub[2]]),
        wu.homog.hxform(xform, [ub[0], ub[1], lb[2]]),
        wu.homog.hxform(xform, [ub[0], lb[1], ub[2]]),
        wu.homog.hxform(xform, [ub[0], lb[1], lb[2]]),
    ]
    mycgo = [
        cgo.CYLINDER, a[0][0], a[0][1], a[0][2], b[0][0], b[0][1], b[0][2], r, 1, 1, 1, 1, 1, 1,
        cgo.CYLINDER, a[1][0], a[1][1], a[1][2], b[1][0], b[1][1], b[1][2], r, 1, 1, 1, 1, 1, 1,
        cgo.CYLINDER, a[2][0], a[2][1], a[2][2], b[2][0], b[2][1], b[2][2], r, 1, 1, 1, 1, 1, 1,
        cgo.CYLINDER, a[3][0], a[3][1], a[3][2], b[3][0], b[3][1], b[3][2], r, 1, 1, 1, 1, 1, 1,
        cgo.CYLINDER, a[4][0], a[4][1], a[4][2], b[4][0], b[4][1], b[4][2], r, 1, 1, 1, 1, 1, 1,
        cgo.CYLINDER, a[5][0], a[5][1], a[5][2], b[5][0], b[5][1], b[5][2], r, 1, 1, 1, 1, 1, 1,
        cgo.CYLINDER, a[6][0], a[6][1], a[6][2], b[6][0], b[6][1], b[6][2], r, 1, 1, 1, 1, 1, 1,
        cgo.CYLINDER, a[7][0], a[7][1], a[7][2], b[7][0], b[7][1], b[7][2], r, 1, 1, 1, 1, 1, 1,
        cgo.CYLINDER, a[8][0], a[8][1], a[8][2], b[8][0], b[8][1], b[8][2], r, 1, 1, 1, 1, 1, 1,
        cgo.CYLINDER, a[9][0], a[9][1], a[9][2], b[9][0], b[9][1], b[9][2], r, 1, 1, 1, 1, 1, 1,
        cgo.CYLINDER, a[10][0], a[10][1], a[10][2], b[10][0], b[10][1], b[10][2], r, 1, 1, 1, 1,
        1, 1, cgo.CYLINDER, a[11][0], a[11][1], a[11][2], b[11][0], b[11][1], b[11][2], r, 1, 1,
        1, 1, 1, 1
    ]
    # yapf: disable
    #   l=10#*sqrt(3)
    #   m=-l
    #   mycgo += [
    #      cgo.CYLINDER, 0,0,0, l, 0, 0, r, 1, 1, 1, 1, 1, 1,
    #      cgo.CYLINDER, 0,0,0, 0, l, 0, r, 1, 1, 1, 1, 1, 1,
    #      cgo.CYLINDER, 0,0,0, 0, 0, l, r, 1, 1, 1, 1, 1, 1,
    #      cgo.CYLINDER, 0,0,0, m, 0, 0, r, 1, 1, 1, 1, 1, 1,
    #      cgo.CYLINDER, 0,0,0, 0, m, 0, r, 1, 1, 1, 1, 1, 1,
    #      cgo.CYLINDER, 0,0,0, 0, 0, m, r, 1, 1, 1, 1, 1, 1,
    #
    #      cgo.CYLINDER, 0,0,0, l, l, 0, r, 1, 1, 1, 1, 1, 1,
    #      cgo.CYLINDER, 0,0,0, 0, l, l, r, 1, 1, 1, 1, 1, 1,
    #      cgo.CYLINDER, 0,0,0, l, 0, l, r, 1, 1, 1, 1, 1, 1,
    #      cgo.CYLINDER, 0,0,0, m, l, 0, r, 1, 1, 1, 1, 1, 1,
    #      cgo.CYLINDER, 0,0,0, 0, m, l, r, 1, 1, 1, 1, 1, 1,
    #      cgo.CYLINDER, 0,0,0, m, 0, l, r, 1, 1, 1, 1, 1, 1,
    #      cgo.CYLINDER, 0,0,0, m, m, 0, r, 1, 1, 1, 1, 1, 1,
    #      cgo.CYLINDER, 0,0,0, 0, m, m, r, 1, 1, 1, 1, 1, 1,
    #      cgo.CYLINDER, 0,0,0, m, 0, m, r, 1, 1, 1, 1, 1, 1,
    #      cgo.CYLINDER, 0,0,0, l, m, 0, r, 1, 1, 1, 1, 1, 1,
    #      cgo.CYLINDER, 0,0,0, 0, l, m, r, 1, 1, 1, 1, 1, 1,
    #      cgo.CYLINDER, 0,0,0, l, 0, m, r, 1, 1, 1, 1, 1, 1,
    #
    #   ]
    # yapf: enable
    cmd.load_cgo(mycgo, "CUBE")
    cmd.set_view(v)
