import numpy as np
import willutil as wu


def main():
    frames = wu.sym.frames("tet")
    print("first frames of canonical tet symmetry\n", frames[:2])
    axes = wu.sym.axes("tet")
    print("tetrahedral main 2fold axis", axes[2])
    print("tetrahedral main 3fold axis", axes[3])

    axes = wu.sym.axes("icos")
    print("icos main 2fold axis", axes[2])
    print("icos main 3fold axis", axes[3])
    print("icos main 5fold axis", axes[5])

    print("oct sym axes", wu.sym.axes("oct"))
    print("d3 sym axes", wu.sym.axes("d3"))

    syminfo = wu.sym.get_syminfo("tet")
    print("12 symmetric frames", syminfo.frames.shape)
    print("primary 2fold axis", syminfo.axes[2])
    print("all 2fold axes\n", syminfo.axesall[2])
    print("cyclic symmetry angles", syminfo.point_angles)

    print("octahedral axis angles", wu.sym.nfold_axis_angles["oct"])

    frames_tc2 = wu.sym.frames("tet", bbsym="c2")
    frames_tc3 = wu.sym.frames("tet", bbsym="c3")
    print("tetrahedral dimer first frames\n", frames_tc2[:2])
    print("tetrahedral trimer first frames\n", frames_tc3[:2])

    frames_c3t = wu.sym.frames("tet", asym_of="c3")
    print(
        "trimeric symmetry from tetrahedral frames. first and last align with c3 symmetry and only contain one subunit. middle frames are for whole dimers\n",
        frames_c3t,
    )

    print("mimimum angle between any pair of symaxes", wu.sym.min_symaxis_angle("tet"))

    print("aubiguous axes tet", wu.sym.ambiguous_axes("tet"))
    print("aubiguous axes oct, inferred 2fold could be 4fold", wu.sym.ambiguous_axes("oct"))

    frames = wu.sym.frames("oct")
    xglobal = wu.homog.rand_xform()
    xlocal = wu.homog.rand_xform_small(3, rot_sd=np.radians(1), cart_sd=0.01)
    randframes = xglobal @ frames[:3] @ xlocal
    fitinfo = wu.sym.compute_symfit("oct", randframes)
    to_original = fitinfo.xfit @ xglobal
    i = np.argmin(np.sum((np.eye(4) - frames @ to_original) ** 2, axis=(1, 2)))
    x2orig = frames[i] @ to_original
    print(
        "randomly move/rotate 3 octahedral frames and fit, off by: \n",
        wu.hangle_of(x2orig),
        "radians",
        np.linalg.norm(x2orig[:3, 3]),
        "cart",
    )


if __name__ == "__main__":
    main()
