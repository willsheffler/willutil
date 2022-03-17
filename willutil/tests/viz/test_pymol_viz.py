import willutil as wu
import willutil.homog as hm
import willutil.viz

def main():
    pass
    # pymol_viz_example()

def pymol_viz_example():
    import willutil.viz
    frame1 = hm.hframe([1, 0, 0], [0, 1, 0], [0, 0, 1], [9, 0, 0])
    rel = hm.hrot([1, 0, 0], 90, [0, 0, 10])
    rel[0, 3] = 3
    frame2 = rel @ frame1
    xinfo = wu.sym.rel_xform_info(frame1, frame2)
    wu.viz.showme(xinfo)

if __name__ == '__main__':
    main()
