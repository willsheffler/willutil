import willutil as wu
import willutil.homog as hm

def pymol_viz_example():
    frame1 = hm.hframe([1, 0, 0], [0, 1, 0], [0, 0, 1], [9, 0, 0])
    rel = hm.hrot([1, 0, 0], 90, [0, 0, 10])
    rel[0, 3] = 3
    frame2 = rel @ frame1
    xinfo = hm.rel_xform_info(frame1, frame2)
    wu.viz.showme(xinfo)

if __name__ == '__main__':
    pymol_viz_example()
