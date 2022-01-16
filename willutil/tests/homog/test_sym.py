import willutil.homog as hm

def test_sym_frames():
    assert len(hm.tetrahedral_axes_all[2] == 6)
    assert len(hm.tetrahedral_axes_all[3] == 4)
    assert len(hm.tetrahedral_axes_all[7] == 4)
    assert len(hm.octahedral_axes_all[2] == 12)
    assert len(hm.octahedral_axes_all[3] == 8)
    assert len(hm.octahedral_axes_all[4] == 6)
    assert len(hm.icosahedral_axes_all[2] == 30)
    assert len(hm.icosahedral_axes_all[3] == 20)
    assert len(hm.icosahedral_axes_all[5] == 12)

if __name__ == '__main__':
    test_sym_frames()
    # import willutil as wu
    # wu.viz.showme(hm.tetrahedral_axes_all[2], colors=(1, 0, 0))
    # wu.viz.showme(hm.tetrahedral_axes_all[3], colors=(0, 0, 1))
    # wu.viz.showme(hm.tetrahedral_axes_all[7], colors=(0, 1, 0))
    # wu.viz.showme(hm.octahedral_axes_all[2], colors=(1, 0, 0))
    # wu.viz.showme(hm.octahedral_axes_all[3], colors=(0, 0, 1))
    # wu.viz.showme(hm.octahedral_axes_all[4], colors=(0, 1, 0))
    # wu.viz.showme(hm.icosahedral_axes_all[2], colors=(1, 0, 0))
    # wu.viz.showme(hm.icosahedral_axes_all[3], colors=(0, 0, 1))
    # wu.viz.showme(hm.icosahedral_axes_all[5], colors=(0, 1, 0))
