import numpy as np
import willutil as wu

def main():
    test_remove_if_same_axis()

def test_remove_if_same_axis():

    f = wu.sym.frames('tet')
    assert f.shape == (12, 4, 4)
    assert wu.sym.frames('tet', bbsym='c2').shape == (6, 4, 4)
    assert wu.sym.frames('tet', bbsym='c3').shape == (4, 4, 4)

    f = wu.sym.frames('oct')
    assert f.shape == (24, 4, 4)
    assert wu.sym.frames('oct', bbsym='c2').shape == (12, 4, 4)
    assert wu.sym.frames('oct', bbsym='c3').shape == (8, 4, 4)
    assert wu.sym.frames('oct', bbsym='c4').shape == (6, 4, 4)

    f = wu.sym.frames('icos')
    assert f.shape == (60, 4, 4)
    assert wu.sym.frames('icos', bbsym='c2').shape == (30, 4, 4)
    assert wu.sym.frames('icos', bbsym='c3').shape == (20, 4, 4)
    assert wu.sym.frames('icos', bbsym='c5').shape == (12, 4, 4)

def test_sym():
    assert wu.sym.symframes.tetrahedral_frames.shape == (12, 4, 4)
    assert wu.sym.symframes.octahedral_frames.shape == (24, 4, 4)
    assert wu.sym.symframes.icosahedral_frames.shape == (60, 4, 4)
    x = np.concatenate([
        wu.sym.symframes.tetrahedral_frames,
        wu.sym.symframes.octahedral_frames,
        wu.sym.symframes.icosahedral_frames,
    ])
    assert np.all(x[..., 3, 3] == 1)
    assert np.all(x[..., 3, :3] == 0)
    assert np.all(x[..., :3, 3] == 0)

def test_sym_frames():
    assert len(wu.sym.tetrahedral_axes_all[2] == 6)
    assert len(wu.sym.tetrahedral_axes_all[3] == 4)
    assert len(wu.sym.tetrahedral_axes_all['2b'] == 4)
    assert len(wu.sym.octahedral_axes_all[2] == 12)
    assert len(wu.sym.octahedral_axes_all[3] == 8)
    assert len(wu.sym.octahedral_axes_all[4] == 6)
    assert len(wu.sym.icosahedral_axes_all[2] == 30)
    assert len(wu.sym.icosahedral_axes_all[3] == 20)
    assert len(wu.sym.icosahedral_axes_all[5] == 12)

if __name__ == '__main__':
    main()
