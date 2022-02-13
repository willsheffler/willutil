import numpy as np
import willutil as wu

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
