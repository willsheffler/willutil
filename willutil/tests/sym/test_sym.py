import numpy as np
import willutil as wu

def main():
    # test_remove_if_same_axis()
    # test_sym_frames()
    # test_frames_asym_of()
    test_frames_asym_remove_sameaxis()

def test_frames_asym_of():
    f = wu.sym.frames('icos', asym_of='c5')
    assert len(f) == 12
    f = wu.sym.frames('icos', asym_of='c3')
    assert len(f) == 20

    f = wu.sym.frames('icos', asym_of='c2')
    assert len(f) == 30
    f = wu.sym.frames('oct', asym_of='c4')
    assert len(f) == 6

    f = wu.sym.frames('oct', asym_of='c3')
    assert len(f) == 8
    f = wu.sym.frames('oct', asym_of='c2')
    assert len(f) == 12

    f = wu.sym.frames('tet', asym_of='c3')
    assert len(f) == 4
    f = wu.sym.frames('tet', asym_of='c2')
    assert len(f) == 6

def test_frames_asym_remove_sameaxis():
    syms = 'tet oct icos'.split()
    csyms = 'c2 c3 c4 c5'.split()
    config = [
        ('tet  c2'.split(), (4, 6, 6, 12)),
        ('tet  c3'.split(), (2, 4, 4, 12)),
        ('oct  c2'.split(), (7, 12, 12, 24)),
        ('oct  c3'.split(), (4, 8, 8, 24)),
        ('oct  c4'.split(), (3, 6, 6, 24)),
        ('icos c2'.split(), (16, 30, 30, 60)),
        ('icos c3'.split(), (8, 20, 20, 60)),
        ('icos c5'.split(), (4, 12, 12, 60)),
    ]
    for i, ((sym, csym), (n1, n2, n3, n4)) in enumerate(config):
        # print(i, sym, csym)
        cart = 10 * wu.sym.axes(sym, csym)

        f = wu.sym.frames(sym, bbsym=csym, asym_of=csym, axis=[0, 0, 1])
        assert len(f) == n1
        # print(i, sym, csym, len(f))
        # f[:, :, 3] += wu.homog.hdot(f, cart)
        # wu.viz.showme(f, spheres=0.5, name=f'test_{sym}_{csym}_bbsym_asymof')

        f = wu.sym.frames(sym, asym_of=csym, axis=[0, 0, 1])
        assert len(f) == n2
        # print(i, sym, csym, len(f))
        # f[:, :, 3] += wu.homog.hdot(f, cart)
        # wu.viz.showme(f, spheres=0.4, name=f'test_{sym}_{csym}_asymof')

        f = wu.sym.frames(sym, bbsym=csym, axis=[0, 0, 1])
        assert len(f) == n3
        # print(i, sym, csym, len(f))
        # f[:, :, 3] += wu.homog.hdot(f, cart)
        # wu.viz.showme(f, spheres=0.3, name=f'test_{sym}_{csym}_bbsym')

        f = wu.sym.frames(sym, axis=[0, 0, 1], axis0=wu.sym.axes(sym, csym))
        assert len(f) == n4
        # print(i, sym, csym, len(f))
        # f[:, :, 3] += wu.homog.hdot(f, cart)
        # wu.viz.showme(f, spheres=0.2, name=f'test_{sym}_{csym}_all')

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
