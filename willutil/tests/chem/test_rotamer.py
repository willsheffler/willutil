import numpy as np
import willutil as wu

def main():
    test_rotamers()
    test_align()

def test_rotamers():
    rs = wu.chem.get_rotamerset()
    r = rs[(rs.resname == 'ASP') | (rs.resname == 'ASN')]
    assert r.rotnum.min() == 0
    assert r.rotnum.max() + 1 == len(r.rotnum[r.atomnum == 1])

def test_align():
    rs = wu.chem.get_rotamerset()
    r = rs.rotamers('ASP')
    r.align(['CG', 'OD1', 'OD2'])
    assert np.allclose(r.coords[:, -3], 0, atol=1e-4)
    assert np.allclose(r.coords[:, -3, 1:], 0, atol=1e-4)
    assert np.allclose(r.coords[:, -3, 2:], 0, atol=1e-4)
    assert np.allclose(r.coords[0, -3:], r.coords[:, -3:], atol=1e-4)

if __name__ == '__main__':
    main()
