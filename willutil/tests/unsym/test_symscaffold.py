import pytest
import willutil as wu
import numpy as np

numba = pytest.importorskip("numba")


def main():
    run_symscaffold_cyclic_nbrs()
    run_symscaffold_vis()


# def run_symscaffold():
# assert 0


def run_symscaffold_cyclic_nbrs():
    scaff = wu.unsym.SymScaffIcos()
    n2, n3a, n3b = scaff.nbr2, scaff.nbr3a, scaff.nbr3b
    n5a, n5b, n5c, n5d = scaff.nbr5a, scaff.nbr5b, scaff.nbr5c, scaff.nbr5d
    assert np.all(n2[n2] == np.arange(60))
    assert np.all(n3a[n3a[n3a]] == np.arange(60))
    assert np.all(n3b[n3b[n3b]] == np.arange(60))
    assert np.all(n3a[n3b] == np.arange(60))
    assert np.all(n3b[n3a] == np.arange(60))
    assert np.all(n5a[n5a[n5a[n5a[n5a]]]] == np.arange(60))
    assert np.all(n5b[n5b[n5b[n5b[n5b]]]] == np.arange(60))
    assert np.all(n5c[n5c[n5c[n5c[n5c]]]] == np.arange(60))
    assert np.all(n5d[n5d[n5d[n5d[n5d]]]] == np.arange(60))
    assert np.all(n5a[n5d] == np.arange(60))
    assert np.all(n5b[n5c] == np.arange(60))
    assert np.all(n5d[n5a] == np.arange(60))
    assert np.all(n5c[n5b] == np.arange(60))
    assert np.all(n5a[n5b][n5b] == np.arange(60))


def run_symscaffold_vis():
    scaff = wu.unsym.SymScaffIcos()
    nplaced, keep = list(), list()
    best = 0, None
    for i in range(10_000):
        scaff.reset()
        placed = wu.unsym.mark555(
            scaff.occ,
            scaff.nbrs,
        )
        nplaced.append(len(placed))
        if len(placed) > 17:
            keep.append(placed)
        if (i + 1) % 10_000 == 0:
            print("====================================")
            for count in np.histogram(nplaced, bins=[24, 25, 26, 27, 28, 29, 30, 31])[0]:
                # for count in np.histogram(nplaced, bins=[0, 13, 15, 17, 19, 21])[0]:
                print(count / len(nplaced))
            print(flush=True)

        if np.sum(scaff.occ) > best[0]:
            best = np.sum(scaff.occ), placed
        if len(placed) > 9999:
            ic(i)
            scaff.placed = placed
            wu.showme(scaff, headless=False, showcen=True)
            return

        scaff.reset()
    scaff.placed = best[1]
    for p in scaff.placed:
        for f, a in p:
            scaff.occ[f, a] = True
    wu.showme(scaff)


if __name__ == "__main__":
    main()
