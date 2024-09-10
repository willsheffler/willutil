import numpy as np
import willutil as wu

def find_dups_in_rows(a):
    return np.any((a[:, :-1] == a[:, 1:]) * (a[:, :-1] != 0))

def main():
    b = np.random.randint(0, 9, (9, 9))
    b[:] = 0
    b = b % 9
    # print(np.sort(b, axis=1))

    print(find_dups_in_rows(b))
    print(find_dups_in_rows(b.T))

    # print(b[:3, :3])
    # print(b[3:6, 3:6])
    b33 = b.reshape(3, 3, 3, 3).swapaxes(1, 2).reshape(9, 9)
    print(find_dups_in_rows(b33))
    # print(b33)
    # print(np.sort(b33, axis=1))

    # test_unhashable_set()

def test_unhashable_set():
    for i in range(10):
        a = set(np.random.randint(7, size=2))
        b = set(np.random.randint(7, size=3))
        ua = wu.UnhashableSet(a)
        ub = wu.UnhashableSet(b)
        # ic(a)
        # ic(b)
        # ic(a - b)
        # ic(setminus(a, b))
        assert (a - b) == set(ua.difference(ub))
        # ic(b.intersection(a))
        # ic(setisect(b, a))
        assert b.intersection(a) == set(ub.intersection(ua))
        # ic(a == b)
        # ic(setequal(a, b))
        assert (a == b) == (ua == ub)

if __name__ == "__main__":
    main()
