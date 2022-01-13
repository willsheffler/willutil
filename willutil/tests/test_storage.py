import tempfile
import willutil as wu

def test_pickle_bunch():
    with tempfile.TemporaryDirectory() as tmpdir:
        b = wu.Bunch(config=wu.Bunch())
        wu.save(b, tmpdir + '/foo')
        c = wu.load(tmpdir + '/foo')
        assert b == c
        print(b)
        print(c)

if __name__ == '__main__':
    test_pickle_bunch()
