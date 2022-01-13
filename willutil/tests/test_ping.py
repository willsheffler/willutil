from willutil import PING

def foo():
    return bar()

def bar():
    return baz()

def baz():
    return PING('hello from baz', printit=False)

def test_PING():
    msg = foo()

    assert 'test_ping.foo:4' in msg
    assert 'test_ping.bar:7' in msg
    assert 'test_ping.baz:10' in msg
    assert 'hello from baz' in msg

if __name__ == '__main__':
    test_PING()
