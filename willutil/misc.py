import datetime

def tobytes(s):
    if isinstance(s, str): return s.encode()
    return s

def tostr(s):
    if isinstance(s, bytes): return s.decode()
    return s

def datetimetag():
    now = datetime.datetime.now()
    return now.strftime('%Y_%m_%d_%H_%M_%S')

def seconds_between_datetimetags(tag1, tag2):
    t1 = datetime_from_tag(tag1)
    t2 = datetime_from_tag(tag2)
    duration = t2 - t1
    return duration.total_seconds()

def datetime_from_tag(tag):
    vals = tag.split('_')
    assert len(vals) == 6
    vals = list(map(int, vals))
    # if this code is actually in service after 2099...
    # this failing assertion will be the least of our troubles
    # even worse if it's before I was born....(WHS)
    assert 1979 < vals[0] < 2100
    assert 0 < vals[1] <= 12  # months
    assert 0 < vals[2] <= 31  # days
    assert 0 < vals[3] <= 60  # hour
    assert 0 < vals[4] <= 60  # minute
    assert 0 < vals[5] <= 60  # second
    return datetime.datetime(*vals)

def generic_equals(this, that, checktypes=False, debug=False):
    import numpy as np
    if debug:
        print('generic_equals on types', type(this), type(that))
    if checktypes and type(this) != type(that):
        return False
    if isinstance(this, (str, bytes)):  # don't want to iter over strs
        return this == that
    if isinstance(this, dict):
        if len(this) != len(that):
            return False
        for k in this:
            if k not in that:
                return False
            if not generic_equals(this[k], that[k], checktypes, debug):
                return False
    if hasattr(this, '__iter__'):
        return all(generic_equals(x, y, checktypes, debug) for x, y in zip(this, that))
    if isinstance(this, np.ndarray):
        return np.allclose(this, that)
    if hasattr(this, 'equal_to'):
        return this.equal_to(that)
    if debug:
        print('!!!!!!!!!!', type(this))
        if this != that:
            print(this)
            print(that)
    return this == that
