import datetime, sys, inspect, os, functools
import willutil as wu

_WARNINGS_ISSUED = set()

def WARNME(message, once=True):
   if once and message not in _WARNINGS_ISSUED:
      import traceback
      print('-' * 80, flush=True)
      print(message)
      traceback.print_stack()
      _WARNINGS_ISSUED.add(message)
      print('-' * 80)
      return True
   return False

class Tee:
   def __init__(self, fd1, fd2=sys.stdout):
      if isinstance(fd1, str):
         self.fname = fd1
         fd1 = open(fd1, 'w')
      self.fd1 = fd1
      self.fd2 = fd2
      self.with_stderr = False

   # def __del__(self):
   #     if self.fd1 != sys.stdout and self.fd1 != sys.stderr:
   #         self.fd1.close()
   #     if self.fd2 != sys.stdout and self.fd2 != sys.stderr:
   #         self.fd2.close()

   def write(self, text):
      self.fd1.write(text)
      self.fd2.write(text)
      self.flush()

   def flush(self):
      self.fd1.flush()
      self.fd2.flush()

def stdout_tee(fname, with_stderr=False):
   print('!!!!!!! stdout_tee', fname, 'with_stderr:', with_stderr)
   tee = Tee(fname)
   sys.stdout = tee
   if with_stderr:
      sys.stderr = Tee(tee.fd1, sys.stderr)
      sys.stdout.with_stderr = True

def stdout_untee():
   tee = sys.stdout
   tee.fd1.close()
   sys.stdout = tee.fd2
   if tee.with_stderr:
      sys.stderr = sys.stderr.fd2
   print('!!!!!!! stdout_untee', tee.fname)

class Flusher:
   def __init__(self, out):
      self.out = out

   def write(self, *args, **kw):
      self.out.write(*args, **kw)
      self.out.flush()

   def close(self):
      self.out.close()

def tobytes(s):
   if isinstance(s, str): return s.encode()
   return s

def tostr(s):
   if isinstance(s, bytes): return s.decode()
   return s

def datetimetag(sep='_'):
   now = datetime.datetime.now()
   if sep == 'label':
      return now.strftime(f'y%Ym%md%dh%Hm%Ms%S')
   return now.strftime(f'%Y{sep}%m{sep}%d{sep}%H{sep}%M{sep}%S')

def datetag(sep='_'):
   now = datetime.datetime.now()
   if sep == 'label':
      return now.strftime(f'y%Ym%md%d')
   return now.strftime(f'%Y{sep}%m{sep}%d')

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
