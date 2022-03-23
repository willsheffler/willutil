import tempfile
import willutil as wu

def test_pickle_bunch():

   # for parallel testing, only do on the main thread
   import threading
   if not threading.current_thread() is threading.main_thread():
      return

   with tempfile.TemporaryDirectory() as tmpdir:
      b = wu.Bunch(config=wu.Bunch())
      wu.save(b, tmpdir + '/foo')
      c = wu.load(tmpdir + '/foo')
      assert b == c
      print(b)
      print(c)

if __name__ == '__main__':
   test_pickle_bunch()
