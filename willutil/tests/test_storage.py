import tempfile
import willutil as wu

def main():
   test_fname_and_extension()
   test_pickle_bunch()

def test_fname_and_extension():

   print(list(wu.storage.fname_extensions('d/e/f').values()))
   assert list(wu.storage.fname_extensions('d/e/f').values()) == ['d/e/', 'f', '', '', 'f', 'f', '', 'd/e/f']
   assert list(wu.storage.fname_extensions('f.gz').values()) == ['', 'f', '', '.gz', 'f.gz', 'f', '.gz', 'f']
   assert list(wu.storage.fname_extensions('/d/e/f.tar.gz').values()) == [
       '/d/e/', 'f', '', '.tar.gz', 'f.tar.gz', 'f', '.tar.gz', '/d/e/f'
   ]
   assert list(wu.storage.fname_extensions('/d/f.k.tar.gz').values()) == [
       '/d/', 'f', '.k', '.tar.gz', 'f.k.tar.gz', 'f.k', '.k.tar.gz', '/d/f.k'
   ]
   assert list(wu.storage.fname_extensions('/d/f.k.xz').values()) == [
       '/d/', 'f', '.k', '.xz', 'f.k.xz', 'f.k', '.k.xz', '/d/f.k'
   ]
   assert list(wu.storage.fname_extensions('f.k').values()) == ['', 'f', '.k', '', 'f.k', 'f.k', '.k', 'f.k']

   assert list(wu.storage.fname_extensions('d/f.i.j.k.l').values()) == [
       'd/', 'f.i.j.k', '.l', '', 'f.i.j.k.l', 'f.i.j.k.l', '.l', 'd/f.i.j.k.l'
   ]

def test_pickle_bunch():

   # for parallel testing, only do on the main thread
   import threading
   if threading.current_thread() is not threading.main_thread():
      return

   with tempfile.TemporaryDirectory() as tmpdir:
      b = wu.Bunch(config=wu.Bunch())
      wu.save(b, tmpdir + '/foo')
      c = wu.load(tmpdir + '/foo')
      assert b == c
      # print(b)
      # print(c)

if __name__ == '__main__':
   main()
