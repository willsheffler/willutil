from time import perf_counter

def test_init_is_fast():
   t0 = perf_counter()
   t = perf_counter() - t0
   print('time to import willutil ', t)
   assert t < 0.5

if __name__ == '__main__':
   test_init_is_fast()
