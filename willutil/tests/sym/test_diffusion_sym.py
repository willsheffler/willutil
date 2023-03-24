import willutil as wu
from willutil.sym.diffusion_sym import *

def main():
   test_symmatrix12()
   test_symmatrix24()
   test_symmatrix60()

def test_symmatrix12():
   symmatrix = wu.sym.symframes.tet_symmatrix
   frames = wu.hconvert(wu.sym.symframes.tet_Rs)
   assert wu.hvalid(frames)
   first = frames[symmatrix[0]] @ wu.hinv(frames[0])
   for i in range(len(symmatrix)):
      x = frames[symmatrix[i]] @ wu.hinv(frames[i])
      assert np.allclose(first, x)

def test_symmatrix24():
   symmatrix = wu.sym.symframes.oct_symmatrix
   frames = wu.hconvert(wu.sym.symframes.oct_Rs)
   assert wu.hvalid(frames)
   first = frames[symmatrix[0]] @ wu.hinv(frames[0])
   for i in range(len(symmatrix)):
      x = frames[symmatrix[i]] @ wu.hinv(frames[i])
      assert np.allclose(first, x)

def test_symmatrix60():
   symmatrix = wu.sym.symframes.icos_symmatrix
   frames = wu.hconvert(wu.sym.symframes.icos_Rs)
   assert wu.hvalid(frames)
   first = frames[symmatrix[0]] @ wu.hinv(frames[0])
   for i in range(len(symmatrix)):
      x = frames[symmatrix[i]] @ wu.hinv(frames[i])
      assert np.allclose(first, x)

if __name__ == '__main__':
   main()
