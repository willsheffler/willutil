import willutil as wu
import numpy as np

from willutil.search.slidedock import slide_dock_oligomer

def main():
   _test_slidedock_onecomp()

def _test_slidedock_onecomp():
   # pdb = wu.readpdb('/home/sheffler/project/symmmotif_HE/input/test_trimer_icos_2.pdb')
   # xyz = pdb.ncac(splitchains=True)
   # np.save('/tmp/xyz.npy', xyz)
   xyz = np.load('/tmp/xyz.npy')
   startaxis = np.load('/tmp/startaxis.npy')

   print(xyz.shape)
   # wu.showme(xyz)
   ic(wu.hcom(xyz.reshape(-1, 3)))

   with wu.Timer():
      newxyz = slide_dock_oligomer(
          'icos',
          'c3',
          'c2',
          xyz,
          startaxis=startaxis,
      )
   wu.dumppdb('test_slide_dock_oligomer.pdb', wu.hxform(wu.sym.frames('icos'), newxyz))

if __name__ == '__main__':
   main()
