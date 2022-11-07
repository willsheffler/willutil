import numpy as np
import willutil as wu

def main():
   test_asufit()

def test_asufit():
   fname = wu.tests.testdata.test_data_path('pdb/px012.pdb')
   pdb = wu.pdb.load_pdb(fname)
   xyz = np.stack([pdb.df['x'], pdb.df['y'], pdb.df['z']]).T

   ic(xyz.shape)

   axstan = np.arctan(wu.sym.min_symaxis_angle('icos'))
   # frames = wu.sym.frames('icos')
   ax2 = wu.sym.axes('icos')[2]
   ax3 = wu.sym.axes('icos')[3]
   frames = [np.eye(4), wu.hrot(ax2, 180), wu.hrot(ax3, 120)]  #, wu.hrot(ax3, 240)]
   wu.sym.asufit(frames, xyz, radbias=1 / axstan)

if __name__ == '__main__':
   main()