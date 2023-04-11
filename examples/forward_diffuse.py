import sys, os
import numpy as np

sys.path.append('/home/sheffler/src/willutil_dev')

import willutil as wu

def fake_diffuse(fname, noise, outname, nfold=1):
   pdb = wu.readpdb(fname).subset(chain='A')
   coords = pdb.ncaco()
   ic(coords.shape, np.max(coords))
   symframes = wu.sym.frames(f'C{nfold}')
   for i, beta in enumerate(noise):
      print('step', i, noise[i])
      delta = wu.hrandsmall(len(coords), cart_sd=beta * 500, rot_sd=beta * 10, centers=coords[:, 1])
      coords = wu.hxform(delta, coords, outerprod=False)
      symcoords = wu.hxform(symframes, coords)
      wu.dumppdb(f'{outname}_{i:03}.pdb', symcoords)

def betaschedule(T, bT, b0):
   noise = bT + 0.5 * (b0 - bT) * (1 + np.cos((np.arange(T - 1) / (T - 1)) * np.pi))
   return np.concatenate([[0], noise])

def main():
   fname = '/home/sheffler/Downloads/redes_sym_nn5_15_bb_regularized_aligned_0001.pdb'
   # fname = '/home/sheffler/for/alexi/C8_n5_24.pdb'
   noise = betaschedule(T=100, bT=0.01, b0=0.0002)
   outname = os.path.basename(fname) + '_fakediffuse'
   fake_diffuse(fname, noise, outname=outname, nfold=8)

if __name__ == '__main__':
   main()
