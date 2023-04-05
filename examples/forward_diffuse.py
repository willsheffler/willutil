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
      delta = wu.hrandsmall(len(coords), cart_sd=beta * 500, rot_sd=beta * 1, centers=coords[:, 1])
      coords = wu.hxform(delta, coords, outerprod=False)
      symcoords = wu.hxform(symframes, coords)
      wu.dumppdb(f'{outname}_{i:03}.pdb', symcoords)

def betaschedule(T, bT, b0):
   noise = bT + 0.5 * (b0 - bT) * (1 + np.cos((np.arange(T - 1) / (T - 1)) * np.pi))
   return np.concatenate([[0], noise])

def main():
   fname = '/home/sheffler/for/alexi/C8_n5_24.pdb'
   noise = betaschedule(T=20, bT=0.035, b0=0.005)
   fake_diffuse(fname, noise, outname='fakediffuse', nfold=1)

if __name__ == '__main__':
   main()
