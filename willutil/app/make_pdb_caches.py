import sys
import logging
import willutil as wu

logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
# logging.basicConfig(encoding='utf-8', level=logging.DEBUG)
import sys

def getpdbs(files_or_pattern):
   return wu.pdb.pdbread.load_pdbs(
      files_or_pattern,
      cache=True,
      skip_errors=True,
   )

def extract_het_pdbfiles(files_or_pattern):
   for fname, pdb in wu.pdb.pdbread.gen_pdbs(
         files_or_pattern,
         cache=True,
         skip_errors=True,
   ):
      # print('extract_het_pdbfiles', fname)
      h = pdb.subset(het=True, removeres=['HOH'])
      print('save', fname + '.het.pickle')
      wu.save(h, fname.replace('.pickle', '') + '.het.pickle')

def main(pattern):
   with wu.Timer():
      print(sys.argv[:4])
      # pdbs = getpdbs(files_or_pattern)
      # extract_het_pdbfiles(files_or_pattern)
      import glob
      import os
      for f in glob.glob(pattern):
         # print(f)
         os.rename(f, f.replace('pickle.het', 'het'))

if __name__ == '__main__':
   # main(sys.argv[1:])
   # main('/home/sheffler/data/rcsb/divided/pg/1pg0.pdb1.gz')
   main('/home/sheffler/data/rcsb/divided/??/*.pdb?.gz.pickle.het.pickle')
