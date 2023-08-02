import willutil as wu
import sys

def main():
   for fname in sys.argv[1:]:
      print(fname)
      pdb = wu.readpdb(fname)
      ncac = pdb.ncac(splitchains=True)
      crd = wu.chem.add_bb_o_guess(ncac)
      wu.dumppdb(fname + 'replace_bbo.pdb', crd)

if __name__ == '__main__':
   main()
