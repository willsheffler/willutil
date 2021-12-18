import sys, logging
import willutil as wu

logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
# logging.basicConfig(encoding='utf-8', level=logging.DEBUG)
import sys

with wu.Timer():
   print(sys.argv[:4])
   wu.pdb.pdbread.load_pdbs(sys.argv[1:], cache=True)
