import tempfile
import random
from willutil.pdb.pdbfile import PDBFile
from willutil.viz.pymol_viz import pymol_load

@pymol_load.register(PDBFile)
def pymol_viz_pdbfile(
   pdb,
   state,
   name='pdb',
   **kw,
):
   tag = str(random.random())[2:]
   # ic(tag)
   with tempfile.TemporaryDirectory() as td:

      pdb.dump_pdb(f'{td}/{tag}.pdb')

      from pymol import cmd

      cmd.load(f'{td}/{tag}.pdb')
