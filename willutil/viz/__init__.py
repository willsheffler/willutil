from willutil.viz import pymol_cgo
from willutil.viz import pymol_viz
from willutil.viz.primitives import *
from willutil.viz.viz_xtal import *
from willutil.viz.viz_rigidbody import *
from willutil.viz.pymol_viz import *

try:
   from willutil.viz import pymol_cgo
   from willutil.viz import pymol_viz
   from willutil.viz.primitives import *
   from willutil.viz.viz_xtal import *
   from willutil.viz.pymol_viz import *
except ImportError:
   printed_warning = False

   def showme(*a, **b):
      global printed_warning
      if not printed_warning:
         printed_warning = True
         print('!' * 80)
         print('WARNING willutil.viz.showme not available without pymol')
         print('!' * 80)

from willutil.viz.plot import *
