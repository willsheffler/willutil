from deferred_import import deferred_import
import icecream

icecream.install()

from willutil.bunch import Bunch, bunchify, unbunchify
from willutil.timer import Timer
from willutil.ping import PING
from willutil import chem
from willutil import storage
from willutil.storage import load, save, load_package_data, open_package_data
from willutil.inprocess import InProcessExecutor
from willutil.cache import Cache, GLOBALCACHE
from willutil import runtests
from willutil import pdb
from willutil import mc
from willutil import misc
from willutil import tests
from willutil import reproducibility
from willutil import format
from willutil import homog
from willutil import sym
from willutil import viz

from willutil.mc import MonteCarlo

from willutil.sym import compute_symfit
from willutil.pdb import dump_pdb_from_ncac_points

# deferr import of cpp libs to avoid compilation if unnecessary
cpp = deferred_import('willutil.cpp')
# from willutil import cpp

emptybunch = Bunch()

from willutil.viz import showme

# figure things in homog starting with h should't pollute namespace too bad
from willutil.sym import compute_symfit as symfit
from willutil.homog import (
   # I,
   align_vector as halign,
   align_vector,
   align_vectors as halign2,
   angle as hangle,
   line_angle as hline_angle,
   angle_degrees as hangle_degrees,
   angle_of as hangle_of,
   angle_of_degrees as hangle_of_degrees,
   axis_ang_cen_of as haxis_ang_cen_of,
   axis_angle_hel_of as haxis_angle_hel_of,
   axis_angle_of as haxis_angle_of,
   hcoherence,
   hconstruct,
   hdiff,
   hdist,
   hdot,
   hinv,
   hmean,
   hnorm,
   hnorm2,
   hnormalized,
   hpoint,
   hpoint,
   hrot,
   htrans,
   hvec,
   line_angle,
   line_angle_degrees,
   rand_xform_small as hrand,
   rot_of,
   trans_of,
   xaxis_of,
   yaxis_of,
   zaxis_of,
   hxform,
)
