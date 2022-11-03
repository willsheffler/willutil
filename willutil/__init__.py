from deferred_import import deferred_import
import icecream

icecream.install()

from willutil.bunch import Bunch, bunchify, unbunchify
from willutil.timer import Timer
from willutil.ping import PING
from willutil import chem
from willutil import storage
from willutil.storage import load, save, load_package_data, open_package_data, package_data_path
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
   angle_degrees as hangle_degrees,
   angle_of as hangle_of,
   angle_of_degrees as hangle_of_degrees,
   axis_ang_cen_of as haxis_ang_cen_of,
   axis_angle_hel_of as haxis_angle_hel_of,
   axis_angle_of as haxis_angle_of,
   hcoherence,
   hcom,
   hcom_flat,
   hconstruct,
   hcross,
   hdiff,
   hdist,
   hdot,
   hexpand,
   hinv,
   hmean,
   hnorm,
   hnorm2,
   hnormalized,
   hpoint,
   hpoint,
   hrog,
   hrog_flat,
   hrot,
   htrans,
   hvec,
   hpow,
   hray,
   hscale,
   hxform,
   line_angle as hline_angle,
   line_angle,
   line_angle_degrees,
   rand_xform_small as hrand,
   rand_rot_small as hrandrot,
   rot_of,
   trans_of,
   xaxis_of,
   yaxis_of,
   zaxis_of,
)
from willutil.homog.thgeom import (
   t_rot,
   th_angle,
   th_axis,
   th_axis_angle_cen,
   th_axis_angle_cen_hel,
   th_axis_angle,
   th_axis_angle_hel,
   th_com,
   th_dot,
   th_homog,
   th_intersect_planes,
   th_is_valid_quat_rot,
   th_mean_along,
   th_point_line_dist2,
   th_norm,
   th_norm2,
   th_normalized,
   th_point,
   th_point_in_plane,
   th_quat_to_rot,
   th_quat_to_upper_half,
   th_quat_to_xform,
   th_rand_quat,
   th_rand_xform,
   th_rand_xform_small,
   th_randpoint,
   th_randunit,
   th_randvec,
   th_ray_in_plane,
   th_rms,
   th_rmsfit,
   th_rog,
   th_rot,
   th_rot_to_quat,
   th_vec,
   th_xform,
)