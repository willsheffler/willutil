__all__ = ('MonteCarlo', 'RigidBody', 'compute_symfit', 'showme', 'halign', 'halign2', 'halign_vector', 'hangle',
           'hangle_degrees', 'hangle_of', 'hangle_of_degrees', 'haxis_ang_cen_of', 'haxis_angle_hel_of',
           'haxis_angle_of', 'hcart', 'hcart3', 'hcoherence', 'hcom', 'hcom_flat', 'hconstruct', 'hcross', 'hdiff',
           'hdist', 'hdot', 'hexpand', 'hinv', 'hline_angle', 'hmean', 'hnorm', 'hnorm2', 'hnormalized', 'hpoint',
           'hframe', 'hpow', 'hproj', 'hprojperp', 'hrand', 'hrandpoint', 'hrandrot', 'hray', 'hrmsfit', 'hrog',
           'hrog_flat', 'hrot', 'hscaled', 'htrans', 'hvalid', 'hvec', 'hxaxis_of', 'hxform', 'hyaxis_of', 'hzaxis_of',
           'line_angle', 'line_angle_degrees', 'rot_of', 't_rot', 'th_angle', 'th_axis', 'th_axis_angle',
           'th_axis_angle_cen', 'th_axis_angle_cen_hel', 'th_axis_angle_hel', 'th_com', 'th_com_flat', 'th_dot',
           'th_homog', 'th_intersect_planes', 'th_is_valid_quat_rot', 'th_mean_along', 'th_norm', 'th_norm2',
           'th_normalized', 'th_point', 'th_point_in_plane', 'th_point_line_dist2', 'th_proj', 'th_projperp',
           'th_quat_to_rot', 'th_quat_to_upper_half', 'th_quat_to_xform', 'th_rand_quat', 'th_rand_xform',
           'th_rand_xform_small', 'th_randpoint', 'th_randunit', 'th_randvec', 'th_ray_in_plane', 'th_rms', 'dssp',
           'th_rmsfit', 'th_rog', 'th_rot', 'th_rot_to_quat', 'th_vec', 'th_xform', 'unhomog')

import os
from deferred_import import deferred_import
import icecream as ic

ic.install()

datadir = os.path.join(os.path.dirname(__file__), 'data')

def datapath(f):
   return os.path.join(datadir, f)

from willutil.bunch import Bunch, bunchify, unbunchify
from willutil.timer import Timer
from willutil.sym.symfit import compute_symfit
from willutil.rigid.rigidbody import RigidBody
from willutil.search.montecarlo import MonteCarlo

from willutil.pdb.pdbdump import dump_pdb_from_ncac_points

from willutil import chem
from willutil import storage
from willutil.storage import load, save, load_package_data, open_package_data, package_data_path
from willutil.inprocess import InProcessExecutor
from willutil.cache import Cache, GLOBALCACHE
from willutil import runtests
from willutil import pdb
from willutil import search
from willutil import misc
from willutil import tests
from willutil import reproducibility
from willutil import format
from willutil import homog
from willutil import sym
from willutil import viz
from willutil import rigid
from willutil import rosetta

from willutil.chem import dssp
from willutil.rosetta import NotPose

# deferr import of cpp libs to avoid compilation if unnecessary
cpp = deferred_import('willutil.cpp')
# from willutil import cpp

# emptybunch = Bunch()

from willutil.viz import showme
# figure things in homog starting with h should't pollute namespace too bad
from willutil.sym.symfit import compute_symfit as symfit

from willutil.pdb import dumppdb, readpdb

from willutil.homog import (
   # I,
   halign,
   halign2,
   angle as hangle,
   angle_degrees as hangle_degrees,
   angle_of as hangle_of,
   angle_of_degrees as hangle_of_degrees,
   axis_ang_cen_of as haxis_ang_cen_of,
   axis_angle_hel_of as haxis_angle_hel_of,
   axis_angle_of as haxis_angle_of,
   hcart,
   hcart3,
   hcoherence,
   hcom,
   hcom_flat,
   hconstruct,
   hcross,
   hdiff,
   hdist,
   hdot,
   hexpand,
   hframe,
   hinv,
   hmean,
   hnorm,
   hnorm2,
   hnormalized,
   hpoint,
   hproj,
   hprojperp,
   hrog,
   hrog_flat,
   hrot,
   htrans,
   hvalid,
   hvec,
   hrmsfit,
   unhomog,
   hpow,
   hray,
   hscaled,
   hxform,
   line_angle as hline_angle,
   line_angle,
   line_angle_degrees,
   hrand,
   hrandrot,
   hrandpoint,
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
   th_com_flat,
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
   th_proj,
   th_projperp,
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
