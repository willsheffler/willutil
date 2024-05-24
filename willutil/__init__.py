from deferred_import import deferred_import
import icecream as ic

from willutil.bunch import Bunch, bunchify, unbunchify
from willutil.timer import Timer, timed, checkpoint

from willutil.rigid.rigidbody import RigidBody, RigidBodyFollowers
from willutil.search.montecarlo import MonteCarlo

from willutil.pdb.pdbdump import dump_pdb_from_ncac_points

from willutil import chem
from willutil import storage
from willutil.storage import load, save, load_package_data, open_package_data, package_data_path
from willutil.inprocess import InProcessExecutor
from willutil.cache import Cache, GLOBALCACHE
from willutil import runtests
from willutil import pdb
from willutil import sampling
from willutil import search
from willutil import misc
from willutil import tests
# from willutil import reproducibility
# from willutil import format
from willutil import homog
from willutil import sym
# from willutil import viz
viz = deferred_import('willutil.viz')
from willutil import rigid
from willutil import rosetta
# from willutil import unsym
from willutil.chem import dssp
from willutil.rosetta import NotPose

from willutil.tests import test_data_path

from willutil.misc import WARNME, datetag, datetimetag, UnhashableSet, printheader

from willutil.homog import thgeom as h

# deferr import of cpp libs to avoid compilation if unnecessary
cpp = deferred_import('willutil.cpp')
# from willutil import cpp

# emptybunch = Bunch()

# figure things in homog starting with h should't pollute namespace too bad
from willutil.sym.symfit import compute_symfit

from willutil.pdb import dumppdb, readpdb, readfile, readcif, dumpstruct, dumpcif

from willutil.homog import (
   # I,
   isarray,
   halign,
   halign2,
   angle as hangle,
   angle_degrees as hangle_degrees,
   angle_of as hangle_of,
   angle_of_degrees as hangle_of_degrees,
   axis_ang_cen_of as haxis_ang_cen_of,
   axis_angle_hel_of as haxis_angle_hel_of,
   axis_angle_cen_hel_of as haxis_angle_cen_hel_of,
   axis_angle_of as haxis_angle_of,
   hcart,
   hcart3,
   hori3,
   hcoherence,
   line_angle as hangline,
   hcentered,
   hcentered3,
   hcom,
   hcom_flat,
   hconstruct,
   hcross,
   hdiff,
   hconvert,
   dihedral as hdihedral,
   hdist,
   hdot,
   hexpand,
   hframe,
   hinv,
   hmean,
   hnorm,
   hnorm2,
   hnormalized,
   hparallel,
   axis_of as haxisof,
   hpoint,
   hproj,
   hprojperp,
   hrog,
   hrog_flat,
   hrot,
   htrans,
   hvalid,
   hvalid_norm,
   hvec,
   hrmsfit,
   unhomog,
   hpow,
   hray,
   hscaled,
   hxform,
   hxformpts,
   hxformvec,
   hxformx,
   h_point_line_dist as hpointlinedis,
   hlinesisect,
   hunique,
   hpointlineclose,
   line_angle as hline_angle,
   line_angle,
   line_angle_degrees,
   hrand,
   hrandsmall,
   hrandrot,
   hrandpoint,
   rand_vec as hrandvec,
   rand_unit as hrandunit,
   rot_of,
   trans_of,
   xaxis_of,
   yaxis_of,
   zaxis_of,
   to_xyz,
)

ic.install()

# from willutil.viz import showme
def showme(*a, **kw):
   from willutil.viz import showme as viz_showme
   viz_showme(*a, **kw)

#__all__ = ('MonteCarlo', 'RigidBody', 'compute_symfit', 'dssp', 'halign', 'halign2', 'halign_vector', 'hangle',
#           'hangle_degrees', 'hangle_of', 'hangle_of_degrees', 'hangline', 'haxis_ang_cen_of', 'haxis_ang_cen_hel_of',
#           'haxis_angle_hel_of', 'haxis_angle_of', 'haxisof', 'hcart', 'hcart3', 'hcoherence', 'hcom', 'hcom_flat',
#           'hconstruct', 'hcross', 'hdiff', 'hdihedral', 'hdist', 'hdot', 'hexpand', 'hframe', 'hinv', 'hline_angle',
#           'hmean', 'hnorm', 'hnorm2', 'hnormalized', 'hori3', 'hparallel', 'hpoint', 'hpointlineclose',
#           'hpointlinedis', 'hpow', 'hproj', 'hprojperp', 'hrand', 'hrandsmall', 'hrandpoint', 'hrandrot', 'hrandvec',
#           'hrandunit', 'hray', 'hrmsfit', 'hrog', 'hrog_flat', 'hrot', 'hscaled', 'htrans', 'hvalid', 'hvec',
#           'hxaxis_of', 'hxform', 'hxformpts', 'hxformvec', 'hyaxis_of', 'hzaxis_of', 'line_angle',
#           'line_angle_degrees', 'rot_of', 'showme', 't_rot', 'thangle', 'thaxis', 'thaxis_angle', 'thaxis_angle_cen',
#           'thaxis_angle_cen_hel', 'thaxis_angle_hel', 'thcom', 'thcom_flat', 'thdot', 'thhomog', 'thintersect_planes',
#           'this_valid_quat_rot', 'thmean_along', 'thnorm', 'thnorm2', 'thnormalized', 'thpoint', 'thpoint_in_plane',
#           'thpoint_line_dist2', 'thproj', 'thprojperp', 'thquat_to_rot', 'thquat_to_upper_half', 'thquat_to_xform',
#           'thrand_quat', 'thrand_xform', 'thrand_xform_small', 'thrandpoint', 'thrandunit', 'thrandvec',
#           'thray_in_plane', 'thrms', 'thrmsfit', 'throg', 'throt', 'throt_to_quat', 'thvec', 'thxform', 'thxformpts',
#           'to_xyz', 'unhomog', 'hcentered', 'hcentered3', 'hunique', 'hconvert', 'WARNME', 'datetag', 'datetimetag',
#           'pkgdata', 'isarray', 'hxformx', 'hlinesisect', 'UnhashableSet', 'hvalid_norm', 'thconstruct')
