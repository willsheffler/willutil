from cppimport import import_hook

from willutil.cpp.geom.bcc import *
from willutil.cpp.geom.miniball import *
from willutil.cpp.geom.xform_dist import *
from willutil.cpp.geom.expand_xforms import *
from willutil.cpp.geom import bcc, miniball

def xform_dist2(*args):
   c, o = xform_dist2_split(*args)
   return c + o