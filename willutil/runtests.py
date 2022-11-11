"""
usage: python runtests.py <projname> 

this script exists for easy editor integration
"""

import sys, os, re, argparse
from time import perf_counter
from collections import defaultdict

_post = defaultdict(lambda: "")

def get_args(sysargv):
   parser = argparse.ArgumentParser()
   parser.add_argument("--projname", help='name of project')
   parser.add_argument("testfile", type=str, default='')
   args = parser.parse_args(sysargv[1:])
   return args.__dict__

def file_has_main(fname):
   with open(fname) as inp:
      for l in inp:
         if l.startswith("if __name__ == "):
            return True
   return False

def testfile_of(path, bname, **kw):

   t = re.sub(f"^{kw['projname']}", f"{kw['projname']}/tests", path) + "/test_" + bname
   print("testfile_of", path, bname, 'is', t)
   if os.path.exists(t):
      return t

def dispatch(
      fname,
      pytest_args='--workers 8',
      file_mappings=dict(),
      overrides=dict(),
      strict=True,
      **kw,
):
   '''for the love of god... clean me up. eh, could be worse'''
   fname = os.path.relpath(fname)
   path, bname = os.path.split(fname)

   if bname in overrides:
      oride = overrides[bname]
      return oride, _post[bname]

   if fname in file_mappings:
      assert len(file_mappings[fname]) == 1
      fname = file_mappings[fname][0]
      path, bname = os.path.split(fname)
   if not strict and bname in file_mappings:
      assert len(file_mappings[bname]) == 1
      bname = file_mappings[bname][0]
      path, bname = os.path.split(bname)

   if not file_has_main(fname) and not bname.startswith("test_"):
      testfile = testfile_of(path, bname, **kw)
      if testfile:
         fname = testfile
         path, bname = os.path.split(fname)

   if not file_has_main(fname) and bname.startswith("test_"):
      cmd = "pytest {pytest_args} {fname}".format(**vars())
   elif fname.endswith(".py") and bname != 'conftest.py':
      cmd = "PYTHONPATH=. python " + fname
   else:
      cmd = "pytest {pytest_args}".format(**vars())

   return cmd, _post[bname]

def main(**kw):
   t = perf_counter()

   post = ""
   if not kw['testfile']:
      cmd = "pytest"
   else:
      if kw['testfile'].endswith(__file__):
         cmd = ""
      else:
         cmd, post = dispatch(
            kw['testfile'],
            **kw,
         )

   print("call:", sys.argv)
   print("cwd:", os.getcwd())
   print("cmd:", cmd)
   print(f"{' util/runtests.py running cmd in cwd ':=^60}")
   sys.stdout.flush()
   # if 1cmd.startswith('pytest '):
   os.putenv("NUMBA_OPT", "1")
   # os.putenv('NUMBA_DISABLE_JIT', '1')

   # print(cmd)
   os.system(cmd)

   print(f"{' main command done ':=^60}")
   os.system(post)
   t = perf_counter() - t
   print(f"{f' runtests.py done, time {t:7.3f} ':=^60}")

_overrides = {
   #    "genrate_motif_scores.py":
   #   "PYTHONPATH=. python rpxdock/app/genrate_motif_scores.py TEST"
}

_file_mappings = {
   'willutil/viz/viz_xtal.py': ['willutil/tests/homog/test_hxtal.py'],
   'willutil/viz/pymol.py': ['willutil/tests/homog/test_hxtal.py'],
   'willutil/homog/sym.py': ['willutil/tests/homog/test_homog.py'],
   'willutil/homog/quat.py': ['willutil/tests/homog/test_homog.py'],
   'willutil/homog/hgeom.py': ['willutil/tests/homog/test_homog.py'],
   "willutil/cpp/geom/expand_xforms.cpp": ["willutil/tests/cpp/geom/test_expand_xforms.py"],
   # 'willutil/pdb/pisces.py': ['willutil/tests/pdb/test_pdbmeta.py'],
   #    "rosetta.py": ["rpxdock/tests/test_body.py"],
   #   "bvh_algo.hpp": ["rpxdock/tests/bvh/test_bvh_nd.py"],
   #    "bvh.cpp": ["rpxdock/tests/bvh/test_bvh.py"],
   #    "bvh_nd.cpp": ["rpxdock/tests/bvh/test_bvh_nd.py"],
   #    "bvh.hpp": ["rpxdock/tests/bvh/test_bvh_nd.py"],
   #    # "dockspec.py": ["rpxdock/tests/search/test_multicomp.py"],
   #    "_orientations.hpp": ["rpxdock/sampling/orientations.py"],
   #    "_orientations.cpp": ["rpxdock/sampling/orientations.py"],
   #    "_orientations_test.cpp": ["rpxdock/sampling/orientations.py"],
   #    "cookie_cutter.cpp": ["rpxdock/tests/cluster/test_cluster.py"],
   #    "xbin.hpp": ["rpxdock/tests/xbin/test_xbin.py"],
   #    "xbin.cpp": ["rpxdock/tests/xbin/test_xbin.py"],
   #    "xbin_util.cpp": ["rpxdock/tests/xbin/test_xbin_util.py"],
   #    "xmap.cpp": ["rpxdock/tests/xbin/test_xmap.py"],
   #    "phmap.cpp": ["rpxdock/tests/phmap/test_phmap.py"],
   #    "phmap.hpp": ["rpxdock/tests/phmap/test_phmap.py"],
   #    "xbin_test.cpp": ["rpxdock/tests/xbin/test_xbin.py"],
   #    "_motif.cpp": ["rpxdock/motif/frames.py"],
   #    "primitive.hpp": ["rpxdock/tests/geom/test_geom.py"],
   #    "dilated_int.hpp": ["rpxdock/tests/util/test_util.py"],
   #    "dilated_int_test.cpp": ["rpxdock/tests/util/test_util.py"],
   #    "numeric.hpp": ["rpxdock/tests/xbin/test_xbin.py"],
   #    "xform_hierarchy.hpp": ["rpxdock/tests/sampling/test_xform_hierarchy.py"],
   #    "xform_hierarchy.cpp": ["rpxdock/tests/sampling/test_xform_hierarchy.py"],
   #    "miniball.cpp": ["rpxdock/tests/geom/test_geom.py"],
   #    "miniball.hpp": ["rpxdock/tests/geom/test_geom.py"],
   #    "smear.hpp": ["rpxdock/tests/xbin/test_smear.py"],
   #    "smear.cpp": ["rpxdock/tests/xbin/test_smear.py"],
   #    "bcc.hpp": ["rpxdock/tests/geom/test_bcc.py"],
   #    "bcc.cpp": ["rpxdock/tests/geom/test_bcc.py"],
   #    "pybind_types.hpp": ["rpxdock/tests/util/test_pybind_types.py"],
   #    "xform_dist.cpp": ["rpxdock/tests/geom/test_geom.py"],
   #    # "hierscore.py": ["rpxdock/tests/search/test_plug.py"],
   #    "component.py": ['rpxdock/tests/score/test_scorefunc.py'],
   #    "xform_hier.py": ['rpxdock/tests/search/test_multicomp.py'],
   #    "lattice_hier.py": ['rpxdock/tests/search/test_multicomp.py'],
   #    "basic.py": ["rpxdock/tests/search/test_onecomp.py"],
   #    "dockspec.py": ["rpxdock/tests/search/test_onecomp.py"],
   #    "pymol.py": ["rpxdock/tests/test_homog.py"],
}

if __name__ == '__main__':
   args = get_args(sys.argv)
   main(file_mappings=_file_mappings, overrides=_overrides, **args)
