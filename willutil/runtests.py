"""
usage: python runtests.py <projname> 

this script exists for easy editor integration
"""

import sys, os, re, argparse
from time import perf_counter
from collections import defaultdict

_post = defaultdict(lambda: "")

parser = argparse.ArgumentParser()
parser.add_argument("--projname", help='name of project')
parser.add_argument("testfile", type=str, default='')
args = parser.parse_args(sys.argv[1:])

def file_has_main(file):
   with open(file) as inp:
      for l in inp:
         if l.startswith("if __name__ == "):
            return True
   return False

def testfile_of(path, bname):

   t = re.sub(f"^{args.projname}", f"{args.projname}/tests", path) + "/test_" + bname
   print("testfile_of", path, bname, 'is', t)
   if os.path.exists(t):
      return t

def dispatch(
      file,
      pytest_args='',
      file_mappings=dict(),
      overrides=dict(),
):
   """for the love of god... clean me up"""
   file = os.path.relpath(file)
   path, bname = os.path.split(file)

   if bname in overrides:
      oride = overrides[bname]
      return oride, _post[bname]

   print("runtests.py dispatch", path, bname)
   if bname in file_mappings:
      if len(file_mappings[bname]) == 1:
         file = file_mappings[bname][0]
         path, bname = os.path.split(file)
      else:
         assert 0

   if not file_has_main(file) and not bname.startswith("test_"):
      testfile = testfile_of(path, bname)
      if testfile:
         file = testfile
         path, bname = os.path.split(file)

   if not file_has_main(file) and bname.startswith("test_"):
      cmd = "pytest {pytest_args} {file}".format(**vars())
   elif file.endswith(".py"):
      cmd = "PYTHONPATH=. python " + file
   else:
      cmd = "pytest {pytest_args}".format(**vars())

   return cmd, _post[bname]

def main(file_mappings, overrides):
   t = perf_counter()

   post = ""
   if not args.testfile:
      cmd = "pytest"
   else:
      if args.testfile.endswith(__file__):
         cmd = ""
      else:
         cmd, post = dispatch(
            args.testfile,
            file_mappings=_file_mappings,
            overrides=_overrides,
         )

   print("call:", sys.argv)
   print("cwd:", os.getcwd())
   print("cmd:", cmd)
   print(f"{' util/runtests.py running cmd in cwd ':=^80}")
   sys.stdout.flush()
   # if 1cmd.startswith('pytest '):
   os.putenv("NUMBA_OPT", "1")
   # os.putenv('NUMBA_DISABLE_JIT', '1')

   # print(cmd)
   os.system(cmd)

   print(f"{' main command done ':=^80}")
   os.system(post)
   t = perf_counter() - t
   print(f"{f' runtests.py done, time {t:7.3f} ':=^80}")

_overrides = {
   #    "genrate_motif_scores.py":
   #   "PYTHONPATH=. python rpxdock/app/genrate_motif_scores.py TEST"
}

_file_mappings = {
   'sym.py': ['willutil/tests/test_homog.py'],
   'quat.py': ['willutil/tests/test_homog.py'],
   'hgeom.py': ['willutil/tests/test_homog.py'],
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
   main(_file_mappings, _overrides)
