from willutil.tests.testdata import *
from willutil.tests import fixtures

def force_pytest_skip(reason):
   import _pytest
   raise _pytest.outcomes.Skipped(reason)
