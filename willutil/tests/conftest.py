import pytest
import willutil.tests.testdata as tdat

@pytest.fixture(scope='session')
def respairdat10():
   return tdat.respairdat10()

@pytest.fixture(scope='session')
def respairdat10_plus_xmap_rots():
   return tdat.respairdat10_plus_xmap_rots()
