import pytest
import willutil.tests.testdata as tdat

@pytest.fixture(scope='session')
def respairdat10():
   return tdat.respairdat10()

@pytest.fixture(scope='session')
def respairdat10_plus_xmap_rots():
   return tdat.respairdat10_plus_xmap_rots()

@pytest.fixture
def pdbfile():
   return tdat.pdbfile()

@pytest.fixture
def pdbfiles():
   return tdat.pdbfiles()

@pytest.fixture
def pdbcontents():
   return tdat.pdbcontents()
