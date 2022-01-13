import pytest
from willutil.tests import fixtures

@pytest.fixture(scope='session')
def respairdat10():
    return fixtures.respairdat10()

@pytest.fixture(scope='session')
def respairdat10_plus_xmap_rots():
    return fixtures.respairdat10_plus_xmap_rots()

@pytest.fixture
def pdbfname():
    return fixtures.pdbfname()

@pytest.fixture
def pdbfnames():
    return fixtures.pdbfnames()

@pytest.fixture
def pdbcontents():
    return fixtures.pdbcontents()

@pytest.fixture
def pdbcontents():
    return fixtures.pdbcontents()

@pytest.fixture
def three_PDBFiles():
    return fixtures.three_PDBFiles()

@pytest.fixture
def pdbfile():
    return fixtures.pdbfile()
