import pytest

# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'


def pytest_collection_modifyitems(session, config, items):
    items[:] = [item for item in items if item.name != 'test_data_path']


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


@pytest.fixture
def ncac():
    return fixtures.ncac()


@pytest.fixture
def ncaco():
    return fixtures.ncaco()


@pytest.fixture
def pdb1pgx():
    return fixtures.pdb1pgx()


@pytest.fixture
def pdb1coi():
    return fixtures.pdb1coi()


@pytest.fixture
def pdb1qys():
    return fixtures.pdb1qys()
