import pytest
import rasterio as rio
from cog_worker import Manager
from cog_worker.distributed import DaskManager
from dask.distributed import Client, LocalCluster
from rasterio import MemoryFile

TEST_COG = "tests/roads_cog.tif"


@pytest.fixture
def cluster():
    with LocalCluster(n_workers=1) as cluster:
        yield cluster


@pytest.fixture
def daskmanager(cluster):
    client = Client(cluster)
    return DaskManager(client, scale=64000)


@pytest.fixture
def manager():
    return Manager(scale=64000)


@pytest.fixture
def sample_function():
    def myfunc(worker):
        return worker.read(TEST_COG)

    return myfunc


def test_execute(daskmanager, manager, sample_function):
    assert (daskmanager.execute(sample_function)[0] == manager.execute(sample_function)[0]).all()


def test_chunk_execute(daskmanager, sample_function):
    results = daskmanager.client.gather(daskmanager.chunk_execute(sample_function, compute=False))
    assert all(isinstance(results[1], tuple) for results in results)


def test_chunck_execute_compute(daskmanager, sample_function):
    results = daskmanager.chunk_execute(sample_function, compute=True)
    assert all(isinstance(results[1], tuple) for results in results)


def test_chunk_save(daskmanager, manager, sample_function):
    with MemoryFile() as m1, MemoryFile() as m2:
        daskmanager.chunk_save(m1, sample_function)
        manager.chunk_save(m2, sample_function)
        m1.seek(0)
        m2.seek(0)
        with rio.open(m1) as src1, rio.open(m2) as src2:
            assert (src1.read() == src2.read()).all()
