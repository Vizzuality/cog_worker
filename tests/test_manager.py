import pytest
import rasterio as rio
from rasterio.io import DatasetWriter
from cog_worker import Manager
from rasterio import MemoryFile, crs

TEST_COG = "tests/roads_cog.tif"


@pytest.fixture
def molleweide_manager():
    return Manager(
        proj="+proj=moll",
        scale=50000,
    )


@pytest.fixture
def sample_function():
    def myfunc(worker):
        return worker.read(TEST_COG)

    return myfunc


def test_preview(molleweide_manager, sample_function):
    arr, bbox = molleweide_manager.preview(sample_function, max_size=123)
    assert max(arr.shape) == 123, "Expected maximum array dimension to be 123px"


def test_tile(molleweide_manager, sample_function):
    arr, bbox = molleweide_manager.tile(sample_function, x=1, y=2, z=3)
    assert arr.shape == (1, 256, 256), "Expected 256x256 tile"


def test_chunk_execute(molleweide_manager, sample_function):
    chunks = list(molleweide_manager.chunk_execute(sample_function, chunksize=123))
    for arr, bbox in chunks:
        assert max(arr.shape) <= 123, "Max chunk size should be 123px"


def test_chunk_params(molleweide_manager):
    chunks = list(molleweide_manager.chunk_params(chunksize=123))
    assert len(chunks) == 18, "Expected ~18 chunks for 123px tiles at 50km scale"


def test__open_writer(molleweide_manager):
    with MemoryFile() as memfile:
        with molleweide_manager._open_writer(memfile, 1, rio.ubyte) as writer:
            assert isinstance(writer, DatasetWriter)


def test_chunk_save(molleweide_manager, sample_function):
    full_arr = molleweide_manager.execute(sample_function)[0]
    with MemoryFile() as memfile:
        molleweide_manager.chunk_save(memfile, sample_function)
        memfile.seek(0)
        with rio.open(memfile) as src:
            assert src.profile["crs"] == crs.CRS.from_string("+proj=moll")
            assert src.profile["transform"][0] == 50000
            arr = src.read()
            assert arr.shape == full_arr.shape
            assert (
                abs(arr.sum() / full_arr.data.sum() - 1) < 0.002
            ), "Error should be less than 0.2%"


def test__write_chunk(molleweide_manager, sample_function):
    with MemoryFile() as memfile:
        arr, bbox = molleweide_manager.execute(sample_function)
        print(arr.mask.sum())
        with molleweide_manager._open_writer(memfile, 1, rio.ubyte) as writer:
            molleweide_manager._write_chunk(writer, arr, bbox)
        memfile.seek(0)
        with rio.open(memfile) as src:
            written = src.read(masked=True)
            assert (written == arr).all()
            assert (written.mask == arr.mask).all()


def test__chunk_bounds(molleweide_manager):
    chunk = molleweide_manager._chunk_bounds(0, 0, 123)
    assert chunk == (
        -18040095.696147293,
        2674978.852256801,
        -11890095.696147293,
        8824978.852256801,
    )


def test__num_chunks(molleweide_manager):
    assert molleweide_manager._num_chunks(123) == (6, 3)
