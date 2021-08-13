import pytest
import rasterio as rio
from rasterio.io import MemoryFile

from cog_worker import Worker


TEST_COG = "tests/roads_cog.tif"


@pytest.fixture
def mercator_worker():
    return Worker(scale=50000)


def test_xy_bounds(mercator_worker):
    assert mercator_worker.xy_bounds() == (
        -20037508.342789244,
        -19971868.88040857,
        20037508.342789244,
        19971868.88040857,
    )
    assert mercator_worker.xy_bounds(True) == (
        -20837508.342789244,
        -20771868.88040857,
        20837508.342789244,
        20771868.88040857,
    )


def test_latlng_bounds(mercator_worker):
    assert mercator_worker.lnglat_bounds() == (-180, -85, 180, 85)
    for x, y in zip(
        mercator_worker.lnglat_bounds(True),
        (-180.0, -85.58878478755108, 180.0, 85.58878478755108),
    ):
        assert pytest.approx(x, 0.001) == y


def test_empty(mercator_worker):
    assert mercator_worker.empty().shape == (1, 831, 834)


def test_read(mercator_worker):
    arr = mercator_worker.read(TEST_COG)
    assert arr.shape == mercator_worker.empty().shape
    assert arr.sum() == 20206

    mosaic_arr = mercator_worker.read([TEST_COG, TEST_COG])
    assert (mosaic_arr == arr).all()


def test_write(mercator_worker):
    arr = mercator_worker.read(TEST_COG)
    with MemoryFile() as memfile:
        mercator_worker.write(arr, memfile)
        memfile.seek(0)
        with rio.open(memfile) as src:
            written = src.read(masked=True)
            assert (written.shape[1], written.shape[2]) == (
                mercator_worker.height,
                mercator_worker.width,
            )
            assert written.sum() == mercator_worker.clip_buffer(arr).sum()


def test_clip_buffer(mercator_worker):
    arr = mercator_worker.empty()
    arr = mercator_worker.clip_buffer(arr)
    arr = mercator_worker.clip_buffer(arr)
    assert (arr.shape[1], arr.shape[2]) == (
        mercator_worker.height,
        mercator_worker.width,
    )


def test__buffer_bbox(mercator_worker):
    assert mercator_worker._buffer_bbox() == (
        -20837508.342789244,
        -20771868.88040857,
        20837508.342789244,
        20771868.88040857,
    )