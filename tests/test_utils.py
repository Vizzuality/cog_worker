import rasterio as rio
from pyproj import Proj
from rasterio.io import MemoryFile
from cog_worker import utils


def test__get_profile():
    proj = Proj("wgs84")
    profile = utils._get_profile(
        3, 0.0833333, (-180, -90, 180, 90), proj, rio.uint8, nodata=0
    )
    with MemoryFile() as memfile:
        with rio.open(memfile, "w", **profile) as dst:
            assert Proj(dst.profile["crs"]) == proj
            assert dst.profile["dtype"] == profile["dtype"]
            assert dst.transform == profile["transform"]


def test__bbox_size():
    assert utils._bbox_size((-180, -90, 180, 90), 0.1666666) == (2160, 1080)
