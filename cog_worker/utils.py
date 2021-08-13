"""Utility functions."""
from typing import Tuple, Type, Union

from rasterio import transform
from pyproj import Proj
import numpy as np

from .types import BoundingBox


def _get_profile(
    count: int,
    scale: float,
    proj_bounds: BoundingBox,
    proj: Proj,
    dtype: Union[Type, np.dtype],
    **kwargs
) -> dict:
    width, height = _bbox_size(proj_bounds, scale)
    affine = transform.from_origin(proj_bounds[0], proj_bounds[3], scale, scale)
    profile = {
        "driver": "GTiff",
        "interleave": "pixel",
        "blockxsize": 512,
        "blockysize": 512,
        "tiled": True,
        "compress": "lzw",
        "crs": proj.srs,
        "transform": affine,
        "dtype": dtype,
        "width": width,
        "height": height,
        "count": count,
    }
    profile.update(kwargs)
    return profile


def _bbox_size(
    bounds: BoundingBox,
    scale: float,
) -> Tuple[int, int]:
    left, bottom, right, top = bounds
    return (round((right - left) / scale), round((top - bottom) / scale))
