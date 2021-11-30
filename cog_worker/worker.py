"""Reading COGs.

The Worker class keeps track of the region, projection, and scale to
conduct analysis in.

When writing cog_worker functions, the main method you will use is
:obj:`Worker.read()`, which is a wrapper around ``rio_tiler`` to clip,
reproject and resample the data into the target resolution.

Example:
    Read a COG, reprojecting it onto a global 1-deg lat-long grid::

        from cog_worker import Worker
        from rasterio.plot import show

        worker = Worker(bounds=(-180, -90, 180, 90), proj=4326, scale=1.0)
        arr = worker.read('example-cog-url.tif')
        show(arr)
"""
import logging
from typing import Sequence, Union

from pyproj import Proj
from pyproj.enums import TransformDirection
import rasterio as rio
from rasterio._err import CPLE_AppDefinedError
import numpy as np
from rio_tiler.errors import EmptyMosaicError
from rio_tiler.models import ImageData
from rio_tiler.io.cogeo import COGReader
from rio_tiler.mosaic.reader import mosaic_reader

from cog_worker.types import BoundingBox
from cog_worker.utils import _bbox_size, _get_profile

logger = logging.getLogger(__name__)


class Worker:
    """Class for reading Cloud Optimized GeoTIFFs."""

    def __init__(
        self,
        bounds: BoundingBox = (-180, -85, 180, 85),
        proj_bounds: BoundingBox = None,
        proj: Union[int, str, Proj] = 3857,
        scale: float = 10000,
        buffer: int = 16,
    ):
        """Initialize a Worker with a bounding box, scale, and projection.

        Args:
            bounds (BoundingBox): The region to be analyzed as a (west, south, east, north) tuple.
                Ignored when ``proj_bounds`` is provided.
            proj_bounds (BoundingBox): The region to be analyzed in the Worker's projection.
                Overrides ``bounds`` when provided.
            proj (pyproj.Proj, str, int): The projection to analyze in. See
                ``pyproj.Proj`` for valid values (https://pyproj4.github.io/pyproj/).
            scale (float): The pixel size for analysis in the projection's units (usually meters or degrees).
            buffer (int): The number of additional pixels to read on all sides to avoid edge effects.
                The ideal buffer size depends on your analysis (e.g. whether you plan to use convolutions or
                distance functions).
        """
        self._proj = (
            proj if isinstance(proj, Proj) else Proj(proj, preserve_units=False)
        )

        if proj_bounds is None:
            proj_bounds = self._proj.transform_bounds(*bounds)

        self._bounds = proj_bounds
        self._scale = scale
        self._buffer = buffer

        self._width, self._height = _bbox_size(self._bounds, scale)

    @property
    def proj(self) -> Proj:
        """The projection used for reading."""
        return self._proj

    @property
    def bounds(self) -> BoundingBox:
        """The the bounding box in projected coordinates."""
        return self._bounds

    @property
    def scale(self) -> float:
        """The size of pixels in projection units."""
        return self._scale

    @property
    def buffer(self) -> int:
        """The number of additional pixels to read on all sides of the Worker's bounding box."""
        return self._buffer

    @property
    def width(self) -> int:
        """The width of the Worker's bounding box in pixels."""
        return self._width

    @property
    def height(self) -> int:
        """The height of the Worker's bounding box in pixels."""
        return self._height

    def xy_bounds(self, buffered: bool = False) -> BoundingBox:
        """Return the Worker's bounding box in projected coordinates.

        Args:
            buffered (bool): Buffer the worker's bounding box
        """
        return self._buffer_bbox() if buffered else self.bounds

    def lnglat_bounds(self, buffered: bool = False) -> BoundingBox:
        """Return the Worker's bounding box in geographic coordinates.

        Note:
            When using a projected coordinate system, the geographic bounding box
            that covers the Worker's projected extent may be larger
            than the bounding box used to instantiate the Worker.

        Args:
            buffered (bool): Buffer the Worker's bounding box.
        """
        pts = max(self.width, self.height) + (buffered * self.buffer * 2) - 1
        bounds = self.xy_bounds(buffered)
        return self.proj.transform_bounds(
            *bounds, pts, direction=TransformDirection.INVERSE
        )

    def empty(self, mask: bool = False) -> np.ndarray:
        """Return a zeroed array covering the Worker's extent including the buffer.

        Args:
            mask (bool): Return a Numpy masked array with all pixels masked.
                Otherwise returns a standard Numpy array filled with zeros.
        """
        arr = np.zeros((1, self.height + self.buffer * 2, self.width + self.buffer * 2))
        if mask:
            _mask = np.ones(
                (self.height + self.buffer * 2, self.width + self.buffer * 2)
            )
            arr = np.ma.array(arr, mask=_mask)
        return arr

    def read(self, src: Union[str, Sequence[str]], **kwargs) -> np.ma.MaskedArray:
        """Read a COG, reprojecting and clipping as necessary.

        The read method uses ``rio_tiler.COGReader`` to takes advantage of the
        file structure and internal overviews in COGs, minimizing the amount of
        data that needs to be read and transferred when working at reduced resolutions.

        In general, any valid GDAL path can be read. This may be a url pointing to a COG, a local
        GeoTIFF or a GDAL virtual file system path. However, it may be very inefficient to
        read data sources that are not valid Cloud Optimized GeoTIFFs.

        If a list of data sources is provided, ``Worker.read`` will use ``rio_tiler.mosaic_reader``
        to mosaic the sources together.

        Note:
            The resampling method used to generate the COG's internal overviews will affect
            how it appears at reduced resolutions.

        Args:
            src (str, list): The data source to read or list of sources to mosiac.
            **kwargs: Additional keyword arguments to pass to ``rio_tiler.COGReader.part``
                or ``rio_tiler.mosaic_reader``. See: https://cogeotiff.github.io/rio-tiler/.

        Returns:
            A Numpy masked array containing the data for the Worker's bounding box and its
            buffer.

        Note:
            The mask values of a Numpy masked array is the inverse of a GDAL (alpha) mask.
            A masked value of True corresponds to nodata or an alpha value of 0.
        """
        proj_bounds = self._buffer_bbox()
        width, height = _bbox_size(proj_bounds, self._scale)

        if isinstance(src, str):
            img = _read_COG(src, proj_bounds, self._proj.crs, width, height, **kwargs)
        elif isinstance(src, Sequence):
            try:
                img, asset = mosaic_reader(
                    src, _read_COG, proj_bounds, self.proj.crs, width, height, **kwargs
                )
            except EmptyMosaicError:
                return self.empty(mask=True)

        arr = img.data
        mask = (img.mask == 0) | np.isnan(arr)

        return np.ma.array(arr, mask=mask)

    def write(self, arr: np.ndarray, dst: str, **kwargs):
        """Write a Numpy array to a GeoTIFF.

        The write method will create a GeoTIFF with a profile matching the Worker's properties.
        Uses ``rasterio.open`` under the hood.

        Args:
            arr (numpy.ndarray): The array to write. Must be 2 or 3-dimensional, with a width and
                height matching the Worker (including or excluding the buffer). If the array
                includes the Worker's buffer, the buffer will be clipped before writing.
            dst (str): The file path to write to.
            **kwargs: Additional keyword arguments to pass to rasterio.open
                See: https://rasterio.readthedocs.io/en/latest/topics/writing.html
        """
        arr = self.clip_buffer(arr)
        count, height, width = arr.shape
        profile = _get_profile(
            count, self.scale, self._bounds, self.proj, arr.dtype, **kwargs
        )

        with rio.open(dst, "w", **profile) as writer:
            writer.write(arr)
            if isinstance(arr, np.ma.MaskedArray):
                mask = np.ma.getmask(arr)
                if len(mask.shape) == 3:
                    mask = np.any(mask, axis=0)
                writer.write_mask(~mask)

    def clip_buffer(self, arr: np.ndarray) -> np.ndarray:
        """Clip the buffer pixels from an array if they exist.

        Args:
            arr (numpy.ndarray): The array to clip.

        Returns:
            The array with buffer pixels removed.

        Raises:
            ValueError: If the array's shape does not match the Worker's width and height
        """
        if len(arr.shape) == 2:
            arr = arr[np.newaxis]
        buffer_width = self.width + self.buffer * 2
        buffer_height = self.height + self.buffer * 2
        c, h, w = arr.shape
        if w == self.width and h == self.height:
            return arr
        elif w == buffer_width and h == buffer_height:
            return arr[:, self.buffer : -self.buffer, self.buffer : -self.buffer]
        else:
            raise ValueError(
                f"Array not expected size. Was {w}x{h} expected {self.width}x{self.height} or {buffer_width}x{buffer_height}"
            )

    def _buffer_bbox(self) -> BoundingBox:
        """Returns the worker's bounding box extended by the buffered pixels."""
        l, b, r, t = self._bounds
        _buffer = self.buffer * self.scale

        return (l - _buffer, b - _buffer, r + _buffer, t + _buffer)


def _read_COG(
    asset: str,
    proj_bounds: BoundingBox,
    crs: Union[str, int, Proj],
    width: int,
    height: int,
    **kwargs,
) -> ImageData:
    """Read part of a COG, warping and resampling to a target shape."""
    tries = 6
    while tries > 0:
        try:
            with COGReader(asset, **kwargs) as cog:  # type: ignore
                return cog.part(
                    proj_bounds,
                    bounds_crs=crs,
                    dst_crs=crs,
                    max_size=None,
                    width=width,
                    height=height,
                )
        except CPLE_AppDefinedError as e:
            # Ignore some strange GDAL errors when reading in some projections
            # see: https://rasterio.groups.io/g/main/message/780
            logger.debug(e)
            if tries <= 1:
                raise e
        tries -= 1
    raise Exception(f"Failed reading asset {asset}")
