"""Previewing, chunking, and executing analysis.

The Manager class is used to divide an area of analysis into chunks of manageable size,
and execute functions on each of these chunks.

When executing functions, the Manager instantiates a :obj:`cog_worker.worker.Worker` and passes
it to the function as its first parameter. The Worker keeps track of the scale, projection,
and bounds of its piece of the analysis, which it uses to handle the reading and writing of
Cloud Optimized GeoTIFFs.

Example:
    Use the manager to preview an analysis before executing it::

        from cog_worker import Manager
        from rasterio.plot import show
        def my_analysis(worker):
            arr = worker.read('example-cog.tif')
            # calculations ...
            return arr

        manager = Manager()
        arr, bbox = manager.preview(my_analysis)
        show(arr)

    Execute the analysis in chunks, saving the results to disk::

        manager.chuck_save('output.tif', myanalysis):

"""
import math
import logging
from typing import IO, Iterable, Iterator, Mapping, Optional, Tuple, Type, Union, Any


import numpy as np
import morecantile
from pyproj import Proj
import rasterio as rio
from rasterio.io import DatasetWriter
import rasterio.windows

import cog_worker.worker
from .utils import _bbox_size, _get_profile
from .types import WorkerFunction, BoundingBox

logger = logging.getLogger(__name__)


class Manager:
    """Class for managing scalable analysis of Cloud Optimized GeoTIFFs."""

    def __init__(
        self,
        bounds: BoundingBox = (-180, -85, 180, 85),
        proj: Union[int, str, Proj] = 3857,
        scale: float = 10000,
        buffer: int = 16,
    ):
        """Initialize a Manager with a projection, scale, and bounding box for analysis.

        Args:
            bounds (BoundingBox): The region to be analyzed as a (west, south,
                east, north) tuple.
            proj (pyproj.Proj, str, int): The projection to analyze in.
                Generally accepts any proj4 string, WKT projection, or EPSG
                code. See pyproj.Proj for valid values.
            scale (float): The pixel size for analysis in the projection's units
                (usually meters or degrees).
            buffer (int): When dividing analysis into chunks, the number of additional pixels
                to read on all sides to avoid edge effects. The ideal buffer size depends on
                your analysis (e.g. whether you use convolutions or distance functions).
        """
        self.proj = proj if isinstance(proj, Proj) else Proj(proj, preserve_units=False)
        self.bounds = bounds
        self.scale = scale
        self.buffer = buffer
        self._proj_bounds = self.proj.transform_bounds(*bounds)
        self.tms = morecantile.TileMatrixSet.custom(
            list(self._proj_bounds), self.proj.crs
        )

    def execute(
        self,
        f: WorkerFunction,
        f_args: Iterable = None,
        f_kwargs: Mapping = None,
        clip: bool = True,
        **kwargs
    ) -> Tuple[Any, BoundingBox]:
        """Execute a function that takes a cog_worker.worker.Worker as its first parameter.

        The execute method is the underlying method for running analysis. By default, it
        will run the function once for the Manager's given scale and bounding box.

        When executing functions, the Manager instantiates a cog_worker.worker.Worker and passes
        it to the function as its first parameter. The Worker keeps track of the scale, projection,
        and bounds of its piece of the analysis, which it uses to handle the reading and writing of
        Cloud Optimized GeoTIFFs.

        Args:
            f (:obj:`cog_worker.types.WorkerFunction`): The function to execute. The function will recieve a
                cog_worker.worker.Worker as its first argument.
            f_args (list): Additional arguments to pass to the function.
            f_kwargs (dict): Additional keyword arguments to pass to the function.
            clip (bool): Whether or not to clip the buffer from the completed analysis.
            **kwargs: Additional keyword arguments to overload the Manager's properties.
                (bounds, proj, scale, or buffer)

        Returns:
            A tuple containing the return value of the function and the bounding
            box of the executed analysis in the target projection.
        """
        args = {
            "bounds": self.bounds,
            "proj": self.proj,
            "scale": self.scale,
            "buffer": self.buffer,
        }
        args.update(kwargs)
        return _execute(f, f_args, f_kwargs, clip, **args)

    def preview(
        self,
        f: WorkerFunction,
        f_args: Iterable = None,
        f_kwargs: Mapping = None,
        bounds: Optional[BoundingBox] = None,
        max_size: int = 1024,
        **kwargs
    ) -> Tuple[Any, BoundingBox]:
        """Preview a function by executing it at a reduced scale.

        The preview method automatically reduces the scale of analysis to fit within `max_size`.

        Args:
            f (WorkerFunction): The function to execute. The function will
                recieve a cog_worker.worker.Worker as its first argument.
            f_args (list): Additional arguments to pass to the function.
            f_kwargs (dict): Additional keyword arguments to pass to the function.
            bounds (BoundingBox, default: self.bounds): The region to analize.
            max_size (int): The maximum size (width or height) in pixels to
                compute, ignoring any buffer (default: 1024px).
            **kwargs: Additional keyword arguments to overload the Manager's properties.
                (proj or buffer).

        Returns:
            A tuple containing the return value of the function and the bounding
            box of the executed analysis in the target projection.
        """
        bounds = self.bounds if bounds is None else bounds
        proj = kwargs.pop("proj", self.proj)
        proj = proj if isinstance(proj, Proj) else Proj(proj, preserve_units=False)
        proj_bounds = self.proj.transform_bounds(*bounds)
        width, height = _bbox_size(proj_bounds, self.scale)
        _size = max(width, height)
        scale = self.scale * _size / max_size

        kwargs.update({"proj_bounds": proj_bounds, "proj": proj, "scale": scale})

        return self.execute(f, f_args, f_kwargs, **kwargs)

    def tile(
        self,
        f: WorkerFunction,
        f_args: Iterable = None,
        f_kwargs: Mapping = None,
        z: int = 0,
        x: int = 0,
        y: int = 0,
        tilesize: int = 256,
        **kwargs
    ) -> Tuple[Any, BoundingBox]:
        """Execute a function for the scale and bounds of a TMS tile.

        The tile method supports non-global and non-mercator tiling schemes via
        Morecantile. To generate standard web tiles, instantiate the Manager
        with the default parameters.

        Args:
            f (:obj:`cog_worker.types.WorkerFunction`): The function to execute. The function will
                recieve a cog_worker.worker.Worker as its first argument.
            f_args (list): Additional arguments to pass to the function.
            f_kwargs (dict): Additional keyword arguments to pass to the function.
            bounds (BoundingBox): The region to analize (default: self.bounds)
            max_size (int): The maximum size (width or height) in pixels to compute, ignoring any buffer
                (default: 1024px). Automatically reduces the scale of analysis to fit within `max_size`.
            **kwargs: Additional keyword arguments to overload the Manager's properties.
                (buffer).

        Returns:
            A tuple containing the return value of the function and the bounding
            box of the executed analysis in the target projection.
        """
        proj_bounds = self.tms.xy_bounds(x, y, z)  # type: ignore
        left, bottom, right, top = proj_bounds
        size = max(right - left, top - bottom)
        scale = size / tilesize

        kwargs.update(
            {
                "proj_bounds": proj_bounds,
                "scale": scale,
            }
        )

        return self.execute(f, f_args, f_kwargs, **kwargs)

    def chunk_execute(
        self,
        f: WorkerFunction,
        f_args: Iterable = None,
        f_kwargs: Mapping = None,
        chunksize: int = 512,
    ) -> Iterator[Tuple[Any, BoundingBox]]:
        """Return a generator that executes a function on chunks of at most `chunksize` pixels.

        Note:
            Manager.chunk_execute computes each chunk sequentially, trading time for reduced memory footprint.
            To run large scale analysis in parallel using dask, see cog_worker.distributed.

        Note:
            You can estimate the memory requirement of executing a function at a given chunksize as
            ``(chunksize + 2*buffer)**2 * number_of_bands_or_arrays * bit_depth``.

        Args:
            f (:obj:`cog_worker.types.WorkerFunction`): The function to execute. The function will recieve a
                cog_worker.worker.Worker as its first argument.
            f_args (list): Additional arguments to pass to the function.
            f_kwargs (dict): Additional keyword arguments to pass to the function.
            chunksize (int): Size of the chunks in pixels (excluding buffer).

        Yields:
            A tuple containing the return value of the function and the bounding
            box of the executed analysis in the target projection.
        """
        for params in self.chunk_params(chunksize):
            yield self.execute(f, f_args, f_kwargs, **params)

    def chunk_save(
        self,
        dst: Union[str, IO],
        f: WorkerFunction,
        f_args: Iterable = None,
        f_kwargs: Mapping = None,
        chunksize: int = 512,
        **kwargs
    ):
        """Execute a function in chunks and write each chunk to disk as it is completed.

        The chunk_save method is identical to Manager.chunk_execute, except it writes results to ``dst``
        instead of yielding them. Manager.chunk_save uses the rasterio GeoTiff driver.

        Note:
            The function to be executed will recieve a cog_worker.worker.Worker as its first argument and
            should return a 3-dimensional numpy array of ``chunksize`` (optionally plus the buffer pixels).
            e.g.::

                # Read a cog in chunks and write those chunks to 'test.tif'
                manager.chunk_save('test.tif', lambda worker: worker.read('example-cog-url.tif'))

        Args:
            dst (str): The file path to write to.
            f (:obj:`cog_worker.types.WorkerFunction`): The function to execute.
                The function will recieve a cog_worker.worker.Worker as its first argument
                and must return a 3-dimensional numpy array of ``chunksize`` (including or excluding the buffer).
            f_args (list): Additional arguments to pass to the function.
            f_kwargs (dict): Additional keyword arguments to pass to the function.
            chunksize (int): Size of the chunks in pixels (excluding buffer).
            **kwargs: Additional keyword arguments to pass to rasterio.open.
        """
        chunks = self.chunk_execute(f, f_args, f_kwargs, chunksize)
        arr, bbox = next(chunks)
        with self._open_writer(dst, arr.shape[0], arr.dtype, **kwargs) as _writer:
            self._write_chunk(_writer, arr, bbox)
            for arr, bbox in chunks:
                self._write_chunk(_writer, arr, bbox)

    def _open_writer(
        self, dst: Union[str, IO], count: int, dtype: Type, **kwargs
    ) -> DatasetWriter:
        """Open a rasterio.DatasetWriter with default profile."""
        profile = _get_profile(
            count, self.scale, self._proj_bounds, self.proj, dtype, **kwargs
        )

        return rio.open(dst, "w", **profile)  # type: ignore

    def _write_chunk(
        self,
        writer: DatasetWriter,
        arr: np.ndarray,
        bbox: BoundingBox,
    ):
        """Write a chunk to a rasterio.DatasetWriter."""
        if len(arr.shape) == 2:
            arr = arr[np.newaxis]
        height, width = arr.shape[1:]
        window = rasterio.windows.from_bounds(*bbox, writer.transform, height, width)

        writer.write(arr, window=window)
        if isinstance(arr, np.ma.MaskedArray):
            mask = np.ma.getmask(arr)
            if len(mask.shape) == 3:
                mask = np.any(mask, axis=0)
            writer.write_mask(~mask, window=window)

    def chunk_params(self, chunksize: int = 512, **kwargs):
        """Generate parameters to execute a function in chunks.

        Generates dicts of keyword arguments that can be passed to Manager.execute to run a function in chunks
        of size <chunksize>. This may be useful for distributing tasks to workers to execute in parallel. Each dict
        will contain the projection, scale, bounding box, and buffer. Attributes will be identical except
        for ``proj_bounds`` which define the area to analyze.

        Note:
            ``manager.chunk_execute(f)`` is equivalent to
            ``(manager.execute(f, **params) for params in manager.chunk_params())``

        Args:
            chunksize (int): Size of the chunks in pixels (excluding buffer).
            **kwargs: optional additional keyword arguments to save to the dict (to eventually pass to Manager.execute)
                e.g. ``f``, ``f_args``, ``f_kwargs``

        Yields:
            Dicts of keyword arguments that can be passed to :obj:`cog_worker.manager.Manager.execute()`.
        """
        _args = {
            "proj": self.proj.srs,
            "scale": self.scale,
            "buffer": self.buffer,
        }
        _args.update(kwargs)

        for proj_bounds in self.chunks(chunksize):
            args = _args.copy()
            args["proj_bounds"] = proj_bounds
            yield args

    def chunks(self, chunksize: int = 512) -> Iterator[BoundingBox]:
        """Generate bounding boxes for chunks of at most <chunksize> pixels in the managers scale and projection.

        The chunks method divides the Manager's bounding box into chunks of manageable size.
        Each chunk will be at most <chunksize> pixels, though the geographic extent of the chunk
        depends on the Manager's projection and scale.

        Args:
            chunksize (int): Size of the chunks in pixels (excluding buffer).

        Yields:
            BoundingBox: The bounding box of the chunk in the Manager's projection
        """
        xshards, yshards = self._num_chunks(chunksize)
        for i in range(xshards):
            for j in range(yshards):
                bounds = self._chunk_bounds(i, j, chunksize)
                if np.isfinite(bounds).all():
                    yield bounds

    def _chunk_bounds(
        self,
        x: int,
        y: int,
        chunksize: int,
    ) -> BoundingBox:
        """Get the bounding box of a chunk with index <x>,<y>."""
        left, bottom, right, top = self._proj_bounds
        _chunksize = chunksize * self.scale

        l = left + x * _chunksize
        r = min(l + _chunksize, right)
        t = top - y * _chunksize
        b = max(t - _chunksize, bottom)

        return (l, b, r, t)

    def _num_chunks(
        self,
        chunksize: int,
    ) -> Tuple[int, int]:
        """Return the number of chunks necessary to cover the Manager's bounding box."""
        left, bottom, right, top = self._proj_bounds
        return (
            math.ceil((right - left) / self.scale / chunksize),
            math.ceil((top - bottom) / self.scale / chunksize),
        )


def _execute(
    f: WorkerFunction,
    f_args: Iterable = None,
    f_kwargs: Mapping = None,
    clip: bool = True,
    **kwargs
) -> Tuple[Any, BoundingBox]:
    """Execute a function that takes a cog_worker.worker.Worker as its first parameter.

    Instantiate a cog_worker.worker.Worker and pass it to the function as its first parameter.

    Args:
        f (:obj:`cog_worker.types.WorkerFunction`): The function to execute. The function will recieve a
            cog_worker.worker.Worker as its first argument.
        f_args (list): Additional arguments to pass to the function.
        f_kwargs (dict): Additional keyword arguments to pass to the function.
        clip (bool): Whether or not to clip the buffer from the completed analysis.
        **kwargs: Additional keyword arguments to instantiate the cog_worker.worker.Worker

    Returns:
        A tuple containing the return value of the function and the bounding
        box of the executed analysis in the target projection.
    """
    worker = cog_worker.worker.Worker(**kwargs)

    f_args = [] if f_args is None else f_args
    f_kwargs = {} if f_kwargs is None else f_kwargs

    arr: np.ndarray = f(worker, *f_args, **f_kwargs)  # type: ignore
    if clip and isinstance(arr, np.ndarray):
        arr = worker.clip_buffer(arr)

    return arr, worker.bounds
