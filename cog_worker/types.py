"""cog_worker type definitions."""
from typing import Callable, Tuple, Union

import numpy as np

import cog_worker.worker

BoundingBox = Tuple[float, float, float, float]
"""A ``(west, south, east, north)`` tuple."""

WorkerFunction = Union[
    Callable[["cog_worker.worker.Worker"], np.ndarray],
    Callable,
]
"""A function that can recieve a cog_worker.worker.Worker as its first parameter.

Additional aguments and keyword arguments can be passed to the Worker function
at time of execution with the ``f_args`` and ``f_kwargs`` parameters of
:obj:`cog_worker.manager.Manager.execute()`

Example:

    Read a specific COG and return it as an array::

        def my_analysis(worker: cog_worker.Worker):
            arr = worker.read('example-cog.tif')
            return arr

    Read a COG at a given url and get the neighborhood mean for a 1km square kernel::

        from scipy.ndimage import uniform_filter

        def my_analysis(worker: cog_worker.Worker, source_url: str):
            arr = worker.read(source_url)
            kernel_size = 1000/worker.scale  # in map units (meters)
            return uniform_filter(arr, kernel_size)

    Read a COG and optionally upload each chunk to an S3 bucket as it is computed::

        from rasterio import MemoryFile
        import boto3

        def my_analysis(worker: cog_worker.Worker, dst_bucket: str):
            arr = worker.read('example-cog.tif')

            if dst_bucket:
                with MemoryFile() as memfile:
                    fname = f'output_{worker.scale}_{worker.bounds[0]}_{worker.bounds[3]}.tif'
                    worker.write(arr, memfile)

                    memfile.seek(0)
                    boto3.client('s3').upload_fileobj(memfile, dst_bucket, fname)

            return arr
"""
