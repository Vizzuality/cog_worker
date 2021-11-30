"""Distributed processing with Dask.

The DaskManager class provides an identical interface to the
:obj:`cog_worker.manager.Manager`, but executes tasks in a
`Dask cluster <https://distributed.dask.org/>`_, instead of on your local machine.

Note:
    cog_worker does not include dask.distributed as a dependency by default.
    In order to use cog_worker.distributed you must install dask.distributed::

        pip install dask[distributed]

Example:
    Read a COG in chunks and sum the results::

        from cog_worker.distributed import DaskManager
        from dask.distributed import Client, LocalCluster

        def my_analysis(worker):
            arr = worker.read('example-cog.tif')
            return arr.sum()

        cluster = LocalCluster()
        client = Client(cluster)
        manager = DaskManager(client)

        results = manager.chunk_execute(my_analysis)
        total = sum(results)
"""
from typing import Iterable, Iterator, Mapping, Union, Tuple, Any
import logging

import dask
import dask.distributed
from dask.delayed import Delayed
from pyproj import Proj

import cog_worker
from cog_worker.types import WorkerFunction, BoundingBox


logger = logging.getLogger(__name__)


class DaskManager(cog_worker.manager.Manager):
    """Class for chunking and executing cog_worker functions in a dask cluster.

    The DaskManager identical to the cog_worker.manager.Manager, except that it
    executes functions in a Dask cluster instead of locally.
    """

    def __init__(
        self,
        dask_client: dask.distributed.Client,
        bounds: BoundingBox = (-180, -85, 180, 85),
        proj: Union[int, str, Proj] = 3857,
        scale: float = 10000,
        buffer: int = 16,
    ):
        """Initialize a DaskManager with a dask client.

        Args:
            dask_client (dask.distributed.Client): The dask client to use to
                execute analysis.
            bounds (BoundingBox): The region to be analyzed.
            proj (pyproj.Proj, str, int): The projection to analyze in.
                Generally accepts any proj4 string, WKT projection, or EPSG
                code. See pyproj.Proj for valid values.
            scale (float): The pixel size for analysis in the projection's units
                (usually meters or degrees).
            buffer (int): When dividing analysis into chunks, the number of
                additional pixels to read on all sides to avoid edge effects.
                The ideal buffer size depends on your analysis (e.g. whether you
                use convolutions or distance functions).
        """
        self.client = dask_client
        super().__init__(bounds, proj, scale, buffer)

    def execute(
        self,
        f: WorkerFunction,
        f_args: Iterable = None,
        f_kwargs: Mapping = None,
        clip: bool = True,
        compute: bool = True,
        **kwargs
    ) -> Union[Tuple[Any, BoundingBox], Delayed]:
        """Execute a cog_worker function in the DaskManager's cluster.

        The execute method is the underlying method for running analysis. By
        default, it will run the function over the Manager's bounding box in a
        single chunk.

        When executing functions, the Manager instantiates a
        cog_worker.worker.Worker and passes it to the function as its first
        parameter. The Worker keeps track of the scale, projection, and bounds
        of its piece of the analysis, which it uses to handle the reading and
        writing of Cloud Optimized GeoTIFFs.

        Args:
            f (:obj:`cog_worker.types.WorkerFunction`): The function to execute. The function will
                recieve a cog_worker.worker.Worker as its first argument.
            f_args (list): Additional arguments to pass to the function.
            f_kwargs (dict): Additional keyword arguments to pass to the
                function.
            clip (bool): Whether or not to clip the `buffer` from the completed
                analysis.
            compute (bool): Whether or not to compute the chunks immediately.
            **kwargs: Additional keyword arguments to overload the Manager's
                properties. (bounds, proj, scale, or buffer)

        Returns:
            A tuple containing the return value of the function and the bounding
            box of the executed analysis in the target projection. Or, if
            compute is False, a Delayed object.
        """
        args = {
            "f": f,
            "f_args": f_args,
            "f_kwargs": f_kwargs,
            "bounds": self.bounds,
            "proj": self.proj,
            "scale": self.scale,
            "buffer": self.buffer,
            "clip": clip,
        }
        args.update(kwargs)
        task = dask.delayed(cog_worker.manager._execute)(**args)
        if compute:
            future = self.client.compute(task)
            return future.result()  # type: ignore
        return task

    def chunk_execute(
        self,
        f: WorkerFunction,
        f_args: Iterable = None,
        f_kwargs: Mapping = None,
        chunksize: int = 512,
        compute: bool = True,
    ) -> Union[Iterator[Tuple[Any, BoundingBox]], Iterator[Delayed]]:
        """Compute chunks in parallel in the DaskManager's cluster.

        Chunks will be yielded as they are completed. The order in which they
        are yielded is not guaranteed.

        Note:
            You can estimate the memory requirement of executing a function at a
            given chunksize as:
            ``(chunksize + 2*buffer)**2 * number_of_bands_or_arrays * bit_depth``.

        Args:
            f (:obj:`cog_worker.types.WorkerFunction`): The function to execute. The function will
                recieve a cog_worker.worker.Worker as its first argument.
            f_args (list): Additional arguments to pass to the function.
            f_kwargs (dict): Additional keyword arguments to pass to the
                function.
            chunksize (int): Size of the chunks in pixels (excluding buffer).
            compute (bool): Whether or not to compute the chunks immediately.

        Yields:
            A tuple containing the return value of the function for each chunk
            and the bounding box of the executed analysis in the target
            projection. Or, if compute is False, a Delayed object for each chunk.
        """
        tasks = [
            dask.delayed(cog_worker.manager._execute)(f, f_args, f_kwargs, **params)
            for params in self.chunk_params(chunksize)
        ]
        if compute:
            futures = self.client.compute(tasks)
            for future, result in dask.distributed.as_completed(
                futures, with_results=True
            ):
                yield result
        else:
            for t in tasks:
                yield t  # type: ignore
