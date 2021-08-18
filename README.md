# Cog Worker

Scalable geospatial analysis on Cloud Optimized GeoTIFFs.

 - **Documentation**: https://vizzuality.github.io/cog_worker
 - **PyPI**: https://pypi.org/project/cog-worker

cog_worker is a simple library to help write scripts to conduct scaleable
analysis of gridded data. It's intended to be useful for moderate- to large-scale 
GIS, remote sensing, and machine learning applications.

## Installation

```
pip install cog_worker
```

## Examples

See `docs/examples` for Jupyter notebook examples

## Quick start

0. A simple cog_worker script

```python
from rasterio.plot import show
from cog_worker import Manager

def my_analysis(worker):
    arr = worker.read('roads_cog.tif')
    return arr

manager = Manager(proj='wgs84', scale=0.083333)
arr, bbox = manager.preview(my_analysis)
show(arr)
```

1. Define an analysis function that recieves a cog_worker.Worker as the first parameter.

```python
from cog_worker import Worker, Manager
import numpy as np

# Define an analysis function to read and process COG data sources
def MyAnalysis(worker: Worker) -> np.ndarray:

    # 1. Read a COG (reprojecting, resampling and clipping as necessary)
    array: np.ndarray = worker.read('roads_cog.tif')

    # 2. Work on the array
    # ...

    # 3. Return (or post to blob storage etc.)
    return array
```

2. Run your analysis in different scales and projections

```python
import rasterio as rio

# Run your analysis using a cog_worker.Manager which handles chunking
manager = Manager(
    proj = 'wgs84',       # any pyproj string
    scale = 0.083333,  # in projection units (degrees or meters)
    bounds = (-180, -90, 180, 90),
    buffer = 128          # buffer pixels when chunking analysis
)

# preview analysis
arr, bbox = manager.preview(MyAnalysis, max_size=1024)
rio.plot.show(arr)

# preview analysis chunks
for bbox in manager.chunks(chunksize=1500):
    print(bbox)

# execute analysis chunks sequentially
for arr, bbox in manager.chunk_execute(MyAnalysis, chunksize=1500):
    rio.plot.show(arr)

# generate job execution parameters
for params in manager.chunk_params(chunksize=1500):
    print(params)
```

3. Write scale-dependent functions¶

```python
import scipy

def focal_mean(
    worker: Worker,
    kernel_radius: float = 1000 # radius in projection units (meters)
) -> np.ndarray:

    array: np.ndarray = worker.read('sample-geotiff.tif')

    # Access the pixel size at worker.scale
    kernel_size = kernel_radius * 2 / worker.scale
    array = scipy.ndimage.uniform_filter(array, kernel_size)

    return array
```

4. Chunk your analysis and run it in a dask cluster

```python
from cog_worker.distributed import DaskManager
from dask.distributed import LocalCluster, Client

# Set up a Manager with that connects to a Dask cluster
cluster = LocalCluster()
client = Client(cluster)
distributed_manager = DaskManager(
    client,
    proj = 'wgs84',
    scale = 0.083333,
    bounds = (-180, -90, 180, 90),
    buffer = 128
)

# Execute in worker pool and save chunks to disk as they complete.
distributed_manager.chunk_save('output.tif', MyAnalysis, chunksize=2048)
```