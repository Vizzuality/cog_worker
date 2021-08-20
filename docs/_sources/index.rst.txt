Cog Worker
==========

Scalable geospatial analysis on `Cloud Optimized GeoTIFFs (COGs) <https://www.cogeo.org/>`_. 

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Contents

   Introduction <self>
   examples
   api_reference

cog_worker is a simple python library to help write and chunk
analyses of gridded data for applications including GIS, remote sensing, 
and machine learning.

It provides two things:

1. A pattern for writing **projection- and scale-agnostic 
   analyses** of COGs, and
2. Methods for **previewing** the results of these analyses and
   executing them in managable **chunks**.

Under the hood, cog_worker is just a wrapper around `rio-tiler <https://cogeotiff.github.io/rio-tiler/>`_.
It does dynamic reprojection and rescaling of input data, 
enabling you to combine data sources in different projections, 
and surfaces this data as Numpy arrays that work with the familiar 
scientific python tools.

Example
-------

.. code-block:: python

   import numpy as np
   from cog_worker import Manager

   # Write your analysis as a function that takes a cog_worker.Worker
   # as its first parameter
   def my_analysis(worker):
      arr = worker.read('example-cog.tif')
      # calculations ...
      return arr

   # Run the function in chunks in a given scale and projection
   manager = Manager(proj='wgs84', scale=0.00083333)
   manager.chunk_save('output.tif', my_analysis, chunksize=512)


Installation
------------

Install from PyPI.

.. code-block::

   pip install cog_worker

If you want to use the :obj:`cog_worker.distributed` module to execute functions in a
`Dask cluster <https://distributed.dask.org/>`_, you will also need to install 
``dask.distributed``.

.. code-block::

   pip install dask[distributed]

Getting started
---------------

Follow the `quick start <examples/1.\ quick-start.html>`_ notebook.

See also
--------

Other tools:

- `rio-cogeo <https://cogeotiff.github.io/rio-cogeo/>`_ Cloud Optimized GeoTIFF (COG) creation and validation plugin for Rasterio. (Need to turn your data into COGs?)
- `titiler <https://developmentseed.org/titiler>`_ - A modern dynamic tile server built on top of FastAPI and Rasterio/GDAL. (Need to serve COGs as tiles for a webmap?)
- `stackstac <https://stackstac.readthedocs.io/en/latest/index.html>`_ - Easier cloud-native geoprocessing. (All your data have the same projection/scale?)

Dependencies:

- `rasterio <https://rasterio.readthedocs.io/>`_
- `numpy <https://numpy.org/doc/>`_
- `pyproj <https://pyproj4.github.io/pyproj/>`_
- `dask <https://docs.dask.org>`_
- `gdal <https://gdal.org>`_

Links
-----

- `cog_worker Github <https://github.com/vizzuality/cog_worker>`_
- `cog_worker PyPI package <https://pypi.org/project/cog-worker>`_