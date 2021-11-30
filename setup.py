#!/usr/bin/env python
from setuptools import setup

with open("README.md") as f:
    desc = f.read()

setup(
    name="cog_worker",
    version="0.1.4",
    description="Scalable geospatial analysis on Cloud Optimized GeoTIFFs.",
    long_description=desc,
    long_description_content_type="text/markdown",
    license="MIT",
    author="Francis Gassert",
    author_email="francis.gassert@vizzuality.com",
    url="https://github.com/vizzuality/cog_worker",
    packages=["cog_worker"],
    keywords="cog geotiff raster gdal rasterio dask",
    install_requires=[
        "numpy>=1,<2",
        "pyproj>=3.0.0,<4",
        "rasterio>=1.2,<2",
        "morecantile>=3.0.0,<4",
        "rio_tiler>=3.0.0,<4",
    ],
    extras_require={
        "test": ["pytest"],
        "dev": ["pre-commit"],
        "distributed": ["dask[distributed]"],
        "docs": [
            "Sphinx",
            "sphinxcontrib-napoleon",
            "furo",
            "nbsphinx",
            "nbconvert"
        ],
    },
)
