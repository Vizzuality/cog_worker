"""A python module for scalable analysis of Cloud Optimized GeoTIFFs.

COG Worker is a simple library to help you chunk and run large scale analysis
on Cloud Optimized GeoTIFFs (COGS).
"""

__version__ = "0.1.4"

from .worker import Worker  # noqa
from .manager import Manager  # noqa
from .types import *  # noqa
