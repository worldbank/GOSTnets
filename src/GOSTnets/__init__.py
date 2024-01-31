from .core import *
from .conversion_utils import *
from .osm_parser import *
from .network_clean import *

# from .optimization import *

# make sure gdal, geopy, and boltons are installed first
# from .load_osm import *

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("GOSTnets")
except PackageNotFoundError:
    # package is not installed
    pass