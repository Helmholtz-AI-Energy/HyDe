import sys

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

from .lowlevel import *
from .nn import *
from .spatial_domain import *
from .transform_domain.forpdn import *
from .transform_domain.hyminor import *
from .transform_domain.hyres import *
from .transform_domain.otvca import *
from .transform_domain.wsrrr import *
