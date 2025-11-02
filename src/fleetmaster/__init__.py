"""Public API and version retrieval for the fleetmaster package."""

from importlib.metadata import PackageNotFoundError, version

from .core import FleetMaster
from .core.io import load_meshes_from_hdf5

DIST_NAME: str = "fleetmaster"

try:
    __version__: str = version(DIST_NAME)
except PackageNotFoundError:
    __version__ = "unknown"

__all__: list[str] = [
    "FleetMaster",
    "__version__",
    "load_meshes_from_hdf5",
]
