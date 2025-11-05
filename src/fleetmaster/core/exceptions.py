"""Custom exceptions for the Fleetmaster application's core logic."""

from pathlib import Path


class SimulationConfigurationError(ValueError):
    """Base exception for simulation configuration errors."""


class LidAndSymmetryEnabledError(SimulationConfigurationError):
    """Raised when both lid and grid_symmetry are enabled simultaneously."""

    def __init__(
        self,
        message: str = "Cannot have both lid and grid_symmetry True simultaneously.",
    ) -> None:
        super().__init__(message)


class NegativeForwardSpeedError(SimulationConfigurationError):
    """Raised when forward speed is negative."""

    def __init__(self, message: str = "Forward speed must be non-negative.") -> None:
        super().__init__(message)


class NonPositivePeriodError(SimulationConfigurationError):
    """Raised when a simulation period is not positive."""

    def __init__(self, message: str = "Periods must be larger than 0.") -> None:
        super().__init__(message)


class InvalidVectorLength(SimulationConfigurationError):
    """Raised when a vector has an invalid length."""

    def __init__(self, message: str = "Invalid vector length") -> None:
        super().__init__(message)


class HDF5AttributeError(ValueError):
    """Raised when a required attribute is missing from an HDF5 file."""

    def __init__(self, attribute_name: str) -> None:
        message = f"Required attribute '{attribute_name}' is missing from the HDF5 file."
        super().__init__(message)


class MeshLoadError(RuntimeError):
    """Raised when a mesh cannot be loaded from the database."""

    def __init__(self, mesh_name: str) -> None:
        message = f"Failed to load the required mesh '{mesh_name}' from the database."
        super().__init__(message)


class DatabaseFileNotFoundError(FileNotFoundError):
    """Raised when the main HDF5 database file is not found."""

    def __init__(self, path: Path) -> None:
        message = f"HDF5 database not found at path: {path}"
        super().__init__(message)


class BaseMeshIsNoneError(ValueError):
    """Raised when the base mesh is None."""

    def __init__(self, base_mesh_name: str) -> None:
        message = "base_mesh {base_mesh_name} cannot be None when finding best fit."
        super().__init__(message)
