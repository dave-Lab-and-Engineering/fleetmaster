from typing import Any

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

from fleetmaster.core.exceptions import (
    InvalidVectorLength,
    LidAndSymmetryEnabledError,
    NegativeForwardSpeedError,
    NonPositivePeriodError,
)

MESH_GROUP_NAME = "meshes"


def _parse_special_float_value(value: Any) -> Any:
    """Parse string float aliases such as .inf into numeric numpy values."""
    if not isinstance(value, str):
        return value

    lowered = value.strip().lower()
    if lowered in {"inf", ".inf", "infinity", "np.inf", "+inf", "+.inf", "+infinity", "+np.inf"}:
        return np.inf
    if lowered in {"-inf", "-.inf", "-infinity", "-np.inf"}:
        return -np.inf
    if lowered in {"nan", ".nan", "np.nan"}:
        return np.nan

    try:
        return float(value)
    except ValueError:
        return value


class MeshConfig(BaseModel):
    """Configuration for a single mesh, including its path and transformation."""

    file: str
    name: str | None = Field(
        default=None, description="An optional name for the mesh. If not provided, it's derived from the file name."
    )
    translation: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation: list[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0], description="Rotation [roll, pitch, yaw] in degrees."
    )
    cog: list[float] | None = Field(
        default=None, description="Center of Gravity [x,y,z] for this mesh, around which moments are calculated."
    )
    wave_periods: float | list[float] | None = Field(
        default=None, description="Mesh-specific wave periods. Overrides global settings."
    )
    wave_directions: float | list[float] | None = Field(
        default=None, description="Mesh-specific wave directions in degrees. Overrides global settings."
    )

    @field_validator("translation", "rotation")
    def check_vector_length(cls, v: list[float]) -> list[float]:
        if len(v) != 3:
            msg = "Translation and rotation must a of length 3"
            raise InvalidVectorLength(msg)
        return v

    @field_validator("cog")
    def check_cog_length(cls, v: list[float] | None) -> list[float] | None:
        if v is not None and len(v) != 3:
            msg = "Cog must be a list of 3 floats or None"
            raise InvalidVectorLength(msg)
        return v


class SimulationSettings(BaseModel):
    """Defines all possible settings for a simulation.

    The 'description' for each field is used by the CLI in `run.py` to
    automatically generate the help text for its corresponding command-line option.
    """

    base_mesh: str | None = Field(
        default=None, description="Path to the base STL mesh file for defining the origin of the coordinate system."
    )
    base_origin: list[float] | None = Field(
        default=None,
        description="A point [x, y, z] in the local coordinate system of the base_mesh that defines the world origin.",
    )
    stl_files: list[MeshConfig] = Field(description="A list of STL mesh files or mesh configurations.")
    output_directory: str | None = Field(default=None, description="Directory to save the output files.")
    output_hdf5_file: str = Field(default="results.hdf5", description="Path to the HDF5 output file.")
    output_dhyd_file: str | None = Field(
        default=None,
        description=(
            "Optional path for exporting a single-case mafredo hydrodynamic database (.dhyd). "
            "Only supported when exactly one mesh and one simulation case are configured."
        ),
    )
    export_transformed_stl: bool = Field(
        default=False,
        description=("Export the final transformed mesh that is written to HDF5 also as a standalone .stl file."),
    )
    output_transformed_stl_directory: str | None = Field(
        default=None,
        description=(
            "Optional output directory for transformed STL exports. If omitted, files are written to "
            "<output_directory>/transformed_stl."
        ),
    )
    export_to_hyd: bool = Field(
        default=False,
        description=(
            "Export a standalone .dhyd file for a single configured case. If no output_dhyd_file is provided, "
            "the filename is generated automatically from the case name."
        ),
    )
    wave_periods: float | list[float] = Field(default=[5.0, 10.0, 15.0, 20.0])
    wave_directions: float | list[float] = Field(default=[0.0, 45.0, 90.0, 135.0, 180.0])
    forward_speed: float | list[float] = 0.0
    lid: bool = False
    add_center_of_mass: bool = False
    grid_symmetry: bool = False
    water_depth: float | list[float] = np.inf
    water_level: float | list[float] = 0.0
    overwrite_meshes: bool = Field(default=False, description="Overwrite existing meshes in the database.")
    update_cases: bool = Field(default=False, description="Force update of existing simulation cases in the database.")
    drafts: list[float] | None = Field(
        default=None, description="A list of draft values to apply as Z-translations to a base mesh."
    )
    combine_cases: bool = Field(
        default=False, description="Combine all calculated cases for a single STL into one multi-dimensional dataset."
    )

    @field_validator("wave_periods", "wave_directions", mode="before")
    def parse_range_string(cls, v: Any) -> list[float]:
        if isinstance(v, str):
            try:
                start, stop, step = map(float, v.split(":"))
                return np.arange(start, stop, step).tolist()
            except ValueError as e:
                msg = f"Invalid range string format: {v}"
                raise ValueError(msg) from e
        if isinstance(v, (int, float)):
            return [v]
        if isinstance(v, list):
            return v
        msg = f"Unsupported type for wave_periods/wave_directions: {type(v)}"
        raise TypeError(msg)

    @field_validator("water_depth", "water_level", mode="before")
    def normalize_water_values(cls, v: Any) -> Any:
        """Normalize water settings so YAML forms like .inf map to numpy infinity."""
        if isinstance(v, list):
            return [_parse_special_float_value(item) for item in v]
        return _parse_special_float_value(v)

    @field_validator("stl_files", mode="before")
    def normalize_stl_files(cls, v: Any) -> Any:
        if not isinstance(v, list):
            return v
        return [MeshConfig(file=item) if isinstance(item, str) else item for item in v]

    # field validator checks the value of one specific field inmediately
    @field_validator("forward_speed")
    def speed_must_be_non_negative(cls, v: float | list[float]) -> float | list[float]:
        """Validate that forward speed is non-negative."""
        if isinstance(v, list):
            if any(speed < 0 for speed in v):
                raise NegativeForwardSpeedError()
        elif v < 0:
            raise NegativeForwardSpeedError()
        return v

    @field_validator("wave_periods")
    def periods_must_be_positive(cls, v: list[float]) -> list[float]:
        """Validate that wave periods are positive."""
        if isinstance(v, list):
            if any(p <= 0 for p in v):
                raise NonPositivePeriodError()
        elif v <= 0:
            raise NonPositivePeriodError()

        return v

    # model_validator checks the combination of fields, after all fields have been set
    @model_validator(mode="after")
    def check_lid_and_symmetry(self) -> "SimulationSettings":
        """Validate that lid and grid_symmetry are not both enabled."""
        if self.lid and self.grid_symmetry:
            raise LidAndSymmetryEnabledError()

        return self
