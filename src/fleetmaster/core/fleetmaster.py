"""
This module defines the main FleetMaster class, which provides the primary API
for interacting with FleetMaster databases.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import trimesh
import trimesh.transformations as tf
import xarray as xr
from mafredo import Hyddb1

from .engine import (
    MESH_GROUP_NAME,
    EngineMesh,
    _apply_mesh_translation_and_rotation,
    _prepare_capytaine_body,
    create_hyd_from_capytaine_data,
    load_meshes_from_hdf5,
)
from .exceptions import BaseMeshIsNoneError, DatabaseFileNotFoundError, HDF5AttributeError, MeshLoadError
from .fitting import _calculate_chamfer_distance
from .settings import MeshConfig

logger = logging.getLogger(__name__)

HyddbResult = tuple[Any | None, tuple[float, float, float] | None, float | None, float | None]


class FleetMaster:
    """
    The main class for interacting with a FleetMaster database.

    This class handles loading the database, setting parameters, and running
    the fitting process to find the best matching mesh.
    """

    def __init__(self, filename: str | Path) -> None:
        """
        Initializes the FleetMaster object and loads all mesh data from the database.

        Args:
            filename: The path to the HDF5 database file.

        Raises:
            DatabaseFileNotFoundError: If the HDF5 file does not exist.
            HDF5AttributeError: If essential attributes are missing from the file.
            MeshLoadError: If the base mesh cannot be loaded.
        """
        self.filename = Path(filename)
        if not self.filename.exists():
            raise DatabaseFileNotFoundError(path=self.filename)

        self._loaded_meshes: dict[str, EngineMesh] = {}
        self.base_mesh: EngineMesh | None = None
        self.base_mesh_name: str | None = None
        self.candidate_meshes: dict[str, EngineMesh] = {}
        self._loaded_cases: dict[str, Any] = {}

        self._water_depth: float = np.inf
        self._velocity: float = 0.0
        self._origin: tuple[float, float, float] = (0.0, 0.0, 0.0)

        self._best_match_name: str | None = None
        self._match_error: float = np.inf
        self._best_match_hydro_data: xr.Dataset | None = None

        self._load_database()

    def set_waterdepth(self, water_depth: float) -> None:
        """
        Sets the water depth for the fitting process.

        Args:
            water_depth: The water depth in meters. Use -1 for infinite depth.
        """
        self._water_depth = water_depth
        logger.debug(f"Set water depth to: {water_depth}")

    def set_velocity(self, velocity: float) -> None:
        """
        Sets the forward speed for the fitting process.

        Args:
            velocity: The forward speed in m/s.
        """
        self._velocity = velocity
        logger.debug(f"Set velocity to: {velocity}")

    def set_origin(self, x: float, y: float, z: float) -> None:
        """
        Sets the origin point of the database relative to the parent frame.

        Args:
            x: The x-coordinate of the origin.
            y: The y-coordinate of the origin.
            z: The z-coordinate of the origin.
        """
        self._origin = (x, y, z)
        logger.debug(f"Set origin to: {(x, y, z)}")

    def fit_mesh(self, transform: Any, origin: tuple[float, float, float] | None = None) -> None:
        """
        Runs a query on the Fleetmaster database to select the best matching dataset.

        Args:
            transform: A transformation matrix (4x4 numpy array) or similar object
                       representing the target orientation and position.
            origin: The origin to be used for the transformation. If None, the
                    class's internal origin is used.
        """
        if origin is not None:
            self.set_origin(*origin)

        # Assuming 'transform' is a 4x4 transformation matrix.
        # We need to extract translation and rotation from it.
        translation = list(tf.translation_from_matrix(transform))
        # The 'sxyz' convention means rotations are applied in order: roll (x), pitch (y), yaw (z).
        rotation_rad = tf.euler_from_matrix(transform, "sxyz")
        rotation_deg = np.rad2deg(np.asarray(rotation_rad)).tolist()

        logger.info("Starting fitting process...")
        logger.info(f"  - Target Translation: {translation}")
        logger.info(f"  - Target Rotation (deg): {rotation_deg}")
        logger.info(f"  - Water Level: {self._water_depth}")

        best_match, min_distance = self._find_best_matching_mesh(
            target_translation=translation,
            target_rotation=rotation_deg,
        )

        self._best_match_name = best_match
        self._match_error = min_distance

        if best_match:
            logger.info(f"Fitting complete. Best match: '{best_match}' with error: {min_distance:.4f}")
        else:
            logger.warning("Fitting complete, but no suitable match was found.")

    def find_best_case(
        self,
        forward_speed: float,
        water_depth: float,
        water_level: float,
    ) -> None:
        """
        Selects the best matching case from the database based on forward speed,
        water depth, and water level, for the best mesh found by fit_mesh.

        After finding the best case, it loads the corresponding hydrodynamic data.

        Args:
            forward_speed: The target forward speed in m/s.
            water_depth: The target water depth in meters.
            water_level: The target water level in meters.
        """
        if not self._best_match_name:
            logger.warning("Cannot select best case. Run fit_mesh() first.")
            return

        logger.info(
            "Selecting best case for mesh '%s' with parameters: forward_speed=%f, water_depth=%f, water_level=%f",
            self._best_match_name,
            forward_speed,
            water_depth,
            water_level,
        )

        candidate_cases = self._collect_candidate_cases(self._best_match_name)

        if not candidate_cases:
            logger.warning(f"No cases found for mesh '{self._best_match_name}'.")
            return

        best_case = self._find_best_matching_case(candidate_cases, forward_speed, water_depth, water_level)

        if best_case:
            logger.info(f"Best matching case found: {best_case['name']}")
            if not best_case["exact_match"]:
                params = best_case["params"]
                logger.warning(
                    "No exact match found for the given parameters. "
                    "Selected the closest case with parameters: "
                    "forward_speed=%f, water_depth=%f, water_level=%f",
                    params["forward_speed"],
                    params["water_depth"],
                    params["water_level"],
                )

            self._load_hydro_data(best_case["name"])
            self.set_velocity(best_case["params"]["forward_speed"])
            self.set_waterdepth(best_case["params"]["water_depth"])

        else:
            logger.warning("Could not find a suitable case.")

    def _load_hydro_data(self, case_group_name: str) -> None:
        """Loads hydrodynamic data for the given case group name from the pre-loaded cases."""
        logger.info(f"Retrieving hydrodynamic data for case group '{case_group_name}'...")

        case_data = self._loaded_cases.get(case_group_name)

        if case_data:
            self._best_match_hydro_data = case_data["hydro_data"]
            logger.info(f"Successfully retrieved hydrodynamic data for case '{case_group_name}'.")
        else:
            self._best_match_hydro_data = None
            logger.error(f"Case group '{case_group_name}' not found in pre-loaded cases.")

    def get_match_error(self) -> float:
        """
        Returns the error of the best match from the fitting process.

        Returns:
            The Chamfer distance of the best match. Returns np.inf if no
            fit has been performed or no match was found.
        """
        return self._match_error

    def get_best_match_name(self) -> str | None:
        """
        Returns the name of the best matching mesh found during the fitting process.

        Returns:
            The name of the best matching mesh, or None if no fit has been performed.
        """
        return self._best_match_name

    def get_grid(self) -> tuple[Any, Any]:
        """
        Gets the results (offset and grid) of the match.

        This method is not yet fully implemented.

        Returns:
            A tuple containing the offset and the grid data.
        """
        if not self._best_match_name:
            logger.warning("Cannot get grid. No successful fit has been performed yet.")
            return None, None

        logger.info(f"Loading grid data for best match: {self._best_match_name}")
        # Placeholder for implementation.
        # This would involve reading the specific dataset corresponding to
        # self._best_match_name from the HDF5 file.
        offset = None
        grid = None
        logger.warning("get_grid() is not yet implemented.")
        return offset, grid

    def _create_hyddb_from_data(self, hydro_data: dict[str, Any]) -> Any | None:
        """Creates and populates a Hyddb1 object from a dictionary of hydro data."""

        # Extract data from the hydro_data dictionary, which comes from the HDF5 file
        omega = hydro_data["omega"]
        added_mass = hydro_data["added_mass"]
        damping = hydro_data["radiation_damping"]
        directions_deg = np.rad2deg(hydro_data["wave_direction"])
        excitations = hydro_data["excitation_force"]
        force_amps = np.transpose(np.abs(excitations), (2, 0, 1))
        force_phases = np.transpose(np.angle(excitations), (2, 0, 1))

        hyddb = Hyddb1()

        hyddb.set_data(
            omega=omega,
            added_mass=added_mass,
            damping=damping,
            directions=directions_deg,
            force_amps=force_amps,
            force_phase_rad=force_phases,
        )

        return hyddb

    def get_hyddb1(self) -> HyddbResult:
        """
        Returns the hydrodynamic database (Hyddb1), application point, velocity, and water depth.

        Returns:
            A tuple containing:
            - Hyddb1: The hydrodynamic database object from mafredo.
            - tuple: The application point (origin).
            - float: The velocity.
            - float: The water depth.
        """
        if not self._best_match_hydro_data:
            logger.warning("No hydrodynamic data loaded. Run find_best_case() first.")
            return None, None, None, None

        hyddb = create_hyd_from_capytaine_data(self._best_match_hydro_data)
        if not hyddb:
            return None, None, None, None

        return hyddb, self._origin, self._velocity, self._water_depth

    def _load_database(self) -> None:
        """

        Loads all meshes and case data from the HDF5 database file in a single pass.

        """

        if not self.filename.exists():
            raise DatabaseFileNotFoundError(path=self.filename)

        logger.info("Loading all meshes and cases from the database...")

        with h5py.File(self.filename, "r") as f:
            self._load_meshes_from_file(f)

            self._load_cases_from_file(f)

        logger.info(f"Successfully loaded {len(self._loaded_meshes)} meshes and {len(self._loaded_cases)} cases.")

    def _load_meshes_from_file(self, f: Any) -> None:
        """Loads all meshes from the opened HDF5 file object."""

        if "base_mesh" not in f.attrs:
            raise HDF5AttributeError(attribute_name="base_mesh")

        base_mesh_name_str = str(f.attrs["base_mesh"])

        if MESH_GROUP_NAME not in f:
            logger.warning(f"No '{MESH_GROUP_NAME}' group found in HDF5 file. Cannot find any meshes.")

            return

        mesh_group = f[MESH_GROUP_NAME]

        if not isinstance(mesh_group, h5py.Group):
            logger.warning(f"'{MESH_GROUP_NAME}' in HDF5 file is not a group as expected.")

            return

        candidate_mesh_names = [str(name) for name in mesh_group if name != base_mesh_name_str]

        all_meshes_to_load = [base_mesh_name_str, *candidate_mesh_names]

        self._loaded_meshes = {
            mesh.metadata["name"]: EngineMesh(
                name=mesh.metadata["name"], mesh=mesh, config=MeshConfig(file="from_hdf5")
            )
            for mesh in load_meshes_from_hdf5(f, all_meshes_to_load)
        }

        self.base_mesh_name = base_mesh_name_str

        self.base_mesh = self._loaded_meshes.get(self.base_mesh_name)

        if not self.base_mesh:
            raise MeshLoadError(mesh_name=self.base_mesh_name)

        self.candidate_meshes = {
            name: mesh for name, mesh in self._loaded_meshes.items() if name != self.base_mesh_name
        }

    def _load_cases_from_file(self, f: Any) -> None:
        """Loads all cases from the opened HDF5 file object using xarray."""
        loaded_cases = {}

        for group_name in f:
            if group_name == MESH_GROUP_NAME:
                continue  # Skip the mesh group

            if not isinstance(f.get(group_name), h5py.Group):
                continue

            params = self._parse_case_name(group_name)
            if not params:
                logger.debug(f"Could not parse parameters from group name '{group_name}'. Not a case.")
                continue

            try:
                # Use xarray to open the group as a Dataset
                hydro_data = xr.open_dataset(self.filename, group=group_name, engine="h5netcdf")
                loaded_cases[group_name] = {
                    "params": params,
                    "hydro_data": hydro_data,
                }
            except Exception:
                logger.exception(f"Failed to read group '{group_name}' with xarray; skipping.")
                continue

        self._loaded_cases = loaded_cases

    def _collect_candidate_cases(self, mesh_name: str) -> list[dict[str, Any]]:
        """
        Collects all cases associated with a given mesh name from the pre-loaded cases.
        """
        cases = []
        for case_name, case_data in self._loaded_cases.items():
            if case_name.startswith(mesh_name):
                cases.append({"name": case_name, "params": case_data["params"]})
        return cases

    def _parse_case_name(self, group_name: str) -> dict[str, float] | None:
        """
        Parses a case name like '{mesh_name}_wd_{wd}_wl_{wl}_fs_{fs}'
        and returns a dictionary with the parameters.
        """
        parts = group_name.split("_")
        # A valid name should look like '..._wd_X_wl_Y_fs_Z'
        if len(parts) < 6 or parts[-6] != "wd" or parts[-4] != "wl" or parts[-2] != "fs":
            return None

        def parse_val(s: str) -> float:
            return np.inf if s == "inf" else float(s)

        try:
            return {
                "water_depth": parse_val(parts[-5]),
                "water_level": parse_val(parts[-3]),
                "forward_speed": parse_val(parts[-1]),
            }
        except (ValueError, IndexError):
            return None

    def _find_best_matching_case(
        self,
        candidate_cases: list[dict[str, Any]],
        target_speed: float,
        target_depth: float,
        target_level: float,
    ) -> dict[str, Any] | None:
        """
        Finds the best matching case from a list of candidates.
        An exact match is returned if found. Otherwise, the case with the smallest
        Euclidean distance in the parameter space is returned.
        """
        if not candidate_cases:
            return None

        def transform_depth(d: float) -> float:
            """Transforms depth to handle infinity gracefully."""
            if np.isinf(d):
                return 0.0
            return 1.0 / (1.0 + d)

        # First, mark all cases as not exact matches.
        for this_case in candidate_cases:
            this_case["exact_match"] = True

        # First, look for a practically equivalent match.
        for this_case in candidate_cases:
            params = this_case["params"]
            if (
                np.isclose(params["forward_speed"], target_speed)
                and np.isclose(transform_depth(params["water_depth"]), transform_depth(target_depth))
                and np.isclose(params["water_level"], target_level)
            ):
                this_case["exact_match"] = True
                return this_case

        # If no exact match, find the closest one using a custom distance metric.
        distances = []
        for this_case in candidate_cases:
            params = this_case["params"]

            # Standard Euclidean distance for speed and level
            d_speed = (params["forward_speed"] - target_speed) ** 2
            d_level = (params["water_level"] - target_level) ** 2

            # Transformed distance for water depth, handling infinity which is now close to .e.g.
            # a depth of 1000 meters.
            d_depth = (transform_depth(params["water_depth"]) - transform_depth(target_depth)) ** 2

            dist = d_speed + d_depth + d_level
            distances.append(dist)

        if not distances:
            return None

        min_dist_idx = np.argmin(distances)
        best_case = candidate_cases[min_dist_idx]
        best_case["exact_match"] = False
        return best_case

    def _find_best_matching_mesh(
        self,
        target_translation: list[float],
        target_rotation: list[float],
    ) -> tuple[str | None, float]:
        """
        Finds the best matching mesh from the database for a given target transformation.

        The function works as follows:
        1.  Creates a target wetted mesh by transforming the base mesh. This transformation
            only considers the shape-defining degrees of freedom: Z-translation (draft),
            roll, and pitch. XY-translation and yaw are ignored as they do not change
            the submerged shape of the vessel.
        2.  The resulting mesh is cut at the specified water level to get the wetted surface.
        3.  This target wetted mesh is then compared against all pre-computed candidate
            wetted meshes in the database using the Chamfer distance.
        4.  The name of the candidate mesh with the smallest distance is returned as the best match.

        Args:
            target_translation: The target translation [x, y, z].
            target_rotation: The target rotation [roll, pitch, yaw] in degrees.

        Returns:
            A tuple containing the name of the best matching mesh and the corresponding
            Chamfer distance. Returns (None, np.inf) if no match is found.
        """
        if not self.base_mesh or not self.candidate_meshes:
            logger.warning("Base mesh or candidate meshes not loaded. Cannot find best match.")
            return None, np.inf

        # The water_depth is the water level for cutting the mesh.
        water_level = self._water_depth

        # 1. Create the target wetted mesh.
        # We only consider Z-translation (draft), roll, and pitch for the shape.
        # XY-translation and yaw are irrelevant for the wetted surface shape.
        shape_defining_translation = [0.0, 0.0, target_translation[2]]
        shape_defining_rotation = [target_rotation[0], target_rotation[1], 0.0]

        temp_base_mesh = self.base_mesh.copy()
        transformed_base_mesh = _apply_mesh_translation_and_rotation(
            mesh=temp_base_mesh.mesh,
            translation_vector=shape_defining_translation,
            rotation_vector_deg=shape_defining_rotation,
        )

        dummy_config = MeshConfig(file="dummy")
        engine_mesh_for_cutting = EngineMesh(name="target_shape", mesh=transformed_base_mesh, config=dummy_config)

        _, target_wetted_mesh = _prepare_capytaine_body(
            engine_mesh=engine_mesh_for_cutting,
            lid=False,
            grid_symmetry=False,
            water_level=water_level,
        )

        if not target_wetted_mesh or len(target_wetted_mesh.vertices) == 0:
            logger.warning("Target mesh is out of the water. Cannot find any match.")
            return None, np.inf

        # 2. Calculate distances to all candidates
        all_distances = self._calculate_distances_to_candidates(target_wetted_mesh)

        # 3. Find the minimum distance among the results
        if not all_distances:
            logger.warning("No distances could be calculated.")
            return None, np.inf

        best_match_name = min(all_distances, key=lambda k: all_distances[k])
        min_distance = all_distances[best_match_name]

        logger.info(f"Best match found: '{best_match_name}' with a Chamfer distance of {min_distance:.4f}")
        return best_match_name, min_distance

    def _calculate_distances_to_candidates(self, target_wetted_mesh: trimesh.Trimesh) -> dict[str, float]:
        """
        Calculates the Chamfer distance from the target wetted mesh to all candidate meshes.

        Args:
            target_wetted_mesh: The trimesh object of the target wetted surface.

        Returns:
            A dictionary mapping each candidate mesh name to its calculated Chamfer distance.
        """
        if self.base_mesh is None:
            raise BaseMeshIsNoneError(base_mesh_name=str(self.base_mesh_name))
        distances = {}
        logger.info(f"Calculating distances to {len(self.candidate_meshes)} candidate meshes...")

        for name, candidate_mesh in self.candidate_meshes.items():
            # The candidate mesh from the database is already the wetted surface.
            distance = _calculate_chamfer_distance(target_wetted_mesh, candidate_mesh.mesh)
            logger.debug(f"  - Calculated distance to '{name}': {distance:.4f}")
            distances[name] = distance

        return distances
