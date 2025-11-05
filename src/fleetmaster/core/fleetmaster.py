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
import trimesh.transformations as tf
from mafredo import Hyddb1

from .engine import (
    MESH_GROUP_NAME,
    EngineMesh,
    _apply_mesh_translation_and_rotation,
    _prepare_capytaine_body,
    load_meshes_from_hdf5,
)
from .exceptions import DatabaseFileNotFoundError, HDF5AttributeError, MeshLoadError
from .fitting import _calculate_chamfer_distance
from .settings import MeshConfig

logger = logging.getLogger(__name__)


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

        self._water_depth: float = np.inf
        self._velocity: float = 0.0
        self._origin: tuple[float, float, float] = (0.0, 0.0, 0.0)

        self._best_match_name: str | None = None
        self._match_error: float = np.inf
        self._best_match_hydro_data: dict[str, Any] | None = None

        self._load_database_meshes()

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
            self._load_hydro_data(best_match)
        else:
            logger.warning("Fitting complete, but no suitable match was found.")

    def _load_hydro_data(self, mesh_name: str) -> None:
        """Loads hydrodynamic data for the given mesh name from the HDF5 file."""

        # Helper to format values for the group name, similar to engine.py
        def _format_value_for_name(value: float) -> str:
            if value == np.inf:
                return "inf"
            if value == int(value):
                return str(int(value))
            return f"{value:.1f}"

        # The water level for comparison is hardcoded to 0.0 in _find_best_fit_for_candidates
        water_level = 0.0

        # Construct the case-specific group name at the top level
        wd = _format_value_for_name(self._water_depth)
        wl = _format_value_for_name(water_level)
        fs = _format_value_for_name(self._velocity)
        group_path = f"{mesh_name}_wd_{wd}_wl_{wl}_fs_{fs}"

        logger.info(f"Attempting to load hydrodynamic data for mesh '{mesh_name}' from case group '{group_path}'...")

        try:
            with h5py.File(self.filename, "r") as f:
                if group_path not in f:
                    logger.error(f"Case group '{group_path}' not found in HDF5 file.")
                    self._best_match_hydro_data = None
                    return

                group = f[group_path]
                required_datasets = [
                    "omega",
                    "added_mass",
                    "damping",
                    "directions",
                    "force_amps",
                    "force_phase_rad",
                ]

                # Check for presence of all required datasets before loading
                if not all(ds_name in group for ds_name in required_datasets):
                    logger.error(
                        f"One or more required hydrodynamic datasets not found in group '{group_path}'. "
                        "The HDF5 file might be corrupted or incomplete."
                    )
                    self._best_match_hydro_data = None
                    return

                hydro_data = {ds_name: group[ds_name][()] for ds_name in required_datasets}
                self._best_match_hydro_data = hydro_data
                logger.info(f"Successfully loaded hydrodynamic data from group '{group_path}'.")

        except Exception:
            logger.exception(f"Failed to load hydrodynamic data for mesh '{mesh_name}' from group '{group_path}'")
            self._best_match_hydro_data = None

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

    def get_hyddb1(
        self,
    ) -> tuple[Any | None, tuple[float, float, float] | None, float | None, float | None]:
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
            logger.warning("No hydrodynamic data loaded. Run fit() first.")
            return None, None, None, None

        try:
            hyddb = Hyddb1()
            hyddb.set_data(
                omega=self._best_match_hydro_data["omega"],
                added_mass=self._best_match_hydro_data["added_mass"],
                damping=self._best_match_hydro_data["damping"],
                directions=self._best_match_hydro_data["directions"],
                force_amps=self._best_match_hydro_data["force_amps"],
                force_phase_rad=self._best_match_hydro_data["force_phase_rad"],
            )

            application_point = self._origin
            velocity = self._velocity
            waterdepth = self._water_depth

        except Exception:
            logger.exception("Failed to create Hyddb1 object:")
            return None, None, None, None
        else:
            return hyddb, application_point, velocity, waterdepth

    def _load_database_meshes(self) -> None:
        """
        Loads all meshes from the HDF5 database file.
        """
        if not self.filename.exists():
            raise DatabaseFileNotFoundError(path=self.filename)

        base_mesh_name_str: str | None = None
        with h5py.File(self.filename, "r") as f:
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

        if not base_mesh_name_str or not candidate_mesh_names:
            logger.warning("No base mesh or candidate meshes found to perform a match.")
            return

        all_meshes_to_load = [base_mesh_name_str, *candidate_mesh_names]
        self._loaded_meshes = {
            mesh.metadata["name"]: mesh for mesh in load_meshes_from_hdf5(self.filename, all_meshes_to_load)
        }

        self.base_mesh_name = base_mesh_name_str
        self.base_mesh = self._loaded_meshes.get(self.base_mesh_name)
        if not self.base_mesh:
            raise MeshLoadError(mesh_name=self.base_mesh_name)

        self.candidate_meshes = {
            name: mesh for name, mesh in self._loaded_meshes.items() if name != self.base_mesh_name
        }

    def _find_best_matching_mesh(
        self,
        target_translation: list[float],
        target_rotation: list[float],
    ) -> tuple[str | None, float]:
        """
        Finds the best matching mesh from an HDF5 database for a given target transformation.

        The function works as follows:
        1.  For each candidate, transform the base_mesh using a hybrid transformation:
            - XY-translation and Z-rotation from the candidate.
            - Z-translation and XY-rotation from the target transformation.
        2.  Compute the Chamfer distance between the wetted surface of the transformed
            base mesh and the wetted surface of the candidate mesh.
        3.  Return the name of the mesh with the smallest Chamfer distance.

        Args:
            hdf5_path (Path): Path to the HDF5 database file.
            target_translation (list[float]): The target translation [x, y, z] to apply to the base mesh.
            target_rotation (list[float]): The target rotation [roll, pitch, yaw] in degrees.
            water_level (float): The water level to use for cutting the meshes for comparison. Defaults to 0.0.

        Returns:
            A tuple containing the name of the best matching mesh and the corresponding Chamfer distance.
            Returns (None, np.inf) if no match is found.
        """

        if not self.base_mesh or not self.candidate_meshes:
            logger.warning("Base mesh or candidate meshes not loaded. Cannot find best match.")
            return None, np.inf

        # 3. Find the distances for all candidates based on the new logic
        all_distances = self._find_best_fit_for_candidates(
            target_translation=target_translation,
            target_rotation=target_rotation,
            water_level=0.0,
        )

        # 4. Find the minimum distance among the results
        if not all_distances:
            logger.warning("No distances could be calculated.")
            return None, np.inf

        best_match_name = min(all_distances, key=lambda k: all_distances[k])
        min_distance = all_distances[best_match_name]

        logger.info(f"Best match found: '{best_match_name}' with a Chamfer distance of {min_distance:.4f}")
        return best_match_name, min_distance

    def _find_best_fit_for_candidates(
        self,
        target_translation: list[float],
        target_rotation: list[float],
        water_level: float,
    ) -> dict[str, float]:
        """
        Finds the best fit for a base mesh against a set of candidate meshes.

        For each candidate, this function transforms a copy of the base mesh using a hybrid
        transformation derived from the candidate and a target transformation.

        - XY translation from the candidate, Z translation from the target.
        - Z rotation (yaw) from the candidate, XY rotation (roll, pitch) from the target.

        It then calculates the Chamfer distance between the wetted surfaces.

        Args:
            base_mesh: The base trimesh object.
            candidate_meshes: A dictionary mapping mesh names to their trimesh objects.
            target_translation: The target global translation [x, y, z].
            target_rotation: The target global rotation [roll, pitch, yaw] in degrees.
            water_level: The water level at which to cut the mesh for a fair comparison.

        Returns:
            A dictionary mapping each candidate mesh name to its calculated Chamfer distance.
        """
        distances = {}
        logger.info(f"Finding best fit for {len(self.candidate_meshes)} candidate meshes...")

        for name, candidate_mesh in self.candidate_meshes.items():
            candidate_translation = candidate_mesh.metadata.get("translation")
            candidate_rotation = candidate_mesh.metadata.get("rotation")

            if candidate_translation is None or candidate_rotation is None:
                logger.warning(f"Candidate '{name}' is missing translation/rotation metadata. Skipping.")
                distances[name] = np.inf
                continue

            # The goal of the fitting is to find a mesh from the database that best matches
            # the target's submerged shape, which is primarily determined by Z-translation (draft)
            # and X/Y-rotations (roll, pitch). The database contains meshes with varying roll and pitch,
            # but typically constant XY translation and Z-rotation (yaw).
            #
            # To find the best match, we create a hybrid transformation that respects these assumptions:
            # - We use the target's Z-translation (draft) because that's a key property we're matching.
            # - We use the target's roll and pitch for the same reason.
            # - We take the candidate's XY-translation and yaw, because these are considered irrelevant
            #   for the shape matching and are constant in the database generation process.
            #
            # This allows us to transform the base mesh into a shape that is directly comparable
            # with the candidate's wetted surface.
            new_translation = [
                candidate_translation[0],
                candidate_translation[1],
                target_translation[2],
            ]
            new_rotation = [
                target_rotation[0],
                target_rotation[1],
                candidate_rotation[2],
            ]

            temp_base_mesh = self.base_mesh.copy()
            transformed_base_mesh = _apply_mesh_translation_and_rotation(
                mesh=temp_base_mesh,
                translation_vector=new_translation,
                rotation_vector_deg=new_rotation,
            )

            # Create a dummy EngineMesh to use the _prepare_capytaine_body function for cutting the mesh.
            dummy_config = MeshConfig(file="dummy")
            engine_mesh_for_cutting = EngineMesh(name="temp_base", mesh=transformed_base_mesh, config=dummy_config)

            _, cut_transformed_base_mesh = _prepare_capytaine_body(
                engine_mesh=engine_mesh_for_cutting,
                lid=False,
                grid_symmetry=False,
                water_level=water_level,
            )

            if not cut_transformed_base_mesh or len(cut_transformed_base_mesh.vertices) == 0:
                logger.warning(
                    f"Transformed base mesh for candidate '{name}' is out of the water. Assigning infinite distance."
                )
                distances[name] = np.inf
                continue

            # The candidate mesh from the database is already the wetted surface.
            distance = _calculate_chamfer_distance(cut_transformed_base_mesh, candidate_mesh)
            logger.debug(f"  - Calculated distance to '{name}': {distance:.4f}")
            distances[name] = distance

        return distances
