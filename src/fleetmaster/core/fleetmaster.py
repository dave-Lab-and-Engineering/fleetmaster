"""
This module defines the main FleetMaster class, which provides the primary API
for interacting with FleetMaster databases.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import trimesh.transformations as tf

from .fitting import find_best_matching_mesh

logger = logging.getLogger(__name__)


class FleetMaster:
    """
    The main class for interacting with a FleetMaster database.

    This class handles loading the database, setting parameters, and running
    the fitting process to find the best matching mesh.
    """

    def __init__(self, filename: str | Path):
        """
        Initializes the FleetMaster object.

        Args:
            filename: The path to the HDF5 database file.
        """
        self.filename = Path(filename)
        self._water_depth: float = -1  # infinite
        self._velocity: float = 0.0
        self._origin: tuple[float, float, float] = (0.0, 0.0, 0.0)

        self._best_match_name: str | None = None
        self._match_error: float = np.inf

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

    def fit(self, transform: Any, origin: tuple[float, float, float] | None = None) -> None:
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

        best_match, min_distance = find_best_matching_mesh(
            hdf5_path=self.filename,
            target_translation=translation,
            target_rotation=rotation_deg,
            water_level=self._water_depth,  # Assuming water_depth is the water_level for fitting
        )

        self._best_match_name = best_match
        self._match_error = min_distance

        if best_match:
            logger.info(f"Fitting complete. Best match: '{best_match}' with error: {min_distance:.4f}")
        else:
            logger.warning("Fitting complete, but no suitable match was found.")

    def get_match_error(self) -> float:
        """
        Returns the error of the best match from the fitting process.

        Returns:
            The Chamfer distance of the best match. Returns np.inf if no
            fit has been performed or no match was found.
        """
        return self._match_error

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
