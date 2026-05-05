"""Unit tests for the mesh fitting functionality."""

import logging
from pathlib import Path

import numpy as np
import pytest
import trimesh.transformations as tf

from fleetmaster import FleetMaster

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def regression_hdf5_path() -> Path:
    """
    Returns the path to the pre-generated HDF5 database for regression tests.
    This file is expected to be committed to the repository.
    """
    return Path("tests/data/boxship.hdf5").resolve()


# --- Original Test Cases ---
# Define test cases as tuples of (description, target_translation, target_rotation, water_level, expected_match, expected_distance)

TEST_CASES = [
    (
        "Case 1: Exact Match Draft 1 meter",
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 0.0],
        0.0,
        "boxship_t_1_r_00_00_00",
        0.6078757444881737,
    ),
    (
        "Case 2: Match with irrelevant translation/rotation noise (draft 1.0)",
        [2.5, -4.2, -1.1],
        [0.0, 0.0, 15.0],
        0.0,
        "boxship_t_1_r_00_00_00",
        0.5476613062986713,
    ),
    (
        "Case 3: Different match due to significant rotation deviation (draft 1.0)",
        [2.5, -4.2, -1.1],
        [23.0, 19.0, 15.0],
        0.0,
        "boxship_t_1_r_00_10_00",
        0.6653554389732835,
    ),
    (
        "Case 4: Exact Match for draft 2.0",
        [0.0, 0.0, -2.0],
        [0.0, 0.0, 0.0],
        0.0,
        "boxship_t_1_r_00_00_00",
        0.0,
    ),
    (
        "Case 5: Exact Match for draft 2.0 with irrelevant xy-plane and yaw deviation",
        [10.0, -20.0, -2.0],
        [0.0, 0.0, 15.0],
        0.0,
        "boxship_t_1_r_00_00_00",
        0.0,
    ),
    (
        "Case 6: Match for draft 2.0 with noise in all axes",
        [10.0, -20.0, -2.2],
        [4.0, -1.0, 15.0],
        0.0,
        "boxship_t_1_r_00_00_00",
        0.16393340619270197,
    ),
]


@pytest.mark.parametrize(
    "description, target_translation, target_rotation, water_level, expected_match, expected_distance",
    TEST_CASES,
    ids=[case[0] for case in TEST_CASES],
)
def test_fleetmaster_fitting(
    regression_hdf5_path: Path,
    description: str,
    target_translation: list[float],
    target_rotation: list[float],
    water_level: float,
    expected_match: str,
    expected_distance: float,
):
    """Tests the FleetMaster mesh fitting with various scenarios."""
    logger.info(f"Running test: {description}")

    fm = FleetMaster(filename=regression_hdf5_path)

    fm.set_waterdepth(water_level)
    fm.set_velocity(0.0)

    angles_rad = np.deg2rad(target_rotation)
    rot = tf.euler_matrix(angles_rad[0], angles_rad[1], angles_rad[2], axes="sxyz")
    transform = rot.copy()
    transform[0:3, 3] = target_translation

    fm.fit_mesh(transform=transform)

    best_match = fm.get_best_match_name()
    distance = fm.get_match_error()

    assert best_match is not None, "A best match should have been found."
    assert best_match == expected_match
    np.testing.assert_almost_equal(
        distance,
        expected_distance,
        decimal=5,
        err_msg=f"Distance check failed for {description}. Got distance: {distance}, expected: {expected_distance}.",
    )
