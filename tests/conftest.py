"""Shared pytest fixtures for pointclouds."""

from __future__ import annotations

import numpy as np
import pytest

from pointclouds import PointCloud


@pytest.fixture
def tiny_cloud() -> PointCloud:
    """A four-point cloud with classification, for core-path tests."""
    return PointCloud(
        x=np.array([0.0, 1.0, 2.0, 3.0]),
        y=np.array([0.0, 1.0, 0.0, 1.0]),
        z=np.array([5.0, 2.0, 4.0, 1.0]),
        classification=np.array([2, 5, 2, 6], dtype=np.uint8),
    )
