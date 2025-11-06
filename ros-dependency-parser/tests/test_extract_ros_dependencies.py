#!/usr/bin/env python3
"""
Unit tests for ROS dependency extractor.

This module contains unit tests for the ROSDependencyExtractor class and
its helper classes (RosdepResolver, AptPackageUtils).
"""

import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


from extract_ros_dependencies import (
    ROSDependencyExtractor,
    RosdepWrapper,
    AptPackageUtils,
    find_package_path,
)


class TestRosdepWrapper(unittest.TestCase):
    """Test cases for RosdepWrapper class."""

    def test_resolve_to_apt_success(self):
        """Test successful resolution of rosdep key to apt package."""
        result = RosdepWrapper.resolve_to_apt("rclcpp")
        self.assertEqual(result, "ros-humble-rclcpp")

    def test_resolve_to_apt_failure(self):
        """Test resolution failure."""
        result = RosdepWrapper.resolve_to_apt("nonexistent")
        self.assertIsNone(result)

    def test_is_ros_package_true(self):
        """Test identification of ROS package."""
        result = RosdepWrapper.is_ros_package("rclcpp")
        self.assertTrue(result)

    def test_is_ros_package_false(self):
        """Test identification of non-ROS package."""
        result = RosdepWrapper.is_ros_package("libiio-dev")
        self.assertFalse(result)

    def test_is_ros_package_nonexistent(self):
        """Test identification of non-existent package."""
        result = RosdepWrapper.is_ros_package("nonexistent-package-xyz")
        self.assertFalse(result)


class TestAptPackageUtils(unittest.TestCase):
    """Test cases for AptPackageUtils class."""

    def test_get_version_success(self):
        """Test successful version retrieval."""
        result = AptPackageUtils.get_version("libiio-dev")

        self.assertIsNotNone(result)
        self.assertRegex(str(result), r"^\d+\.\d+(\.\d+)?")

    def test_get_version_not_found(self):
        """Test version retrieval for non-existent package."""
        result = AptPackageUtils.get_version("nonexistent")
        self.assertIsNone(result)

    def test_get_license_file(self):
        """Test license file retrieval."""
        result = AptPackageUtils.get_license("libiio-dev")
        self.assertIsNotNone(result)
        print(f"License type: {result}")
        # Should output copyright files in the output directory for inspection
        self.assertTrue(Path("output").exists())

    def test_get_license_file_nonexistent(self):
        """Test license file retrieval for non-existent package."""
        result = AptPackageUtils.get_license("nonexistent-package-xyz")
        self.assertIsNone(result)


class TestFindPackagePath(unittest.TestCase):
    """Test cases for find_package_path function."""

    def test_find_package_path_success(self):
        """Test successful package path finding."""
        result = find_package_path("rclcpp")
        print(result)
        self.assertEqual(result, Path("/opt/ros/humble"))

    @patch("subprocess.run")
    def test_find_package_path_not_found(self, mock_run):
        """Test package path finding for non-existent package."""
        mock_result = Mock()
        result = find_package_path("nonexistent")
        print(result)
        self.assertIsNone(result)


class TestROSDependencyExtractor(unittest.TestCase):
    """Test cases for ROSDependencyExtractor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_package_xml = Path(__file__).parent / "test_data/package.xml"
        self.transitive_depth = 0

    def test_initialization(self):
        """Test extractor initialization."""
        extractor = ROSDependencyExtractor(self.test_package_xml)
        extractor.run(transitive_depth=self.transitive_depth)

        df = extractor.df
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

    def test_parse_missing_package_xml(self):
        """Test handling of missing package.xml file."""
        extractor = ROSDependencyExtractor(Path("missing/path/package.xml"))

        with self.assertRaises(Exception):
            extractor.run(transitive_depth=self.transitive_depth)


if __name__ == "__main__":
    unittest.main()
