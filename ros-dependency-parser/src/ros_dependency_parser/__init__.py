"""
ROS2 Dependency Parser

A tool to parse ROS2 package dependencies, versions, and licenses in Ubuntu environments.
"""
from .extract_ros_dependencies import (
    ROSDependencyExtractor,
    RosdepWrapper,
    AptPackageUtils,
    find_package_path,
)

__version__ = "0.1.0"
__author__ = "Adrian Stanea"
__email__ = "Adrian.Stanea@analog.com"

__all__ = [
    # Main class
    "ROSDependencyExtractor",
    # Helper classes
    "RosdepWrapper",
    "AptPackageUtils",
    # Functions
    "find_package_path",
]