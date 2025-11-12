#!/usr/bin/env python3
"""
Extract apt package dependencies and their versions for a ROS package.xml file.

This module provides tools to analyze ROS package.xml files and extract dependency
information including apt package names, versions, and licenses.
"""

import argparse
import logging
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Optional


try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas is required to run this script. Please install it via 'pip install pandas'.")
    sys.exit(1)

try:
    import openpyxl
except ImportError:
    print(
        "ERROR: openpyxl is required to run this script (excel export). Please install it via 'pip install openpyxl'."
    )
    sys.exit(1)

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

DEPENDENCY_TAGS = [
    "build_depend",
    "build_export_depend",
    "buildtool_depend",
    "buildtool_export_depend",
    "depend",
    "doc_depend",
    "exec_depend",
    "run_depend",
    "test_depend",
]

# Command templates
ROSDEP_RESOLVE_CMD = ["rosdep", "resolve"]
ROS2_PKG_PREFIX_CMD = ["ros2", "pkg", "prefix"]
DPKG_STATUS_CMD = ["dpkg", "-s"]
SUBPROCESS_TIMEOUT = 5

# Path templates
COPYRIGHT_PATH_TEMPLATE = "/usr/share/doc/{}/copyright"
ROS_INDEX_URL_TEMPLATE = "https://index.ros.org/p/{}/"
DEFAULT_OUTPUT_DIR = "./output"

save_dir = Path(DEFAULT_OUTPUT_DIR)
save_dir.mkdir(exist_ok=True, parents=True)

# DataFrame column names
COLUMN_ROSDEP_KEY = "rosdep_key"
COLUMN_APT_PACKAGE_NAME = "apt_package_name"
COLUMN_VERSION = "version"
COLUMN_LICENSE = "license"
COLUMN_ROS_DEPENDENCY_TYPE = "ros_dependency_type"
COLUMN_IS_DIRECT_DEPENDENCY = "is_direct_dependency"
COLUMN_DEPENDENCY_LEVEL = "dependency_level"
COLUMN_IS_ROS_PACKAGE = "is_ros_package"
COLUMN_ROS_INDEX_URL = "ros_index_url"

# All columns for DataFrame initialization
DATAFRAME_COLUMNS = [
    COLUMN_ROSDEP_KEY,
    COLUMN_APT_PACKAGE_NAME,
    COLUMN_VERSION,
    COLUMN_LICENSE,
    COLUMN_ROS_DEPENDENCY_TYPE,
    COLUMN_IS_DIRECT_DEPENDENCY,
    COLUMN_DEPENDENCY_LEVEL,
]

save_dir: Path = Path(DEFAULT_OUTPUT_DIR) # global variable for output directory

# =============================================================================
# Helper Classes
# =============================================================================

class RosdepWrapper:
    """Handle rosdep command interactions."""

    @staticmethod
    def resolve_to_apt(rosdep_key: str) -> Optional[str]:
        """
        Resolve rosdep key to apt package name.

        Args:
            rosdep_key: The rosdep key to resolve

        Returns:
            The apt package name or None if resolution fails
        """
        try:
            result = subprocess.run(
                ROSDEP_RESOLVE_CMD + [rosdep_key],
                capture_output=True,
                text=True,
                timeout=SUBPROCESS_TIMEOUT,
                check=False,
            )

            if result.returncode != 0:
                logger.warning(f"rosdep resolve failed for {rosdep_key}: {result.stderr.strip()}")
                return None

            lines = [line for line in result.stdout.strip().split("\n") if not line.startswith("#")]

            assert len(lines) == 1, "Unexpected rosdep output format"

            apt_package = lines[0].strip()
            if not apt_package:
                logger.warning(f"No apt package found for rosdep key: {rosdep_key}")
                return None

            return apt_package

        except subprocess.TimeoutExpired:
            logger.error(f"Timeout resolving rosdep key: {rosdep_key}")
            return None
        except Exception as e:
            logger.error(f"Error error resolving {rosdep_key}: {type(e).__name__}: {e}")
            return None

    @staticmethod
    def is_ros_package(rosdep_key: str) -> bool:
        """
        Check if a rosdep key corresponds to a ROS package.

        Args:
            rosdep_key: The rosdep key to check

        Returns:
            True if the key resolves to a ROS package (starts with 'ros-'), False otherwise
        """
        try:
            apt_package = RosdepWrapper.resolve_to_apt(rosdep_key)
            if apt_package is None:
                return False
            return apt_package.startswith("ros-")
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout checking if {rosdep_key} is a ROS package")
            return False
        except Exception as e:
            logger.error(f"Error checking if {rosdep_key} is a ROS package: {e}")
            return False


class AptPackageUtils:
    """Handle apt package information retrieval."""

    @staticmethod
    def get_version(package_name: str) -> Optional[str]:
        """
        Get the installed version of an apt package.

        Args:
            package_name: The apt package name

        Returns:
            The version string or None if package not found or version cannot be determined
        """
        try:
            result = subprocess.run(
                DPKG_STATUS_CMD + [package_name],
                capture_output=True,
                text=True,
                timeout=SUBPROCESS_TIMEOUT,
                check=False,
            )

            if result.returncode != 0:
                logger.debug(f"Package {package_name} not found or not installed")
                return None

            for line in result.stdout.split("\n"):
                if line.startswith("Version:"):
                    version = line.split("Version:", 1)[1].strip()
                    return version

            logger.warning(f"Version not found in dpkg output for {package_name}")
            return None

        except subprocess.TimeoutExpired:
            logger.error(f"Timeout getting version for {package_name}")
            return None
        except Exception as e:
            logger.error(f"Error getting version for {package_name}: {e}")
            return None

    @staticmethod
    def get_license(package_name: str, output_dir: Optional[Path] = None) -> Optional[str]:
        """
        Get the license of an apt package.

        Args:
            package_name: The apt package name
            output_dir: Directory where copyright files should be saved

        Returns:
            Comma-separated license identifiers or None if license cannot be determined
        """
        if output_dir is None:
            output_dir = save_dir

        try:
            copyright_file = Path(COPYRIGHT_PATH_TEMPLATE.format(package_name))

            if not copyright_file.exists():
                logger.debug(f"Copyright file not found for {package_name}")
                return None

            copyright_dir = output_dir / "copyright"
            copyright_dir.mkdir(exist_ok=True)
            dest_file = copyright_dir / package_name

            try:
                shutil.copy(copyright_file, dest_file)
            except Exception as e:
                logger.error(f"Error copying copyright file for {package_name}: {e}")

            license_texts = []
            with copyright_file.open(encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("License:"):
                        license_text = line.split("License:", 1)[1].strip()
                        license_texts.append(license_text)

            return ", ".join(license_texts) if license_texts else None

        except Exception as e:
            logger.error(f"Error getting license for {package_name}: {e}")
            return None


def find_package_path(package_name: str) -> Optional[Path]:
    """
    Find the installation path of a ROS package.

    Args:
        package_name: The ROS package name

    Returns:
        Path to the package or None if not found
    """
    try:
        result = subprocess.run(
            ROS2_PKG_PREFIX_CMD + [package_name],
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT,
            check=False,
        )

        if result.returncode == 0:
            pkg_path = Path(result.stdout.strip())
            if pkg_path.exists():
                return pkg_path

        logger.debug(f"Package {package_name} not found")
        return None

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout finding package: {package_name}")
        return None
    except Exception as e:
        logger.error(f"Could not find package using ros2 pkg: {e}")
        return None


# =============================================================================
# Main Extractor Class
# =============================================================================


class ROSDependencyExtractor:
    """Extract and resolve ROS package dependencies to apt packages."""

    def __init__(self, package_xml_path: Path) -> None:
        """
        Initialize with path to package.xml.

        Args:
            package_xml_path: Path to the ROS package.xml file to analyze
        """
        self.package_xml_path = package_xml_path
        self.transitive_depth = 0
        self.df = pd.DataFrame(columns=DATAFRAME_COLUMNS)

    def run(self, transitive_depth: int = 0) -> None:
        """
        Execute the full extraction process.

        Args:
            transitive_depth: Depth to traverse transitive dependencies (0=direct only, -1=unlimited)
        """
        self.transitive_depth = transitive_depth

        rosdep_keys, dependency_type = self.parse_ros_package_xml_dependencies(self.package_xml_path)
        for rosdep_key, dep_type in zip(rosdep_keys, dependency_type):
            self._add_dependency(rosdep_key, dep_type, is_direct=True, level=0)

        if transitive_depth and self.transitive_depth != 0:
            logger.info(
                f"Starting transitive dependency scan (depth: {'unlimited' if self.transitive_depth == -1 else self.transitive_depth})"
            )
            self.get_transitive_dependencies_recursive(current_level=0)
            logger.info(f"Transitive scan complete. Total dependencies: {len(self.df)}")

        # Create is_ros_package column
        self.df[COLUMN_IS_ROS_PACKAGE] = self.df[COLUMN_ROSDEP_KEY].apply(RosdepWrapper.is_ros_package)

        # For ROS packages, add a ros_index_url column
        self.df[COLUMN_ROS_INDEX_URL] = self.df.apply(
            lambda row: ROS_INDEX_URL_TEMPLATE.format(row[COLUMN_ROSDEP_KEY]) if row[COLUMN_IS_ROS_PACKAGE] else None,
            axis=1,
        )

    def export(
        self,
        output_path: Path,
        exclude_dependency_tags: Optional[List[str]] = None,
    ) -> None:
        """
        Export results to Excel file.

        Args:
            output_path: Path where the Excel file should be saved
            exclude_dependency_tags: List of dependency tags to exclude from export
        """
        df_export = self.df.copy()

        if exclude_dependency_tags:
            df_export = df_export[~df_export[COLUMN_ROS_DEPENDENCY_TYPE].isin(exclude_dependency_tags)]
            logger.info(
                f"Exporting {len(df_export)} / {len(self.df)} dependencies after excluding: {exclude_dependency_tags}."
            )

        df_export.to_excel(output_path.with_suffix(".xlsx"), index=False)
        logger.info(f"Exported results to: {output_path.with_suffix('.xlsx').absolute()}")

    def print(self) -> None:
        """Print the dependency DataFrame to the console."""
        print(self.df)

    # ==========================================================================
    # Internal Methods
    # ==========================================================================

    def _add_dependency(
        self,
        rosdep_key: str,
        dependency_type: str,
        is_direct: bool = True,
        level: int = 0,
    ) -> None:
        """
        Add a dependency to the DataFrame.

        Args:
            rosdep_key: The rosdep key to add
            dependency_type: Type of dependency
            is_direct: Whether this is a direct dependency
            level: Dependency level
        """
        apt_key = RosdepWrapper.resolve_to_apt(rosdep_key)
        row = {
            COLUMN_ROSDEP_KEY: rosdep_key,
            COLUMN_APT_PACKAGE_NAME: apt_key,
            COLUMN_VERSION: AptPackageUtils.get_version(apt_key) if apt_key else None,
            COLUMN_LICENSE: AptPackageUtils.get_license(apt_key) if apt_key else None,
            COLUMN_ROS_DEPENDENCY_TYPE: dependency_type,
            COLUMN_IS_DIRECT_DEPENDENCY: is_direct,
            COLUMN_DEPENDENCY_LEVEL: level,
        }
        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)

    # ==========================================================================
    # Helper methods
    # ==========================================================================

    def parse_ros_package_xml_dependencies(self, package_xml_path: Path) -> Tuple[List[str], List[str]]:
        """
        Parse dependencies from a ROS package.xml file.

        Args:
            package_xml_path: Path to the package.xml file

        Returns:
            Tuple of (rosdep_keys, dependency_types)
        """
        logger.info(f"Extracting ROS dependency keys from: {package_xml_path}")

        if not package_xml_path.exists():
            raise Exception(f"package.xml not found at: {package_xml_path}")

        try:
            tree = ET.parse(package_xml_path)
            root = tree.getroot()

            dependencies = []
            for tag in DEPENDENCY_TAGS:
                for elem in root.findall(tag):
                    if elem.text:
                        rosdep_key = elem.text.strip()

                        dependencies.append((rosdep_key, tag))
            rosdep_keys = [dep[0] for dep in dependencies]
            dependency_types = [dep[1] for dep in dependencies]

            logger.info(f"Found {len(rosdep_keys)} dependencies in package.xml")
            return (rosdep_keys, dependency_types)
        except Exception as e:
            logger.error(f"Error parsing package.xml: {e}")

    def get_transitive_dependencies_recursive(self, current_level: int = 0) -> None:
        """
        Recursively resolve transitive dependencies up to specified depth.

        Args:
            current_level: Current recursion depth level
        """
        # Base case: Check if we should stop
        if self.transitive_depth != -1 and current_level >= self.transitive_depth:
            logger.info(f"Reached maximum transitive depth: {self.transitive_depth}")
            return

        logger.info(f"Scanning transitive dependencies at level {current_level + 1}")

        # Get all rosdep keys from previous level that are ROS packages
        if current_level == 0:
            # First level: get direct dependencies
            current_rosdep_keys = self.df[
                (self.df[COLUMN_IS_DIRECT_DEPENDENCY] == True) & (self.df[COLUMN_APT_PACKAGE_NAME].notna())
            ][COLUMN_ROSDEP_KEY].unique()
        else:
            # Subsequent levels: get dependencies from previous level
            current_rosdep_keys = self.df[
                (self.df[COLUMN_DEPENDENCY_LEVEL] == current_level) & (self.df[COLUMN_APT_PACKAGE_NAME].notna())
            ][COLUMN_ROSDEP_KEY].unique()

        if len(current_rosdep_keys) == 0:
            logger.info(f"No dependencies found at level {current_level + 1}, stopping")
            return

        # Track newly discovered dependencies
        new_dependencies_found = False
        seen_rosdep_keys = set(self.df[COLUMN_ROSDEP_KEY].dropna())

        for rosdep_key in current_rosdep_keys:
            # Only process ROS packages (not system packages)
            if not RosdepWrapper.is_ros_package(rosdep_key):
                logger.debug(f"Skipping non-ROS package: {rosdep_key}")
                continue

            # Find package path
            pkg_path = find_package_path(rosdep_key)
            if not pkg_path:
                logger.warning(f"Could not find package path for: {rosdep_key}")
                continue

            # Find package.xml in the package
            package_xml_candidates = [
                pkg_path / "share" / rosdep_key / "package.xml",
                pkg_path / "package.xml",
            ]

            package_xml_path = None
            for candidate in package_xml_candidates:
                if candidate.exists():
                    package_xml_path = candidate
                    break

            if not package_xml_path:
                logger.warning(f"Could not find package.xml for: {rosdep_key}")
                continue

            # Parse transitive dependencies
            try:
                transitive_keys, transitive_types = self.parse_ros_package_xml_dependencies(package_xml_path)

                # Add new dependencies to dataframe
                for trans_key, trans_type in zip(transitive_keys, transitive_types):
                    # Skip if we've already seen this dependency
                    if trans_key in seen_rosdep_keys:
                        continue

                    logger.debug(f"Found new transitive dependency: {trans_key} (from {rosdep_key})")

                    self._add_dependency(trans_key, trans_type, is_direct=False, level=current_level + 1)
                    seen_rosdep_keys.add(trans_key)
                    new_dependencies_found = True

            except Exception as e:
                logger.error(f"Error processing transitive dependencies for {rosdep_key}: {e}")
                continue

        # Recursive call for next level
        if self.transitive_depth == -1:
            # Unlimited depth: continue only if new dependencies were found
            if new_dependencies_found:
                logger.info(f"Found new dependencies at level {current_level + 1}, continuing...")
                self.get_transitive_dependencies_recursive(current_level + 1)
            else:
                logger.info(f"No new dependencies at level {current_level + 1}, stopping")
        else:
            # Limited depth: continue until we reach the limit
            self.get_transitive_dependencies_recursive(current_level + 1)


# =============================================================================
# Configuration and Setup Functions
# =============================================================================


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging with appropriate format and level.

    Args:
        verbose: Enable DEBUG level logging if True, INFO otherwise
    """
    log_level = logging.DEBUG if verbose else logging.INFO

    # Create console handler with formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)

    # Configure root logger
    logging.basicConfig(level=log_level, handlers=[console_handler])


def get_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Extract ROS2 package dependencies and versions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # usage="./%(prog)s --package-xml-path /path/to/package.xml [options]",
    )

    parser.add_argument(
        "--package-xml-path", type=Path, help="Absolute path to package.xml of a ROS package.", required=True
    )

    parser.add_argument(
        "--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR), help="Directory to save output files."
    )

    parser.add_argument(
        "--exclude-dependency-tags",
        nargs="+",
        metavar="TAG",
        help="Exclude specific dependency tags from final report."
             f"\nAvailable tags: {', '.join(DEPENDENCY_TAGS)}. "
             f"Default: doc_depend, test_depend",
        choices=DEPENDENCY_TAGS,
        default=[
            "doc_depend",
            "test_depend",
        ],
    )

    parser.add_argument(
        "--transitive-depth",
        type=int,
        default=0,
        help="Depth of transitive dependencies to scan. 0=direct only (default), N=N levels deep, -1=unlimited (scan until no new dependencies found)",
    )

    parser.add_argument("--export", action="store_true", help="Export results to a file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logging")

    return parser


def main() -> None:
    """
    Main entry point for the ROS dependency extractor.

    Parses command-line arguments, configures logging, extracts dependencies,
    and optionally exports results to Excel files.
    """
    parser = get_parser()
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)

    # Configure save directory
    global save_dir
    save_dir = Path(args.output_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Output directory set to: {save_dir.absolute()}")

    logger.info(f"Scanning package at: {args.package_xml_path}")
    try:
        extractor = ROSDependencyExtractor(args.package_xml_path)
        extractor.run(args.transitive_depth)
        extractor.print()

        if args.export:
            # Export all results
            extractor.export(save_dir / f"{args.package_xml_path.parent.name}_dependencies")

            # Export filtered results
            extractor.export(
                save_dir / f"{args.package_xml_path.parent.name}_dependencies_filtered",
                exclude_dependency_tags=args.exclude_dependency_tags,
            )

    except Exception as e:
        logger.error(f"Dependency extraction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
