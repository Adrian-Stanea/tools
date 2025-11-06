# ROS Dependency Parser

A Python tool to extract and analyze ROS2 package dependencies, including apt packages, versions, and licenses.

## Description

This tool parses ROS `package.xml` files to extract dependency information and resolves them to system apt packages. It provides detailed information about each dependency including version numbers, licenses, and can traverse transitive dependencies.

## Features

- ✅ **Dependency Extraction**: Parse all dependency types from ROS package.xml files
- ✅ **Apt Resolution**: Resolve ROS dependencies to apt package names using rosdep
- ✅ **Version Detection**: Retrieve installed version information for apt packages
- ✅ **License Information**: Extract and save license information from copyright files
- ✅ **Transitive Dependencies**: Recursively scan transitive dependencies with configurable depth
- ✅ **ROS Package Detection**: Identify which dependencies are ROS packages
- ✅ **Excel Export**: Export results to formatted Excel spreadsheets
- ✅ **Filtering**: Exclude specific dependency tags (e.g., test, doc dependencies)

## Requirements

- Ubuntu system with ROS2 environment configured
- Python 3.X
- Required Python packages (see `requirements.txt`)

### NOTE

The script has been tested on Ubuntu 22.04 LTS and ROS2 Humble. Make sure to source
the ROS2 environment before running this script

```bash
source /opt/ros/humble/setup.sh
```

## Installation

```bash
# Clone the repository
# git clone <repo_url>
cd ros-dependency-parser

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

Extract dependencies from a package.xml file:

```bash
./extract_ros_dependencies.py --package-xml-path </absolute/path/to/package.xml>
```

### Export to Excel

Export results to Excel files:

```bash
./extract_ros_dependencies.py \
    --package-xml-path </absolute/path/to/package.xml> \
    --export
```

### Transitive Dependencies

Scan transitive dependencies (up to 2 levels deep):

```bash
./extract_ros_dependencies.py \
    --package-xml-path </absolute/path/to/package.xml> \
    --transitive-depth 2 \
    --export
```

Scan all transitive dependencies (unlimited depth):

```bash
./extract_ros_dependencies.py \
    --package-xml-path </absolute/path/to/package.xml> \
    --transitive-depth -1 \
    --export
```

### Filtering Dependencies

Exclude specific dependency tags:

```bash
./extract_ros_dependencies.py \
    --package-xml-path </absolute/path/to/package.xml> \
    --exclude-dependency-tags doc_depend test_depend \
    --export
```

### Verbose Logging

Enable debug logging:

```bash
./extract_ros_dependencies.py \
    --package-xml-path </absolute/path/to/package.xml> \
    --verbose
```

### Get Help

```bash
./extract_ros_dependencies.py --help
```

## Output

The tool generates:

1. **Console Output**: DataFrame printed to terminal showing all dependencies
2. **Excel Files** (when --export is used):
   - `{package_name}_dependencies.xlsx`: All dependencies
   - `{package_name}_dependencies_filtered.xlsx`: Filtered dependencies (excluding specified tags)
3. **Copyright Files**: License files saved in `output/copyright/` directory

### Output Columns

- `rosdep_key`: ROS dependency key from package.xml
- `apt_package_name`: Resolved apt package name
- `version`: Installed version of the apt package
- `license`: License information from copyright file
- `ros_dependency_type`: Type of dependency (e.g., build_depend, exec_depend)
- `is_direct_dependency`: True for direct dependencies, False for transitive
- `dependency_level`: Depth level (0 for direct, 1+ for transitive)
- `is_ros_package`: True if the dependency is a ROS package
- `ros_index_url`: URL to ROS Index page (for ROS packages only)

## Testing

Run the unit tests:

```bash
python3 -m unittest tests.test_extract_ros_dependencies  -v
```

Run a subset of tests:
```bash
python3 -m unittest tests.test_extract_ros_dependencies.TestFindPackagePath  -v
```


## Workflow

1. Build your ROS package to resolve dependencies
2. Run this script to extract the dependency list
3. Review the generated Excel files
4. Check license information in the copyright directory