#!/usr/bin/env python3
import sys
import subprocess
import importlib
import pkg_resources

# -----------------------------------------------------------------------------
# A dictionary mapping each plugin to its required packages.
# Packages can be specified either without a version (e.g., "requests")
# or with an exact version requirement (e.g., "pandas==1.5.0").
#
# Example:
#   "plugin_a": ["requests", "numpy==1.21.0"]
# -----------------------------------------------------------------------------
PLUGIN_DEPENDENCIES = {
    "plugin_a": ["requests", "numpy"],
    "plugin_b": ["pandas==2.0.3", "pyyaml"]
}

def ensure_exact_version_installed(package_spec):
    """
    Checks if package_spec (e.g., 'numpy==1.21.0') is installed at the correct version.
    If not installed or if version is incorrect, it installs/upgrades via pip.
    """
    try:
        # This will raise DistributionNotFound or VersionConflict if there's a mismatch
        pkg_resources.require([package_spec])
    except pkg_resources.DistributionNotFound:
        # Package is not installed at all
        print(f"[INFO] Installing missing dependency: {package_spec}")
        subprocess.run([sys.executable, "-m", "pip", "install", package_spec], check=True)
    except pkg_resources.VersionConflict:
        # Wrong version installed
        print(f"[INFO] Upgrading '{package_spec}' to the required version.")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", package_spec], check=True)

def ensure_dependencies_installed(plugin_names=None):
    """
    Ensures that all dependencies for the given plugin(s) are installed.
    If no plugin_names are provided, checks for all known plugins.
    """

    # If no specific plugins are requested, check dependencies for all plugins
    if not plugin_names:
        plugin_names = list(PLUGIN_DEPENDENCIES.keys())

    # Accumulate a list of all required packages for these plugins
    packages_to_check = []
    for plugin in plugin_names:
        required_packages = PLUGIN_DEPENDENCIES.get(plugin, [])
        packages_to_check.extend(required_packages)

    # Use a set to avoid duplicates
    packages_to_check = set(packages_to_check)

    # Check for each package:
    for pkg_spec in packages_to_check:
        if "==" in pkg_spec:
            # If the package has a version spec (e.g. "numpy==1.21.0"),
            # use our exact version check
            ensure_exact_version_installed(pkg_spec)
        else:
            # Otherwise, do a simpler import check (only presence, not version)
            try:
                # Some PyPI packages may import under a different name (e.g., 'PyYAML' -> import 'yaml'),
                # so you may need adjustments if package name != import name.
                importlib.import_module(pkg_spec)
            except ImportError:
                print(f"[INFO] Installing missing dependency: {pkg_spec}")
                subprocess.run([sys.executable, "-m", "pip", "install", pkg_spec], check=True)

if __name__ == "__main__":
    # Example usage:
    #   python check_dependencies.py plugin_a plugin_b
    # If called without arguments, it checks all known plugins in PLUGIN_DEPENDENCIES.
    plugin_args = sys.argv[1:]
    ensure_dependencies_installed(plugin_args)
