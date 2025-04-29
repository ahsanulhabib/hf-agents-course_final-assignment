import yaml
from pathlib import Path
import sys


_config_data = None
_config_path = None


def find_config_path() -> Path:
    """Tries to find the config.yaml file."""
    # Start from the directory of this file
    current_dir = Path(__file__).parent
    # Check common locations relative to this file's package structure
    # 1. Project root (../.. from here)
    project_root_path = current_dir.parent.parent / "config.yaml"
    if project_root_path.is_file():
        return project_root_path

    # 2. Current working directory (less reliable but a fallback)
    cwd_path = Path.cwd() / "config.yaml"
    if cwd_path.is_file():
        return cwd_path

    raise FileNotFoundError(
        "Configuration file 'config.yaml' not found in project root or current directory."
    )


def load_config() -> dict:
    """Loads the YAML configuration file and caches it."""
    global _config_data, _config_path
    if _config_data is not None:
        return _config_data

    try:
        _config_path = find_config_path()
        print(f"Loading configuration from: {_config_path}")
        with open(_config_path, "r", encoding="utf-8") as f:
            _config_data = yaml.safe_load(f)

        if not isinstance(_config_data, dict):
            raise TypeError(
                f"Config file '{_config_path}' did not parse as a dictionary."
            )

        print("Configuration loaded successfully.")
        return _config_data

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)  # Exit if config file is essential and not found
    except yaml.YAMLError as e:
        print(f"ERROR: Error parsing YAML file '{_config_path}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Error loading configuration file '{_config_path}': {e}")
        sys.exit(1)


# Load config once when this module is imported
CONFIG = load_config()


# Optional: Helper functions to access nested keys safely
def get_config_value(keys, default=None):
    """Safely get a nested value from the loaded config."""
    data = CONFIG
    try:
        for key in keys:
            data = data[key]
        return data
    except (KeyError, TypeError):
        return default


if __name__ == "__main__":
    # Example usage of the config loader
    print("Loaded configuration:", CONFIG)
    print(
        "Example config value:", get_config_value(["example", "key"], "default_value")
    )
