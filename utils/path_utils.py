from __future__ import annotations

import yaml # type: ignore

def load_config(file_path: str) -> dict:
    """Load YAML configuration from a file.

    Args:
        file_path (str): Path to the YAML configuration file.

    Returns:
        dict: The loaded configuration.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config