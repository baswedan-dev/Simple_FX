import yaml
from pathlib import Path
from typing import Dict, Any
import os

def load_config(config_file: str = 'config.yml') -> Dict[str, Any]:
    """
    Load YAML configuration file with environment variable substitution

    Args:
        config_file: Configuration file name

    Returns:
        Parsed configuration dictionary
    """
    config_path = Path(__file__).parent.parent.parent / 'config' / config_file

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Substitute environment variables
    if config:
        config = _substitute_env_vars(config)

    return config

def _substitute_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively substitute environment variables in config

    Args:
        config: Configuration dictionary

    Returns:
        Configuration with substituted values
    """
    if isinstance(config, dict):
        for key, value in config.items():
            if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                env_var = value[2:-1]
                config[key] = os.getenv(env_var, value)
            elif isinstance(value, (dict, list)):
                config[key] = _substitute_env_vars(value)
    elif isinstance(config, list):
        for i, item in enumerate(config):
            if isinstance(item, (dict, list)):
                config[i] = _substitute_env_vars(item)

    return config

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration structure and values

    Args:
        config: Configuration dictionary

    Returns:
        True if configuration is valid

    Raises:
        ValueError: If configuration is invalid
    """
    # Phase 1 only needs data section
    required_sections = ['data']

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    # Validate data section
    data_config = config['data']
    if data_config['max_gap_forward_fill'] > 5:
        raise ValueError("max_gap_forward_fill cannot exceed 5 bars")

    if data_config['cache_ttl_hours'] < 1:
        raise ValueError("cache_ttl_hours must be at least 1 hour")

    # Validate logging section if present
    if 'logging' in config:
        logging_config = config['logging']
        if 'level' in logging_config:
            valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            if logging_config['level'] not in valid_levels:
                raise ValueError(f"Invalid log level: {logging_config['level']}")

    return True