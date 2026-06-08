import sys
import json

def load_config(path):
    with open(path, "r") as f:
        config = json.load(f)
    return config

def parse_value(value):
    # if value.lower() == "none":
    #     return None
    # Do not need yet, but can enable and be setup

    if value.lower() == "true":
        return True

    if value.lower() == "false":
        return False

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass
    
    # String
    return value

def apply_overrides(config, overrides):
    for path, value in overrides.items():
        keys = path.split(".")

        current = config  # Reset for every override

        for key in keys[:-1]:
            if key not in current:
                raise KeyError(f"Path '{path}' not found in config")

            if not isinstance(current[key], dict):
                raise KeyError(f"'{key}' in path '{path}' is not a dictionary")

            current = current[key]

        if keys[-1] not in current:
            raise KeyError(f"Path '{path}' not found in config")

        current[keys[-1]] = value

    return config

def apply_cli_overrides(config):
    """
    Override nested config values using dot notation.

    Example:
        python main.py training.epochs=12 model.hidden_dim=256
    """

    overrides = {}

    for arg in sys.argv[1:]:
        if "=" not in arg:
            continue

        path, value = arg.split("=", 1)

        value = parse_value(value)
        overrides[path] = value

    if overrides:
        config = apply_overrides(config, overrides)
        print("\n=== CLI Overrides ===")
        for k, v in overrides.items():
            print(f"{k}: {v}")
        print("=====================\n")

    return config
