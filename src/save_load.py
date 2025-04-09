import json
import os

# Directory to store parameter JSON files
param_dir = "project/params"

# Ensure the directory exists
os.makedirs(param_dir, exist_ok=True)

def save_params(params, model_name):
    """
    Save the fine-tuned parameters as a JSON file.

    Args:
        params (dict): A dictionary containing the model's fine-tuned parameters.
        model_name (str): The name of the model for which parameters will be saved.

    Returns:
        None
    """
    param_path = os.path.join(param_dir, f"{model_name}_params.json")
    with open(param_path, "w") as f:
        json.dump(params, f, indent=4)
    print(f"Parameters saved to {param_path}")

def load_params(model_name):
    """
    Load fine-tuned parameters from a JSON file if available.

    Args:
        model_name (str): The name of the model for which parameters are to be loaded.

    Returns:
        dict or None: The dictionary containing the model's fine-tuned parameters, 
                      or None if no parameters are found.
    """
    param_path = os.path.join(param_dir, f"{model_name}_params.json")
    try:
        with open(param_path, "r") as f:
            params = json.load(f)
        print(f"Parameters loaded from {param_path}")
        return params
    except FileNotFoundError:
        print(f"No saved parameters found for {model_name}. Proceeding with default parameters.")
        return None