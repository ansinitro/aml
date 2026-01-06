import os
import random
import numpy as np
import tensorflow as tf
import json

def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across all libraries.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"Random seed set to {seed}")

def save_results(results, filename='results.json'):
    """Save model performance metrics to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {filename}")

def load_results(filename='results.json'):
    """Load model performance metrics from a JSON file."""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None
