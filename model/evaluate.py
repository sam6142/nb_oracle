"""
Model evaluation functions.
Shared across all notebooks and scripts.
"""
import numpy as np


def compute_wmape(actual, predicted):
    """
    Weighted Mean Absolute Percentage Error.
    
    In plain English: "on average, how far off are our predictions?"
    Lower = better. 0.15 means "off by about 15%"
    
    Handles zeros better than regular MAPE.
    """
    total_actual = np.sum(np.abs(actual))
    if total_actual == 0:
        return 0.0
    total_error = np.sum(np.abs(actual - predicted))
    return total_error / total_actual