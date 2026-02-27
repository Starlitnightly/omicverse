import numpy as np


def rescale_linear(array, new_min, new_max):
    """Rescale an arrary linearly

    args:
        array: a number array
        new_min: the minimum value for new scale
        new_max: the maximum value for new scale
    """
    minimum, maximum = np.min(array), np.max(array)
    if maximum - minimum == 0:
        return array
    m = (new_max - new_min) / (maximum - minimum)
    b = new_min - m * minimum

    return m * array + b
