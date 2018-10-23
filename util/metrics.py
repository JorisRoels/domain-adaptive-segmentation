
import numpy as np

# jaccard distance
def jaccard(x, y):

    # binarize
    x = x > 0.5
    y = y > 0.5

    # compute jaccard
    intersection = np.sum(np.multiply(x, y))
    union = np.sum(x) + np.sum(y) - intersection
    return intersection / union

# dice coefficient
def dice(x, y):

    # binarize
    x = x > 0.5
    y = y > 0.5

    # stabilizing constant
    eps = 1e-10

    # compute dice
    intersection = np.sum(np.multiply(x, y))
    return 2*(intersection + eps) / (np.sum(x) + np.sum(y) + eps)
