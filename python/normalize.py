import numpy as np

def normalize(x):
    x = x.astype(np.float64)
    x -= x.mean(axis=0)
    x /= x.std(axis=0)
    return x

