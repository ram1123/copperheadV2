import numpy as np

# define cut ranges to do polynomial fits. pt ranges beyond that point we fit with a constant
poly_fit_ranges = {
    "2018" : {
        "njet0" : [10, 120],
        "njet1" : [14, 100],
        "njet2" : [8, 120],
    },
    "2017" : {
        "njet0" : [10, 80],
        "njet1" : [11, 80],
        "njet2" : [10, 90],
    },
    "2016postVFP" : {
        "njet0" : [20, 100],
        "njet1" : [12, 100],
        "njet2" : [10, 100],
    },
    "2016preVFP" : {
        "njet0" : [10, 100],
        "njet1" : [10, 100],
        "njet2" : [10, 100],
    },
}

def define_custom_binning():
    """
    Returns an array of custom bin edges:
    0-50 in steps of 0.25, 50-80 in steps of 1, 80-100 in 2.5, 100-200 in 10.
    """
    edges = []
    x = 0.0
    while x < 20.0:
        edges.append(x)
        x += 0.5
    while x < 40.0:
        edges.append(x)
        x += 1.0
    while x < 80.0:
        edges.append(x)
        x += 1.5
    while x <= 100.0:
        edges.append(x)
        x += 2.5
    while x <= 150.0:
        edges.append(x)
        x += 5.0
    while x <= 200.0:
        edges.append(x)
        x += 10.0
    # Ensure the last edge is exactly 200
    if edges[-1] < 200.0:
        edges.append(200.0)
    # round the edges to avoid floating point issues
    edges = np.round(edges, 2)
    return np.unique(edges).tolist()


def define_custom_binning_2018_ZEROjet():
    """
    Returns an array of custom bin edges:
    0-50 in steps of 0.25, 50-80 in steps of 1, 80-100 in 2.5, 100-200 in 10.
    """
    edges = []
    x = 0.0
    while x < 30.0:
        edges.append(x)
        x += 0.1
    while x < 60.0:
        edges.append(x)
        x += 0.8
    while x < 80.0:
        edges.append(x)
        x += 2.5
    while x <= 100.0:
        edges.append(x)
        x += 10.0
    while x <= 200.0:
        edges.append(x)
        x += 25.0
    # Ensure the last edge is exactly 200
    if edges[-1] < 200.0:
        edges.append(200.0)
    # round the edges to avoid floating point issues
    edges = np.round(edges, 2)
    return np.unique(edges).tolist()

def define_custom_binning_2018_ONEjet():
    """
    Returns an array of custom bin edges:
    0-50 in steps of 0.25, 50-80 in steps of 1, 80-100 in 2.5, 100-200 in 10.
    """
    edges = []
    x = 0.0
    while x < 20.0:
        edges.append(x)
        x += 0.5
    while x < 40.0:
        edges.append(x)
        x += 1.0
    while x < 80.0:
        edges.append(x)
        x += 1.5
    while x <= 100.0:
        edges.append(x)
        x += 2.5
    while x <= 150.0:
        edges.append(x)
        x += 5.0
    # while x <= 200.0:
    #     edges.append(x)
    #     x += 50.0
    # Ensure the last edge is exactly 200
    if edges[-1] < 200.0:
        edges.append(200.0)
    # round the edges to avoid floating point issues
    edges = np.round(edges, 2)
    return np.unique(edges).tolist()
