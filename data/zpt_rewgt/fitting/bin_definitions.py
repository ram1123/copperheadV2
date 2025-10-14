import numpy as np

# define cut ranges to do polynomial fits. pt ranges beyond that point we fit with a constant
poly_fit_ranges = {
    "2018" : {
        "njet0" : [10, 110],
        "njet1" : [12, 80],
        "njet2" : [17, 100],
    },
    "2017" : {
        "njet0" : [9.5, 110],
        "njet1" : [10, 100],
        "njet2" : [13, 115],
    },
    "2016postVFP" : {
        "njet0" : [10, 100],
        "njet1" : [12, 80],
        "njet2" : [10, 90],
    },
    "2016preVFP" : {
        "njet0" : [30, 80],
        "njet1" : [13, 90],
        "njet2" : [15, 100],
    },
}

def define_custom_binning_default():
    """
    Returns an array of custom bin edges:
    0-50 in steps of 0.25, 50-80 in steps of 1, 80-100 in 2.5, 100-200 in 10.
    """
    edges = []
    x = 0.0
    while x < 20.0:
        edges.append(x)
        x += 1.0
    while x < 40.0:
        edges.append(x)
        x += 2.5
    while x < 80.0:
        edges.append(x)
        x += 2.5
    while x <= 100.0:
        edges.append(x)
        x += 2.5
    while x <= 150.0:
        edges.append(x)
        x += 5.0
    while x <= 200.0:
        edges.append(x)
        x += 15.0
    # Ensure the last edge is exactly 200
    if edges[-1] < 200.0:
        edges.append(200.0)
    # round the edges to avoid floating point issues
    edges = np.round(edges, 2)
    return np.unique(edges).tolist()


def define_custom_binning():
    """
    Returns an array of custom bin edges:
    0-50 in steps of 0.25, 50-80 in steps of 1, 80-100 in 2.5, 100-200 in 10.
    """
    edges = []
    x = 0.0
    while x <= 30.0:
        edges.append(x)
        x += 1.0
    while x < 50.0:
        edges.append(x)
        x += 2.5
    while x < 60.0:
        edges.append(x)
        x += 5.0
    while x < 80.0:
        edges.append(x)
        x += 5.0
    while x < 100.0:
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
