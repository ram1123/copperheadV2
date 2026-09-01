import numpy as np

# define cut ranges to do polynomial fits. pt ranges beyond that point we fit with a constant
poly_fit_ranges = {
    "2016preVFP": {
        "njet0": [30, 80],
        "njet1": [13, 90],
        "njet2": [15, 100],
    },
    "2016postVFP": {
        "njet0": [10, 100],
        "njet1": [12, 80],
        "njet2": [10, 90],
    },
    "2017": {
        "njet0": [9.5, 110],
        "njet1": [10, 100],
        "njet2": [13, 115],
    },
    "2018": {
        "njet0": [10, 110],
        "njet1": [11, 100],
        "njet2": [10, 120],
    },

    "2022preEE": {
        "njet0": [15, 80],
        "njet1": [21, 100],
        "njet2": [18, 110],
    },
    "2022postEE": {
        "njet0": [15, 80.5],
        "njet1": [20, 80.5],
        "njet2": [20, 110.0],
    },
    "2023": {
        "njet0": [12, 80],
        "njet1": [18, 80],
        "njet2": [10, 120],
    },
    "2023BPix": {
        "njet0": [10, 80],
        "njet1": [21, 80],
        "njet2": [10, 120],
    },
    "2024": {
        "njet0": [15, 75],
        "njet1": [21, 90],
        "njet2": [15, 110],
    },
}


def define_custom_binning(njets="1"):
    """
    Returns an array of custom bin edges:
    Build variable bin edges using (x_end, step) segments.
    """

    segments_map = {
        "0": [
            (15.0, 0.2),
            (30.0, 0.5),
            (50.0, 1.0),
            (80.0, 2.5),
            (120.0, 10.0),
            (200.0, 25.0),
        ],

        "1": [
            (10.0, 1.0),
            (30.0, 1.0),
            (50.0, 2.5),
            (80.0, 2.5),
            (120.0, 10.0),
            (200.0, 25.0),
        ],

        "2": [
            (10.0, 0.5),
            (30.0, 0.5),
            (50.0, 1.0),
            (80.0, 2.5),
            (120.0, 10.0),
            (200.0, 25.0),
        ],
    }

    nj = str(njets)

    edges = [0.0]
    x = 0.0

    for x_end, step in segments_map[nj]:
        while x + step < x_end + 1e-12:
            x += step
            edges.append(x)

    # ---- Force ONLY final boundary to be exactly 200 ----
    if edges[-1] != 200.0:
        edges[-1] = 200.0


    # NOTE: if last bin width is smaller than the previous one, merge the last two bins
    if (edges[-1] - edges[-2]) < (edges[-2] - edges[-3]):
        edges[-2] = edges[-1]
        edges.pop()
    # round the edges to avoid floating point issues
    edges = np.round(edges, 2)
    edges = np.unique(edges).tolist()
    return edges
