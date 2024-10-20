import time
import numpy as np
import pickle
import awkward as ak
import dask_awkward as dak
from distributed import Client



def scoreEdgesBySigEff(events, load_path: str, save_path: str ):