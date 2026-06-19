from functools import lru_cache
import correctionlib


def get_corrset(json_file: str, NanoAODv=None) -> correctionlib.CorrectionSet:
    """
    Per-process cache of correctionlib CorrectionSet objects.

    - In Dask distributed, each worker is usually a separate process => cache is local to that worker.
    - Avoids repeatedly reading/decompressing/parsing the same JSON .gz file.
    """
    if (not isinstance(json_file, str)) and (NanoAODv is not None): # sometimes json_file is actually Dict[str, str], not str
        json_file = json_file[f"nanoAODv{NanoAODv}"]
    return _load_corrset(json_file)


@lru_cache(maxsize=32)
def _load_corrset(json_file: str) -> correctionlib.CorrectionSet:
    """
    Cache only the resolved JSON file path.

    Keep dict/NanoAODv normalization outside the cached function so lru_cache
    never tries to hash an unhashable config mapping.
    """
    return correctionlib.CorrectionSet.from_file(json_file)




def get_corr_input_names(corr_obj):
    """
    Helper function to get the names of input variables from a correction object.

    Args:
        corr_obj: A correction object with an `inputs` attribute.

    Returns:
        list[str]: A list of input variable names.
    """
    names = [inp.name for inp in corr_obj.inputs]
    return names
