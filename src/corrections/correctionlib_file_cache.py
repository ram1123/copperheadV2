from functools import lru_cache
import correctionlib


@lru_cache(maxsize=32)
def get_corrset(json_file: str) -> correctionlib.CorrectionSet:
    """
    Per-process cache of correctionlib CorrectionSet objects.

    - In Dask distributed, each worker is usually a separate process => cache is local to that worker.
    - Avoids repeatedly reading/decompressing/parsing the same JSON .gz file.
    """
    return correctionlib.CorrectionSet.from_file(json_file)
