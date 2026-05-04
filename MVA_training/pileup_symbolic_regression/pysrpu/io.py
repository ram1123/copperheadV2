import os
from glob import glob

def expand_inputs(inp: str, use_glob: bool) -> list[str]:
    if os.path.isdir(inp):
        return [inp]  # dataset directory
    if use_glob or any(ch in inp for ch in "*?[]"):
        return sorted(glob(inp))
    return [inp]

def load_parquet(paths, use_pyarrow, columns=None, max_rows=None):
    import pandas as pd
    if len(paths) == 1 and os.path.isdir(paths[0]):
        df = pd.read_parquet(paths[0], columns=columns)  # directory dataset supported citeturn0search1
        return df.head(max_rows) if max_rows else df

    if use_pyarrow:
        import pyarrow.dataset as ds
        dataset = ds.dataset(paths, format="parquet")  # multi-file dataset citeturn0search2turn0search6
        tab = dataset.to_table(columns=columns)
        if max_rows:
            tab = tab.slice(0, max_rows)
        return tab.to_pandas()  # Arrow table -> pandas citeturn0search10

    df = pd.concat((pd.read_parquet(p, columns=columns) for p in paths), ignore_index=True)
    return df.head(max_rows) if max_rows else df