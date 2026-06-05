import os
from glob import glob

def extract_sample_name(path: str) -> str:
    norm = os.path.normpath(path)
    if os.path.isdir(norm):
        return os.path.basename(norm)

    parent = os.path.dirname(norm)
    base = os.path.basename(parent)
    if base.isdigit():
        return os.path.basename(os.path.dirname(parent))
    return base


def classify_process_group(sample_name: str) -> str:
    name = sample_name.lower()
    if name.startswith("dy") or "dyto2l" in name or "dyto2mu" in name:
        return "DY"
    if name.startswith("tt") or name.startswith("st") or "ttjets" in name or "single_top" in name:
        return "Top"
    if name.startswith("ewk"):
        return "EWK"
    if name.startswith("ww") or name.startswith("wz") or name.startswith("zz"):
        return "VV"
    if name.startswith("vvv"):
        return "VVV"
    return sample_name


def expand_inputs(inp, use_glob: bool) -> list[str]:
    if isinstance(inp, str):
        raw_inputs = [tok.strip() for tok in inp.split(",") if tok.strip()]
    else:
        raw_inputs = []
        for item in inp:
            raw_inputs.extend(tok.strip() for tok in str(item).split(",") if tok.strip())

    paths: list[str] = []
    seen: set[str] = set()
    for item in raw_inputs:
        if os.path.isdir(item):
            expanded = [item]
        elif use_glob or any(ch in item for ch in "*?[]"):
            expanded = sorted(glob(item, recursive=True))
        else:
            expanded = [item]

        for path in expanded:
            if path not in seen:
                seen.add(path)
                paths.append(path)
    return paths

def load_parquet(paths, use_pyarrow, columns=None, max_rows=None, attach_source_meta=False):
    import pandas as pd
    if len(paths) == 1 and os.path.isdir(paths[0]):
        df = pd.read_parquet(paths[0], columns=columns)
        if attach_source_meta:
            sample_name = extract_sample_name(paths[0])
            df["__sample_name"] = sample_name
            df["__process_group"] = classify_process_group(sample_name)
        return df.head(max_rows) if max_rows else df

    if use_pyarrow:
        import pyarrow.dataset as ds
        if attach_source_meta:
            frames = []
            seen_rows = 0
            for path in paths:
                dataset = ds.dataset(path, format="parquet")
                tab = dataset.to_table(columns=columns)
                if max_rows is not None:
                    remaining = max_rows - seen_rows
                    if remaining <= 0:
                        break
                    tab = tab.slice(0, remaining)
                frame = tab.to_pandas()
                sample_name = extract_sample_name(path)
                frame["__sample_name"] = sample_name
                frame["__process_group"] = classify_process_group(sample_name)
                frames.append(frame)
                seen_rows += len(frame)
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        dataset = ds.dataset(paths, format="parquet")
        tab = dataset.to_table(columns=columns)
        if max_rows:
            tab = tab.slice(0, max_rows)
        return tab.to_pandas()

    frames = []
    seen_rows = 0
    for path in paths:
        frame = pd.read_parquet(path, columns=columns)
        if attach_source_meta:
            sample_name = extract_sample_name(path)
            frame["__sample_name"] = sample_name
            frame["__process_group"] = classify_process_group(sample_name)
        if max_rows is not None:
            remaining = max_rows - seen_rows
            if remaining <= 0:
                break
            frame = frame.head(remaining)
        frames.append(frame)
        seen_rows += len(frame)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return df.head(max_rows) if max_rows else df
