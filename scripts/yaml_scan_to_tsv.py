#!/usr/bin/env python3
import os, sys, re, io, json, yaml, glob, hashlib, subprocess, shlex
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---- helpers ---------------------------------------------------------------

def is_xrootd(path: str) -> bool:
    return path.startswith("root://")

def split_xrootd(url: str):
    # root://host//path  or root://host/path
    m = re.match(r"root://([^/]+)(/.*)", url)
    if not m:
        raise ValueError(f"Bad XRootD url: {url}")
    host, p = m.group(1), m.group(2)
    # canonical double-slash path for xrdfs
    if not p.startswith("//"):
        p = "/" + p
    return host, p

def xrdfs_ls_r(url_dir: str):
    """Recursive ls via xrdfs; returns list of root://.../*.root files."""
    host, rpath = split_xrootd(url_dir)
    cmd = f"xrdfs {host} ls -R {shlex.quote(rpath)}"
    try:
        out = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f"[xrdfs ls] failed for {url_dir}: {e.output}\n")
        return []
    files = []
    for line in out.splitlines():
        line = line.strip()
        if not line or line.endswith(":") or line.endswith("/"):
            continue
        if line.lower().endswith(".root"):
            # join back to full root:// url
            files.append(f"root://{host}{line}")
    return files

def xrdfs_stat_size(url: str):
    host, rpath = split_xrootd(url)
    cmd = f"xrdfs {host} stat -q Size {shlex.quote(rpath)}"
    try:
        out = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT).strip()
        # output like: "Size 12345"
        parts = out.split()
        if len(parts) == 2 and parts[0].lower() == "size":
            return int(parts[1])
    except Exception as e:
        sys.stderr.write(f"[xrdfs stat] size failed for {url}: {e}\n")
    return None

def xrdfs_checksum(url: str):
    """Return ('adler32', value) if available, else (None, None)."""
    host, rpath = split_xrootd(url)
    cmd = f"xrdfs {host} query checksum {shlex.quote(rpath)}"
    try:
        out = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT).strip()
        # typical: "adler32 3a1b2c3d"
        algo, val = out.split()
        return (algo.lower(), val.lower())
    except Exception as e:
        sys.stderr.write(f"[xrdfs checksum] failed for {url}: {e}\n")
        return (None, None)

def sha256_local(path: str, blocksize=4*1024*1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(blocksize)
            if not b: break
            h.update(b)
    return h.hexdigest()

def list_root_files(base: str):
    if is_xrootd(base):
        return xrdfs_ls_r(base)
    # local: recurse
    if os.path.isfile(base) and base.lower().endswith(".root"):
        return [base]
    pattern = os.path.join(base, "**", "*.root")
    return glob.glob(pattern, recursive=True)

# ---- uproot (no internet needed) ------------------------------------------
import uproot

def read_event_count(fileurl: str):
    try:
        with uproot.open(fileurl) as f:
            if "Events" in f:
                return f["Events"].num_entries
    except Exception as e:
        sys.stderr.write(f"[uproot] Events entries failed for {fileurl}: {e}\n")
    return None

def read_runs_sums(fileurl: str):
    """Return (genEventCount, genEventSumw, genEventSumw2) summed over Runs tree (if any)."""
    vals = {"genEventCount": None, "genEventSumw": None, "genEventSumw2": None}
    try:
        with uproot.open(fileurl) as f:
            if "Runs" not in f:
                return (None, None, None)
            t = f["Runs"]
            # read only existing of the three
            fields = [b.name for b in t.branches]
            want = [x for x in ["genEventCount", "genEventSumw", "genEventSumw2"] if x in fields]
            if not want:
                return (None, None, None)
            arrs = t.arrays(want, library="np")
            out = []
            for k in ["genEventCount", "genEventSumw", "genEventSumw2"]:
                if k in arrs:
                    v = float(arrs[k].sum()) if arrs[k].size else 0.0
                    out.append(v)
                else:
                    out.append(None)
            return tuple(out)
    except Exception as e:
        sys.stderr.write(f"[uproot] Runs sums failed for {fileurl}: {e}\n")
        return (None, None, None)

def branches_hash(fileurl: str):
    """Stable SHA256 over Events branch names + types."""
    try:
        with uproot.open(fileurl) as f:
            if "Events" not in f:
                return None
            t = f["Events"]
            items = []
            for b in t.branches:
                # name and type/interpretation
                ityp = getattr(b, "interpretation", None)
                ityp_s = str(ityp) if ityp is not None else str(getattr(b, "dtype", ""))
                items.append(f"{b.name}:{ityp_s}")
            items.sort()
            s = "\n".join(items).encode("utf-8")
            return hashlib.sha256(s).hexdigest()
    except Exception as e:
        sys.stderr.write(f"[uproot] branches hash failed for {fileurl}: {e}\n")
        return None

# ---- YAML parsing ----------------------------------------------------------

def iter_datasets_from_yaml(yaml_path):
    """Yield tuples (year, group, sample, dataset_path_or_dir)"""
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    years = cfg.get("years", {})
    for year, groups in (years or {}).items():
        if not isinstance(groups, dict):
            continue
        for group, samples in groups.items():
            if isinstance(samples, dict):
                for sample, node in samples.items():
                    if isinstance(node, str):
                        yield (year, group, sample, node)
                    elif isinstance(node, list):
                        for ds in node:
                            if isinstance(ds, str):
                                yield (year, group, sample, ds)
            else:
                # fallback: treat as single path or list
                if isinstance(samples, str):
                    yield (year, group, "_", samples)
                elif isinstance(samples, list):
                    for ds in samples:
                        if isinstance(ds, str):
                            yield (year, group, "_", ds)

# ---- main scanning ---------------------------------------------------------

def scan_one_file(ctx, fileurl: str):
    year, group, sample, dataset = ctx
    # size
    if is_xrootd(fileurl):
        size = xrdfs_stat_size(fileurl)
        algo, csum = xrdfs_checksum(fileurl)
        checksum = f"{algo}:{csum}" if algo and csum else ""
    else:
        try:
            size = os.path.getsize(fileurl)
        except Exception:
            size = None
        checksum = f"sha256:{sha256_local(fileurl)}" if os.path.exists(fileurl) else ""
    # quick metadata
    nEv = read_event_count(fileurl)
    gCount, gSumw, gSumw2 = read_runs_sums(fileurl)
    bhash = branches_hash(fileurl)
    print(f"checksum: {checksum}")
    # TSV row
    return [
        year or "",
        group or "",
        sample or "",
        dataset or "",
        fileurl,
        str(size) if size is not None else "",
        checksum,
        str(nEv) if nEv is not None else "",
        "" if gCount is None else f"{int(gCount) if abs(gCount - int(gCount))<1e-6 else gCount}",
        "" if gSumw  is None else f"{gSumw:.6g}",
        "" if gSumw2 is None else f"{gSumw2:.6g}",
        bhash or "",
    ]

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Scan YAML datasets -> TSV with NanoAOD file metadata")
    ap.add_argument("yaml", help="YAML config with years/…/dataset paths (local dirs or root://)")
    ap.add_argument("-o", "--out", default="index.tsv", help="Output TSV path")
    ap.add_argument("--max-workers", type=int, default=8, help="Parallel workers")
    ap.add_argument("--prefix", default="", help="Optional prefix to prepend to each dataset path")
    ap.add_argument("--suffix", default="", help="Optional suffix (e.g. '/NANO*' or '/**/*.root' if your YAML holds higher-level dirs)")
    args = ap.parse_args()

    rows = []
    file_tasks = []
    max_workers = args.max_workers
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for year, group, sample, dataset in iter_datasets_from_yaml(args.yaml):
            base = f"{args.prefix}{dataset}{args.suffix}"
            print(f"[scan] {year}/{group}/{sample}: {base}")
            # Expand to file list
            files = []
            if base.lower().endswith(".root"):
                files = [base]
            else:
                files = list_root_files(base)
            # schedule file scans
            ctx = (year, group, sample, dataset)
            print(f"files: {files}")
            for fu in sorted(set(files)):
                file_tasks.append(ex.submit(scan_one_file, ctx, fu))

        for fut in as_completed(file_tasks):
            try:
                rows.append(fut.result())
            except Exception as e:
                sys.stderr.write(f"[scan] error: {e}\n")

    # write TSV
    header = ["year","group","sample","dataset","file","size_bytes","checksum","nEvents","genEventCount","genEventSumw","genEventSumw2","branches_hash"]
    with open(args.out, "w", newline="") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    print(f"Wrote {len(rows)} rows to {args.out}")

if __name__ == "__main__":
    main()
