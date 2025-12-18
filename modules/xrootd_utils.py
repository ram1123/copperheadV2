import re


AAA_REDIRECTORS = [
    "root://xcache.cms.rcac.purdue.edu/",
    "root://eos.cms.rcac.purdue.edu/",
    "root://cms-xrd-global.cern.ch/",
    "root://xrootd-cms.infn.it/",
    "root://cmsxrootd.fnal.gov/",
    "root://xcache.cms.rcac.purdue.edu/",
    "root://xcache.cms.rcac.purdue.edu/",
]

# Accept prefixes like:
#   "root://xcache.cms.rcac.purdue.edu:1094//"
#   "root://cms-xrd-global.cern.ch//"
#   "root://xrootd-cms.infn.it//"
_ROOT_URL_RE = re.compile(r"^root://([^/]+)/+(.+)$")   # host[:port], tail after the first // (usually 'store/...')


def _sanitize_prefix(prefix: str) -> str:
    """Ensure prefix looks like 'root://host[:port]//'."""
    if not prefix.startswith("root://"):
        raise ValueError(f"Bad AAA redirector prefix: {prefix}")
    if not prefix.endswith("//"):
        prefix = prefix if prefix.endswith("/") else prefix + "/"
        prefix += "/"
    return prefix


def _replace_host(url: str, host_prefix: str) -> str:
    """Replace any ROOT URL host (or /store path) with host_prefix. Preserve the tail path."""
    host_prefix = _sanitize_prefix(host_prefix)

    # /store/... plain path -> just add prefix
    if url.startswith("/store/"):
        return f"{host_prefix}{url.lstrip('/')}"

    # root://HOST[:PORT]//store/...
    m = _ROOT_URL_RE.match(url)
    if m:
        _hostport, tail = m.groups()
        # Always force through the chosen redirector (avoids IPs & bad SAN)
        return f"{host_prefix}{tail.lstrip('/')}"
    # Not a ROOT/STORE path; leave unchanged
    return url


def normalize_paths(files, host_prefix: str):
    """Normalize input 'files' (str | list | tuple | dict) to use host_prefix."""
    host_prefix = _sanitize_prefix(host_prefix)

    if isinstance(files, dict):
        # Preserve per-file metadata dicts
        return {_replace_host(u, host_prefix): meta for u, meta in files.items()}
    if isinstance(files, (list, tuple)):
        return [_replace_host(u, host_prefix) for u in files]
    if isinstance(files, str):
        return _replace_host(files, host_prefix)
    # Unknown type: return as-is
    return files


AAA_ERROR_FRAGMENTS = (
    "TLS error",
    "hostname not in SAN",
    "File did not open properly",
    "File did not vector_read properly",
    "lzma data error",
    "File stat request failed",
    "OSError: Bytes failed to read from open file",
    "OSError: Failed to close file: [ERROR] Operation expired",
    "Operation expired",
)
