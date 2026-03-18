import re

AAA_REDIRECTORS = [
    "root://eos.cms.rcac.purdue.edu/",                  # 2
    "root://eos.cms.rcac.purdue.edu/",                  # 2
    # "root://xcache.cms.rcac.purdue.edu/",                  # 1
]

# Accept prefixes like:
#   "root://xcache.cms.rcac.purdue.edu:1094//"
#   "root://cms-xrd-global.cern.ch//"
#   "root://xrootd-cms.infn.it//"
_ROOT_URL_RE = re.compile(
    r"^root://([^/]+)/+(.+)$"
)  # host[:port], tail after the first // (usually 'store/...')

def _is_local_prefix(prefix: str) -> bool:
    return prefix.startswith("/eos/purdue")


def _sanitize_prefix(prefix: str) -> str:
    """
    Accept either:
      - ROOT redirector: 'root://host[:port]//'
      - Local EOS mount: '/eos/purdue/'
    """
    if prefix.startswith("root://"):
        # Ensure exactly ends with '//' (at least)
        if not prefix.endswith("//"):
            prefix = prefix.rstrip("/") + "//"
        return prefix

    if _is_local_prefix(prefix):
        # Ensure ends with single '/'
        return prefix.rstrip("/") + "/"

    raise ValueError(f"Bad AAA redirector prefix: {prefix}")


def _join_prefix(prefix: str, tail: str) -> str:
    """
    Join sanitized prefix + tail (tail should be relative like 'store/...')
    """
    prefix = _sanitize_prefix(prefix)
    tail = tail.lstrip("/")
    return f"{prefix}{tail}"


def _replace_host(url: str, host_prefix: str) -> str:
    """
    Replace any ROOT URL host (or /store path) with host_prefix.
    Preserves the tail path (typically 'store/...').

    Examples:
      host_prefix='/eos/purdue' + 'root://cms-xrd-global.cern.ch//store/x' -> '/eos/purdue/store/x'
      host_prefix='root://eos.cms...//' + '/store/x' -> 'root://eos.cms...//store/x'
    """
    host_prefix = _sanitize_prefix(host_prefix)

    # If it's already the chosen local prefix, keep it
    if _is_local_prefix(host_prefix) and url.startswith(host_prefix):
        return url

    # /store/... plain path -> just add prefix
    if url.startswith("/store/"):
        return _join_prefix(host_prefix, url.lstrip("/"))

    # root://HOST[:PORT]//store/...
    m = _ROOT_URL_RE.match(url)
    if m:
        _hostport, tail = m.groups()
        return _join_prefix(host_prefix, tail)

    # If you ever see '/eos/purdue/store/...' and want to retarget it,
    # you can optionally map it back to '/store/...'. Leaving unchanged is safer.
    if url.startswith("/eos/purdue/store/"):
        if host_prefix.startswith("root://"):
            # Retarget local EOS -> remote ROOT redirector
            tail = url[len("/eos/purdue/") :]  # becomes 'store/...'
            return _join_prefix(host_prefix, tail)
        return url

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
    "Unable to open file",
    "No servers have access to the file",
    "[ERROR] Socket timeout",
    "[ERROR] Server responded with an error",
)
