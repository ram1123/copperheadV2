from pathlib import Path
import subprocess
import hashlib
from typing import Dict

def get_git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def get_git_state(output_dir: str | Path) -> Dict:
    """
    Collect git commit + dirty state.
    If dirty, save git diff patch into output_dir/git_diff.patch
    and store its SHA1 hash in the returned dict.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "commit": "unknown",
        "dirty": False,
        "diff_hash": None,
        "diff_file": None,
    }

    # --- commit hash ---
    try:
        state["commit"] = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return state  # not a git repo

    # --- dirty check ---
    try:
        subprocess.check_call(
            ["git", "diff", "--quiet"],
            stderr=subprocess.DEVNULL,
        )
        return state  # clean repo
    except subprocess.CalledProcessError:
        state["dirty"] = True

    # --- save diff ---
    try:
        diff = subprocess.check_output(
            ["git", "diff"],
            stderr=subprocess.DEVNULL,
        )

        if diff.strip():
            diff_hash = hashlib.sha1(diff).hexdigest()
            diff_path = output_dir / "git_diff.patch"

            diff_path.write_bytes(diff)

            state["diff_hash"] = diff_hash
            state["diff_file"] = diff_path.name

    except Exception:
        pass

    return state
