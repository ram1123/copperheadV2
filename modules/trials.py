import os
import yaml
from pathlib import Path

_DEFAULT_CFG = Path(__file__).resolve().parents[1] / "configs" / "trials.yml"

def load_trials(cfg_path: str | Path = _DEFAULT_CFG) -> dict:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    base_dir = cfg.get("base_dir", "")
    trials = {}
    for name, info in cfg["trials"].items():
        stage1 = info["stage1"].format(base_dir=base_dir)
        trials[name] = {
            "stage1": stage1,
            "comment": info.get("comment", "")
        }
    return trials

def get_stage1_path(trial: str = "current",
                    cfg_path: str | Path = _DEFAULT_CFG) -> str:
    # Optional env override, e.g. TRIAL=jer_alt_v2
    trial = os.environ.get("HMM_TRIAL", trial)

    trials = load_trials(cfg_path)
    if trial not in trials:
        raise KeyError(f"Unknown trial '{trial}'. Available: {list(trials)}")
    return trials[trial]["stage1"]
