"""Transformer settings from configs/MVA/transformer_categorizer.yaml; sample names via samples.yaml groups."""
from modules.sample_config import get_sample_dict
from modules.selection import REPO_ROOT, transformer_config


def settings() -> dict:
    return transformer_config()


def section(name: str) -> dict:
    return settings()[name]


def samples(kind: str, groups, year: str) -> dict:
    """{group: [process, ...]} of samples.yaml section `kind` ("signal"/"background") for `year`."""
    return get_sample_dict(REPO_ROOT / section("samples")["config"], kind, str(year), selected_groups=list(groups))


def signal_samples(year: str) -> dict:
    """{"ggH": [...], "VBF": [...]}: the training classes' signal processes for `year`."""
    groups = section("samples")["signal_groups"]
    resolved = samples("signal", groups.values(), year)
    return {cls: resolved[group] for cls, group in groups.items()}
