"""Tests for scripts/strip_datacard_variations.py.

The stripper edits the file the fit reads, so the properties that matter are:
it removes exactly the shape rows it should, it touches nothing else, and it can
be run twice (or with a different --keep) without compounding its own effect.
"""
import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "strip_datacard_variations", REPO_ROOT / "scripts" / "strip_datacard_variations.py"
)
sdv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sdv)


# A faithful reduction of a real Stage-3 card: same row order, same column
# padding, same trailing separators.
CARD = """imax 1
jmax *
kmax *
---------------
shapes * vbf_SR_2017 ../templates/vbf_h-peak_2017.root $PROCESS $PROCESS_$SYSTEMATIC
---------------
bin                           vbf_SR_2017
observation                          2282
---------------
bin                           vbf_SR_2017         vbf_SR_2017
process                       DYVBF               ggH_hmm
process                       1                   -1
rate                          1272.992859         5.09103
---------------
Absolute_2017        shape    1.0                 1.0
Absolute             shape    1.0                 1.0
BBEC1_2017           shape    1.0                 1.0
FlavorQCD            shape    1.0                 1.0
RelativeSample_2017  shape    1.0                 1.0
Total                shape    1.0                 1.0
mu_roccor2017        shape    1.0                 1.0
lumi2017             lnN      1.023               1.023
---------------
vbf_SR_2017 autoMCStats 0 1 1
---------------
---------------
"""

ALL_SHAPES = [
    "Absolute_2017", "Absolute", "BBEC1_2017", "FlavorQCD",
    "RelativeSample_2017", "Total", "mu_roccor2017",
]


@pytest.fixture
def carddir(tmp_path):
    for region in ("SR", "SB"):
        (tmp_path / f"datacard_vbf_{region}_2017.txt").write_text(
            CARD.replace("vbf_SR_2017", f"vbf_{region}_2017")
        )
    return tmp_path


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Total", "Total"),
        ("mu_roccor2017", "mu_roccor"),
        ("Absolute_2017", "Absolute"),
        ("RelativeSample_2016APV", "RelativeSample"),
        ("FlavorQCD", "FlavorQCD"),
    ],
)
def test_base_name_strips_year_tokens(name, expected):
    assert sdv.base_name(name) == expected


def test_keeps_only_requested_sources():
    _, kept, dropped = sdv.strip_text(CARD, sdv.DEFAULT_KEEP)
    assert kept == ["Total", "mu_roccor2017"]
    assert dropped == ["Absolute_2017", "Absolute", "BBEC1_2017",
                       "FlavorQCD", "RelativeSample_2017"]


def test_non_shape_lines_are_untouched():
    """lnN, autoMCStats, rates, headers and separators must survive verbatim."""
    new_text, _, _ = sdv.strip_text(CARD, sdv.DEFAULT_KEEP)
    removed = [ln for ln in CARD.splitlines() if ln not in new_text.splitlines()]
    assert all(sdv.is_shape_line(ln)[0] for ln in removed)
    for keeper in ("lumi2017", "autoMCStats", "observation", "rate", "shapes *", "kmax"):
        assert keeper in new_text


def test_kept_rows_are_byte_identical():
    new_text, _, _ = sdv.strip_text(CARD, sdv.DEFAULT_KEEP)
    original = {ln for ln in CARD.splitlines()}
    assert all(ln in original for ln in new_text.splitlines())


def test_empty_keep_set_removes_every_shape_row():
    _, kept, dropped = sdv.strip_text(CARD, [])
    assert kept == []
    assert dropped == ALL_SHAPES


def test_keep_all_is_a_noop():
    new_text, _, dropped = sdv.strip_text(
        CARD, [sdv.base_name(n) for n in ALL_SHAPES]
    )
    assert dropped == []
    assert new_text == CARD


def test_cli_writes_backup_and_strips(carddir):
    assert sdv.main([str(carddir), "--years", "2017"]) == 0
    for region in ("SR", "SB"):
        card = carddir / f"datacard_vbf_{region}_2017.txt"
        backup = carddir / f"datacard_vbf_{region}_2017.txt.full"
        assert backup.read_text() == CARD.replace("vbf_SR_2017", f"vbf_{region}_2017")
        assert "FlavorQCD" not in card.read_text()
        assert "Total  " in card.read_text()


def test_running_twice_is_idempotent(carddir):
    sdv.main([str(carddir), "--years", "2017"])
    once = (carddir / "datacard_vbf_SR_2017.txt").read_text()
    sdv.main([str(carddir), "--years", "2017"])
    assert (carddir / "datacard_vbf_SR_2017.txt").read_text() == once


def test_rerun_can_widen_the_keep_set(carddir):
    """Re-derivation from the backup means a later, larger --keep is honoured."""
    sdv.main([str(carddir), "--years", "2017"])
    assert "FlavorQCD" not in (carddir / "datacard_vbf_SR_2017.txt").read_text()
    sdv.main([str(carddir), "--keep", "Total,mu_roccor,FlavorQCD", "--years", "2017"])
    assert "FlavorQCD" in (carddir / "datacard_vbf_SR_2017.txt").read_text()


def test_restore_returns_the_original(carddir):
    original = (carddir / "datacard_vbf_SR_2017.txt").read_text()
    sdv.main([str(carddir), "--years", "2017"])
    sdv.main([str(carddir), "--restore", "--years", "2017"])
    assert (carddir / "datacard_vbf_SR_2017.txt").read_text() == original


def test_dry_run_writes_nothing(carddir):
    original = (carddir / "datacard_vbf_SR_2017.txt").read_text()
    assert sdv.main([str(carddir), "--dry-run"]) == 0
    assert (carddir / "datacard_vbf_SR_2017.txt").read_text() == original
    assert not (carddir / "datacard_vbf_SR_2017.txt.full").exists()


def test_combined_cards_are_not_touched(carddir):
    """combineCards.py output is built from these files; stripping it would be
    both too late and double work."""
    (carddir / "HMuMu_13TeV_2017.txt").write_text(CARD)
    sdv.main([str(carddir)])
    assert (carddir / "HMuMu_13TeV_2017.txt").read_text() == CARD


def test_missing_directory_is_an_error(tmp_path):
    assert sdv.main([str(tmp_path / "nope")]) == 2


def test_no_cards_found_is_an_error(tmp_path):
    assert sdv.main([str(tmp_path)]) == 2
