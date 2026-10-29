import pickle
from dataclasses import replace
from types import SimpleNamespace

import awkward as ak
import numpy as np
import pandas as pd
import pytest

from MVA_training.VBF_run3.preprocess_dnn import (
    apply_transformer_training_selection,
    load_config,
    resolve_glob,
)
from MVA_training.VBF_run3 import ray_train_driver
from MVA_training.VBF_run3.utils.pre_scale_cleaning import (
    pre_scaling_clean_transformer,
)
from MVA_training.VBF_run3.submit_ray_dnn import ray_headers


def test_existing_run3_auto_config_stays_cut_based_by_default():
    assert load_config("configs/dnn_run3_vbf.yaml").category_selection == "cut"


def test_transformer_selection_has_only_mass_bveto_and_low_score():
    cfg = replace(
        load_config("configs/dnn_run2_vbf.yaml"),
        category_selection="transformer",
        transformer_score_threshold=0.5,
    )
    events = ak.Array(
        {
            "event": [1, 2, 3, 4, 5, 6, 7],
            "dimuon_mass": [115.0, 134.999, 135.0, 120.0, 120.0, 120.0, 120.0],
            "njets_nominal": [0, 1, 2, 1, 2, 2, 2],
            "nBtagLoose_nominal": [0, 0, 0, 2, 0, 0, 0],
            "nBtagMedium_nominal": [0, 0, 0, 0, 1, 0, 0],
            "transf_ggh_vs_vbf_binary_score": [
                0.1,
                0.49,
                0.1,
                0.1,
                0.1,
                0.5,
                np.nan,
            ],
        }
    )

    selected = apply_transformer_training_selection(events, cfg)

    assert ak.to_list(selected.event) == [1, 2]


def test_transformer_sentinels_cover_zero_one_jet_and_nonfinite_values():
    frame = pd.DataFrame(
        {
            "njets_nominal": [0, 1, 2],
            "dimuon_rapidity": [1.0, 2.0, np.nan],
            "dimuon_pt": [np.inf, 20.0, 30.0],
            "jet1_pt_nominal": [np.nan, 30.0, 40.0],
            "jet1_eta_nominal": [np.inf, 1.0, 2.0],
            "jet1_btagUParTAK4QvG_nominal": [np.nan, np.nan, 0.4],
            "jet2_phi_nominal": [np.nan, np.nan, 2.0],
            "jj_mass_log_nominal": [np.nan, np.nan, 5.0],
            "rpt_nominal": [np.nan, np.nan, np.inf],
            "ll_zstar_log_nominal": [np.nan, np.nan, np.inf],
            "mmj_min_dEta_nominal": [np.nan, np.inf, 1.0],
            "pt_centrality_nominal": [np.nan, np.nan, 500.0],
        }
    )
    features = [column for column in frame if column != "njets_nominal"]

    cleaned = pre_scaling_clean_transformer(frame, features)

    assert cleaned["jet1_pt_nominal"].tolist() == [0.0, 30.0, 40.0]
    assert cleaned["jet1_eta_nominal"].tolist() == [-5.0, 1.0, 2.0]
    assert cleaned["jet1_btagUParTAK4QvG_nominal"].tolist() == [-1.0, -1.0, 0.4]
    assert cleaned["jet2_phi_nominal"].tolist() == [-5.0, -5.0, 2.0]
    assert cleaned["jj_mass_log_nominal"].tolist() == [0.0, 0.0, 5.0]
    assert cleaned["rpt_nominal"].tolist() == [-1.0, -1.0, -1.0]
    assert cleaned["ll_zstar_log_nominal"].tolist() == [-20.0, -20.0, -20.0]
    assert cleaned["mmj_min_dEta_nominal"].tolist() == [-1.0, -1.0, 1.0]
    assert cleaned["pt_centrality_nominal"].tolist() == [-200.0, -200.0, 200.0]
    assert np.isfinite(cleaned[features].to_numpy()).all()


def test_transformer_raw_pt_centrality_is_clipped_but_other_values_are_not():
    frame = pd.DataFrame(
        {
            "njets_nominal": [2, 2, 2],
            "ll_zstar_log_nominal": [-100.0, 0.0, 100.0],
            "pt_centrality_nominal": [-500.0, 10.0, 500.0],
        }
    )

    cleaned = pre_scaling_clean_transformer(
        frame, ["ll_zstar_log_nominal", "pt_centrality_nominal"]
    )

    assert cleaned["ll_zstar_log_nominal"].tolist() == [-100.0, 0.0, 100.0]
    assert cleaned["pt_centrality_nominal"].tolist() == [-200.0, 10.0, 200.0]


def test_run3_flat_scored_layout_is_a_glob_fallback(tmp_path):
    parquet = tmp_path / "2024" / "vbf_powheg" / "0" / "part0.parquet"
    parquet.parent.mkdir(parents=True)
    parquet.touch()

    files = resolve_glob(
        str(tmp_path),
        "2024",
        "{BASE}/{YEAR}/compacted/{PROCESS}/*/*.parquet",
        "vbf_powheg",
    )

    assert files == [str(parquet)]


def test_ray_contract_requests_exactly_four_one_gpu_workers():
    headers = ray_headers(
        SimpleNamespace(
            cpus_per_worker=8,
            ram_per_worker_gb=32,
            min_gpu_memory_gb=16,
            ray_env="/work/pixi/global/.pixi/envs/default",
        )
    )

    assert headers["af-n-workers"] == "4"
    assert headers["af-gpus-per-worker"] == "1"


def test_ray_contract_supports_sixteen_one_gpu_workers_for_four_models():
    headers = ray_headers(
        SimpleNamespace(
            cluster_workers=16,
            cpus_per_worker=4,
            ram_per_worker_gb=16,
            min_gpu_memory_gb=5,
            ray_env="/work/pixi/global/.pixi/envs/default",
        )
    )

    assert headers["af-n-workers"] == "16"
    assert headers["af-gpus-per-worker"] == "1"
    assert headers["af-min-gpu-memory-gb"] == "5"


def test_ray_contract_supports_eight_one_gpu_workers_for_two_models():
    headers = ray_headers(
        SimpleNamespace(
            cluster_workers=8,
            cpus_per_worker=4,
            ram_per_worker_gb=16,
            min_gpu_memory_gb=5,
            ray_env="/work/pixi/global/.pixi/envs/default",
        )
    )

    assert headers["af-n-workers"] == "8"
    assert headers["af-gpus-per-worker"] == "1"


def test_ray_contract_supports_elastic_max_sixteen_workers():
    headers = ray_headers(
        SimpleNamespace(
            max_cluster_workers=16,
            cpus_per_worker=4,
            ram_per_worker_gb=16,
            min_gpu_memory_gb=5,
            ray_env="/work/pixi/global/.pixi/envs/default",
        )
    )

    assert headers["af-max-workers"] == "16"
    assert "af-n-workers" not in headers
    assert headers["af-gpus-per-worker"] == "1"


def test_ray_contract_rejects_elastic_max_above_sixteen():
    with pytest.raises(ValueError, match=r"\[1, 16\]"):
        ray_headers(
            SimpleNamespace(
                max_cluster_workers=17,
                cpus_per_worker=4,
                ram_per_worker_gb=16,
                min_gpu_memory_gb=5,
                ray_env="/work/pixi/global/.pixi/envs/default",
            )
        )


def test_ray_contract_rejects_exact_and_elastic_worker_options_together():
    with pytest.raises(ValueError, match="mutually exclusive"):
        ray_headers(
            SimpleNamespace(
                cluster_workers=4,
                max_cluster_workers=16,
                cpus_per_worker=4,
                ram_per_worker_gb=16,
                min_gpu_memory_gb=5,
                ray_env="/work/pixi/global/.pixi/envs/default",
            )
        )


def test_ray_driver_keeps_fold_task_in_an_importable_module():
    fold_task = ray_train_driver.train_dnn._train_one_fold_ray

    assert fold_task.__module__ == "MVA_training.VBF_run3.train_dnn"
    assert pickle.loads(pickle.dumps(fold_task)) is fold_task
