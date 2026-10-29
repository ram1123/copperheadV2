---
title: VBF-DNN Run 2 and Run 3 Input Columns
---

# VBF-DNN Run 2 / Run 3 输入变量列对照

本文档只描述当前 VBF-DNN 真正送入神经网络的输入列。配置文件是唯一来源：

- Run 2：[configs/dnn_run2_vbf.yaml](../configs/dnn_run2_vbf.yaml)
- Run 3：[configs/dnn_run3_vbf.yaml](../configs/dnn_run3_vbf.yaml)

## 一眼看懂

| 模型 | 物理变量 | 年份 one-hot | 总输入维数 |
|---|---:|---:|---:|
| Run 2 VBF-DNN | 25 | 11 | **36** |
| Run 3 VBF-DNN | 23 | 7 | **30** |

Run 3 的30列是 Run 2 的36列的严格子集。Run 2 独有以下6列，Run 3 没有任何独有列：

```text
jet1_btagUParTAK4QvG_nominal
jet2_btagUParTAK4QvG_nominal
year_2016preVFP
year_2016postVFP
year_2017
year_2018
```

因此默认网络的第一层分别是：

```text
Run 2: Linear(36, 384)
Run 3: Linear(30, 384)
```

两套模型的 checkpoint、scaler 和 `training_features.pkl` 必须配套使用，不能互换。

## 逐列对照

“序号”就是该列送入网络时的位置；`—` 表示该模型不使用此列。

| 类别 | 输入变量列 | Run 2 序号 | Run 3 序号 | 说明 |
|---|---|---:|---:|---|
| 双缪子 | `dimuon_mass` | 1 | 1 | 二缪子不变质量 |
| 双缪子 | `dimuon_pt` | 2 | 2 | 二缪子系统横动量 |
| 双缪子 | `dimuon_pt_log` | 3 | 3 | `log(dimuon_pt)` |
| 双缪子 | `dimuon_rapidity` | 4 | 4 | 二缪子系统 rapidity |
| 双缪子 | `dimuon_ebe_mass_res` | 5 | 5 | event-by-event 质量分辨率 |
| 双缪子 | `dimuon_ebe_mass_res_rel` | 6 | 6 | 相对质量分辨率 `mass_res / dimuon_mass` |
| 双缪子 | `dimuon_cos_theta_cs` | 7 | 7 | Collins-Soper frame 的 `cos(theta)` |
| 双缪子 | `dimuon_phi_cs` | 8 | 8 | Collins-Soper frame 的 `phi` |
| Jet 1 | `jet1_pt_nominal` | 9 | 9 | leading jet 的横动量 |
| Jet 1 | `jet1_eta_nominal` | 10 | 10 | leading jet 的 `eta` |
| Jet 1 | `jet1_phi_nominal` | 11 | 11 | leading jet 的 `phi` |
| Jet 1 | `jet1_btagUParTAK4QvG_nominal` | **12** | — | **仅 Run 2：leading jet 的 UParT Q-vs-G 判别量** |
| Jet 2 | `jet2_pt_nominal` | 13 | 12 | subleading jet 的横动量 |
| Jet 2 | `jet2_eta_nominal` | 14 | 13 | subleading jet 的 `eta` |
| Jet 2 | `jet2_phi_nominal` | 15 | 14 | subleading jet 的 `phi` |
| Jet 2 | `jet2_btagUParTAK4QvG_nominal` | **16** | — | **仅 Run 2：subleading jet 的 UParT Q-vs-G 判别量** |
| Dijet | `jj_mass_nominal` | 17 | 15 | 两个 jet 的不变质量 |
| Dijet | `jj_mass_log_nominal` | 18 | 16 | `log(jj_mass)` |
| Dijet | `jj_dEta_nominal` | 19 | 17 | 两个 jet 的绝对 `delta eta` |
| 软辐射 | `nsoftjets5_nominal` | 20 | 18 | `pT > 5 GeV` 的 soft-activity jet 数量 |
| 软辐射 | `htsoft2_nominal` | 21 | 19 | `pT > 2 GeV` 的 soft-activity jet 标量 `pT` 和 |
| VBF 拓扑 | `rpt_nominal` | 22 | 20 | `pT(mumu+j1+j2) / (pT(mumu)+pT(j1)+pT(j2))` |
| VBF 拓扑 | `ll_zstar_log_nominal` | 23 | 21 | `log(abs(Zeppenfeld variable))` |
| VBF 拓扑 | `mmj_min_dEta_nominal` | 24 | 22 | 二缪子系统与两个 jet 的最小绝对 `delta eta` |
| VBF 拓扑 | `pt_centrality_nominal` | 25 | 23 | 二缪子系统相对两个 jet 的 `pT` centrality |
| 年份 one-hot | `year_2016preVFP` | **26** | — | **仅 Run 2** |
| 年份 one-hot | `year_2016postVFP` | **27** | — | **仅 Run 2** |
| 年份 one-hot | `year_2017` | **28** | — | **仅 Run 2** |
| 年份 one-hot | `year_2018` | **29** | — | **仅 Run 2** |
| 年份 one-hot | `year_2022preEE` | 30 | 24 | 两者都有 |
| 年份 one-hot | `year_2022postEE` | 31 | 25 | 两者都有 |
| 年份 one-hot | `year_2023` | 32 | 26 | 两者都有 |
| 年份 one-hot | `year_2023BPix` | 33 | 27 | 两者都有 |
| 年份 one-hot | `year_2024` | 34 | 28 | 两者都有 |
| 年份 one-hot | `year_2025` | 35 | 29 | 两者都有 |
| 年份 one-hot | `year_2026` | 36 | 30 | 两者都有 |

## Run 2：完整36列，可直接复制

顺序必须保持不变：

```yaml
- dimuon_mass
- dimuon_pt
- dimuon_pt_log
- dimuon_rapidity
- dimuon_ebe_mass_res
- dimuon_ebe_mass_res_rel
- dimuon_cos_theta_cs
- dimuon_phi_cs
- jet1_pt_nominal
- jet1_eta_nominal
- jet1_phi_nominal
- jet1_btagUParTAK4QvG_nominal
- jet2_pt_nominal
- jet2_eta_nominal
- jet2_phi_nominal
- jet2_btagUParTAK4QvG_nominal
- jj_mass_nominal
- jj_mass_log_nominal
- jj_dEta_nominal
- nsoftjets5_nominal
- htsoft2_nominal
- rpt_nominal
- ll_zstar_log_nominal
- mmj_min_dEta_nominal
- pt_centrality_nominal
- year_2016preVFP
- year_2016postVFP
- year_2017
- year_2018
- year_2022preEE
- year_2022postEE
- year_2023
- year_2023BPix
- year_2024
- year_2025
- year_2026
```

## Run 3：完整30列，可直接复制

顺序必须保持不变：

```yaml
- dimuon_mass
- dimuon_pt
- dimuon_pt_log
- dimuon_rapidity
- dimuon_ebe_mass_res
- dimuon_ebe_mass_res_rel
- dimuon_cos_theta_cs
- dimuon_phi_cs
- jet1_pt_nominal
- jet1_eta_nominal
- jet1_phi_nominal
- jet2_pt_nominal
- jet2_eta_nominal
- jet2_phi_nominal
- jj_mass_nominal
- jj_mass_log_nominal
- jj_dEta_nominal
- nsoftjets5_nominal
- htsoft2_nominal
- rpt_nominal
- ll_zstar_log_nominal
- mmj_min_dEta_nominal
- pt_centrality_nominal
- year_2022preEE
- year_2022postEE
- year_2023
- year_2023BPix
- year_2024
- year_2025
- year_2026
```

## 哪些列必须来自 Stage-1 Parquet

上面列表中的物理变量必须存在于 compacted Stage-1 Parquet：

- Run 2：25个物理变量；
- Run 3：23个物理变量。

所有 `year_*` 列不需要存在于 Stage-1 Parquet。`preprocess_dnn.py` 会根据命令行 `--years` 和当前正在处理的年份自动生成 one-hot：当前年份对应列为 `1.0`，其余年份列为 `0.0`。

例如处理2017事件时，Run 2 输入为：

```text
year_2017 = 1.0
其余 year_* = 0.0
```

纯 Run 2 训练中，`year_2022preEE` 到 `year_2026` 始终为零；这些列出现在 Run 2 配置中不代表 Run 2 和 Run 3 样本被自动混合训练。

## 预处理还需要、但不会送入DNN的列

以下列用于事件选择、fold 划分或训练权重，不属于模型输入特征：

```text
event
wgt_nominal
njets_nominal
nBtagLoose_nominal
nBtagMedium_nominal
```

用途如下：

| 列 | 用途 |
|---|---|
| `event` | 用 `event % 4` 确定四折划分 |
| `wgt_nominal` | 训练损失、scaler 和评价指标的事件权重 |
| `njets_nominal` | VBF 选择和缺失 jet 的清理逻辑 |
| `nBtagLoose_nominal` | b-jet veto |
| `nBtagMedium_nominal` | b-jet veto |

## 缩放规则

- 连续物理变量使用训练 fold 计算的 weighted standard scaler。
- `nsoftjets5_nominal` 不缩放。
- 所有 `year_*` one-hot 列不缩放。
- 推理时必须使用与对应 fold 模型一起产生的 `scalers_<fold>.npz`。
