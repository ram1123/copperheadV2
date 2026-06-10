---
title: Cut Flow Cross Check before PR
---

# Command

```bash
time(bash run_analysis_pipeline.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l CrossCheckCutFlow_BR_fatjet -p HPScan_03Sep_17bins -y "2018PR" -m 0 -k )
time(bash run_analysis_pipeline.sh -c configs/datasets/dataset_nanoAODv12_run2.yaml -v 12 -l CrossCheckCutFlow_BR_fatjet_test -p HPScan_03Sep_17bins -y "2018PR" -m 1 -k )
```
