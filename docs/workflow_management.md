---
title: Workflow management: Snakemake
---

- Update the `workflow/Snakefile` as per the need and the config file present in `workflow/config.yaml`.

Command:

```bash
snakemake -s workflow/Snakefile -j 4 --resources gateway=1 --rerun-incomplete --restart-times 3 --latency-wait 60
```
