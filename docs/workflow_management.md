---
title: Workflow management: Snakemake
---

- Update the `workflow/Snakefile` as per the need and the config file present in `workflow/config.yaml`.

1. Command:

    ```bash
    snakemake -s workflow/Snakefile -j 1 --resources gateway=1 --rerun-incomplete --restart-times 3 --latency-wait 60
    ```

2. Check summary

    ```bash
    snakemake -s workflow/Snakefile --summary
    ```

3. Visualize the DAG

    ```bash
    snakemake -s workflow/Snakefile --dag | dot -Tpng > dag.png
    snakemake -s workflow/Snakefile --dag | dot -Tpdf > dag.pdf
    ```

