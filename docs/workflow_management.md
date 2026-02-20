---
title: Workflow management: Snakemake
---


# Reference

- [CMS CAT hackathon](https://indico.cern.ch/event/1623559/timetable/?view=standard)
- [Snakemake CMS Tutorial](https://alefisico.github.io/snakemake-cms-tutorial/index.html)

# Commands

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
    
    snakemake --rulegraph | dot -Tpng > rulegraph.png
    ```

# Issue

1. For the plotting code, zpt and mass calibration it is reading the input file path from `trials.yaml`. But, as I am using the snakemake this looks like not a good idea. I should pass it from the command line so that these steps takes the appropriate input informations.