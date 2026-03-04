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

    snakemake -s workflow/Snakefile plot_all -j 1 --resources gateway=1 --rerun-incomplete --restart-times 3 --latency-wait 60

    # Force run some rule
    
    snakemake -s workflow/Snakefile -j 1 --resources gateway=2 --rerun-incomplete --restart-times 3 --latency-wait 60 --forcerun stage1Compact

    snakemake -s workflow/Snakefile --configfile workflow/config.yaml -j 1 --resources gateway=1 --rerun-incomplete --restart-times 3 --latency-wait 60


    # If the goal is “rerun plots only, do NOT rerun stage1/stage1Compact”, then restrict allowed rules:
    snakemake -s workflow/Snakefile plot_all  -j 3 --resources gateway=1 --rerun-incomplete --restart-times 3 --latency-wait 60 -R plots
    snakemake -s workflow/Snakefile plot_all  -j 3 --resources gateway=1 --rerun-incomplete --restart-times 3 --latency-wait 60 --allowed-rules plots  -R plots
    ```

2. Check summary

    ```bash
    snakemake -s workflow/Snakefile --summary
    
    snakemake -s workflow/Snakefile --summary | python workflow/summary_table.py
    ```

3. Visualize the DAG

    ```bash
    
    snakemake -s workflow/Snakefile --dag | dot -Tpng > dag.png
    
    snakemake -s workflow/Snakefile --dag | dot -Tpdf > dag.pdf

    snakemake --rulegraph | dot -Tpng > rulegraph.png
    snakemake --rulegraph | dot -Tpdf > rulegraph.pdf
    
    ```

# Issue

1. For the plotting code, zpt and mass calibration it is reading the input file path from `trials.yaml`. But, as I am using the snakemake this looks like not a good idea. I should pass it from the command line so that these steps takes the appropriate input informations.
