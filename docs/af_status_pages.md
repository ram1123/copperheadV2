---
title: AF Status Pages
---

!!! info "Purdue AF only"
    These pages are served from inside your Purdue Analysis Facility session through the JupyterHub
    proxy, so the links only work from a browser logged into the AF hub.

## Snakemake status page

Every real `snakemake` run (not `-n` / `--summary`) starts a self-refreshing status page from the
Snakefile's `onstart:` and prints its link:

```
https://cms.geddes.rcac.purdue.edu/user/<your-user>/proxy/8765/snakemake/
```

Open it in a browser that is logged into the AF JupyterHub (the proxy uses the hub login). It shows,
per Snakemake run: per-rule progress, the year × jj-region stage2 → stage3 grid, a job table with log
tails, recent events, stats results, and for every `stage1` job a per-sample chunk table (state,
tries, start/end, wall time, events processed / expected, error text) read from
`<stage1_output>/<year>/_status/job_status.jsonl`. The page re-reads its data every 15 s; the
generator stops itself after 2 h with no snakemake running. Disable it for one run with
`--config status_page=False`.

Start it by hand (e.g. to look at finished runs, or after the pod restarted), from the repo root:

```bash
/work/pixi/global/.pixi/envs/default/bin/python workflow/snakemake_status_page.py --ensure
```

`--ensure` starts, each only if it is not already up, the page generator (detached job
`snakemake_status_page`) and the web server on port 8765 (detached job `status_page_server`, serving
`validation/job_status/`), then prints the link. The same server also serves the other status pages
under `validation/job_status/` (index at `.../proxy/8765/`). To start them separately:

```bash
PY=/work/pixi/global/.pixi/envs/default/bin/python
D=.claude/skills/run-copperheadV2-develop/detach_job.sh
MIN_PROXY_H=0 bash $D snakemake_status_page $PY workflow/snakemake_status_page.py --loop 15
MIN_PROXY_H=0 bash $D status_page_server $PY -m http.server 8765 --bind 127.0.0.1 --directory validation/job_status
# plans page (logs/plans/plans.yaml) -> .../proxy/8765/plans/
MIN_PROXY_H=0 bash $D status_page_plans $PY scripts/plans_status_page.py \
    --plans logs/plans/plans.yaml --out validation/job_status/plans --loop 60
```

The generator copies `workflow/snakemake_status_page.html` into the page dir only when it starts, so
after editing the page or the generator, restart it:
`kill -TERM -- -$(sed -n 2p logs/detached/snakemake_status_page.pid)`, then rerun `--ensure`.
A 502 / "connection refused" at the link means the server is down; `--ensure` brings it back.
