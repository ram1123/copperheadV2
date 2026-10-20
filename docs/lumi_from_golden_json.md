# Luminosity from a Golden JSON

`scripts/compute_lumi_from_golden_json.sh` computes the certified **recorded** integrated luminosity (in
pb⁻¹) covered by a Golden JSON lumimask, using the official `brilcalc` tool. Use it to cross-check the
`total_lumi_pb` / `lumi_pb` values in `configs/datasets/dataset_*.yaml`.

## References

- Run-3 luminosity recommendations (lumi POG): <https://twiki.cern.ch/twiki/bin/view/CMS/LumiRecommendationsRun3>
- `brilcalc` documentation: <https://cms-service-lumi.web.cern.ch/cms-service-lumi/brilwsdoc.html>
- `brilws-docker` (the container the script runs `brilcalc` in): <https://gitlab.cern.ch/cms-cloud/brilws-docker>,
  also on CVMFS at `/cvmfs/cms-bril.cern.ch/cms-lumi-pog/brilws-docker/` (see its `README.md`)
- Normtags: `/cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/` — the script defaults to `normtag_PHYSICS.json`,
  the lumi POG recommendation for physics analyses
- Golden JSONs: `/cvmfs/cms-griddata.cern.ch/cat/metadata/DC/Collisions<YY>/<date>/`

The script runs:

```bash
brilcalc lumi -c web -u /pb -b "STABLE BEAMS" \
    --normtag /cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_PHYSICS.json \
    -i <golden_json>
```

and prints the `totrecorded(/pb)` column of the `#Summary` block.

## Environment setup

No pixi environment and no local `brilws` install are needed. `brilws` depends on a CERN-internal
SQLAlchemy FroNTier dialect that is not on public PyPI, so the script instead sources
`/cvmfs/cms-bril.cern.ch/cms-lumi-pog/brilws-docker/brilws-env`, which runs `brilcalc` inside the
`brilws-docker` image via Singularity/Apptainer. You need:

1. **CVMFS** — `/cvmfs/cms-bril.cern.ch` (brilws-env, normtags) and, for the Golden JSONs,
   `/cvmfs/cms-griddata.cern.ch`. Both are mounted on the Purdue AF.
2. **Singularity/Apptainer** — available on the Purdue AF at `/usr/bin/singularity`.
3. **A valid VOMS proxy** — `brilcalc -c web` authenticates with it. If `X509_USER_PROXY` is unset, the
   script uses the repo-local `voms_proxy.txt` when present. To create or renew one (interactive, asks
   for your grid-certificate passphrase):

    ```bash
    export X509_USER_PROXY=$PWD/voms_proxy.txt
    voms-proxy-init --voms cms --valid 192:00 --out $X509_USER_PROXY
    voms-proxy-info   # check the remaining lifetime
    ```

## Usage

```bash
bash scripts/compute_lumi_from_golden_json.sh <golden_json_path> [normtag_path]
```

- `golden_json_path` — absolute or relative path to the Golden JSON.
- `normtag_path` (optional) — a different normtag, e.g.
  `/cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_BRIL.json`. Defaults to `normtag_PHYSICS.json`.

Example (2023 era C):

```bash
bash scripts/compute_lumi_from_golden_json.sh \
    /cvmfs/cms-griddata.cern.ch/cat/metadata/DC/Collisions23/2026-05-28/Cert_Collisions2023_eraC_367095_368823_Golden.json
# stdout: 18062.659110577
```

### Output

The recorded luminosity in pb⁻¹ is the **only** thing on stdout, so it can be captured directly:

```bash
L=$(bash scripts/compute_lumi_from_golden_json.sh <golden_json>)
```

The `brilcalc` summary table and the check for runs/lumisections in the JSON that are not in the
`brilcalc` results go to stderr, so they still show up in the terminal.

If the input files are missing or `brilcalc` fails, the script exits non-zero; on a `brilcalc` failure it
prints its last 20 lines of output to stderr.

## Notes

- The script deliberately does **not** use `set -e`/`set -u`, and clears its own positional arguments
  before sourcing `brilws-env`: `brilws-env` parses the caller's arguments as Singularity options and
  fails silently under `-e`/`-u`. Either problem makes the script print nothing after its header —
  keep both in mind if you edit it.
- Validated on the 2023 era-C JSON above (run from `/tmp`, `X509_USER_PROXY` unset, so the repo proxy was
  used): exit code 0, output `18062.659110577`, matching a hand-run `brilcalc`. The error paths (missing
  file, failed `brilcalc`) have not been tested.
