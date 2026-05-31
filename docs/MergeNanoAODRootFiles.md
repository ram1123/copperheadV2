---
title: Merge NanoAOD ROOT Files
---

# Merge NanoAOD ROOT Files

The script [scripts/mergeNanoAODRootFiles.py](../scripts/mergeNanoAODRootFiles.py) merges many
NanoAOD ROOT shards with `haddnano.py`, stages the merged result locally, copies it to the final
destination, and verifies the copy with Adler-32 checksums.

## What it does

1. Finds input `.root` files in a local directory.
2. Splits them into groups based on total input size.
3. Runs `haddnano.py` on each group.
4. Uses recursive batching if a single merge would exceed the maximum number of input files.
5. Copies the merged file to a local path or EOS destination.
6. Verifies the destination checksum before deleting the local staged file.

## Basic usage

```bash
python scripts/mergeNanoAODRootFiles.py \
  -i /path/to/input_dir \
  -o /store/user/<user>/merged/UL2018/sample_name \
  -f sample_name.root \
  -y UL2018
```

Local output is also supported:

```bash
python scripts/mergeNanoAODRootFiles.py \
  -i /path/to/input_dir \
  -o /tmp/merged_nanoaod \
  -f sample_name.root \
  -y UL2018
```

## Important options

- `--recursive`: search input ROOT files recursively
- `--scratch-dir`: local staging area for intermediate and merged files
- `--haddnano`: explicit path to `haddnano.py`
- `--chunk-size-mb`: maximum total input size per output chunk
- `--max-files-per-hadd`: maximum number of input files per single `haddnano.py` call
- `--eos-davs`: DAVS endpoint used for remote directory creation
- `--eos-root`: `root://` endpoint used for `xrdcp` and `xrdadler32`

If `--scratch-dir` is not provided, the script uses:

1. `$MERGE_NANOAOD_SCRATCH_DIR/<user>/<year>` when `MERGE_NANOAOD_SCRATCH_DIR` is set
2. otherwise a directory under the system temp area

## Output naming

The script currently writes one merged output per chunk, using:

- `sample.root` -> `sample_part1.root`
- `sample.root` -> `sample_part2.root`
- ...

This matches the existing workflow where very large input directories are intentionally split into
multiple merged outputs.

## External tools

For EOS output, the following tools must be available in the runtime environment:

- `gfal-mkdir`
- `xrdcp`
- `xrdadler32`

The script also relies on:

- ROOT Python bindings
- [haddnano.py](../haddnano.py)

## Notes

- Input discovery assumes the input directory is available on the local filesystem.
- EOS destinations can be provided as `/store/...`, `davs://...`, or `root://...`.
- The script verifies the merged local ROOT file before copying it to the final destination.
