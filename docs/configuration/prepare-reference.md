# Prepare Reference

Implementation:

- `bff/workflows/prepare_reference/config.py`
- `bff/workflows/prepare_reference/main.py`

`bff prepare-reference` converts selected systems from a completed build stage
into CP2K reference inputs. It does not copy the GROMACS inputs; sampling and
validation consume the build stage directly.

```yaml
source: ../01-build
output: ./
systems: [acetate, acetate-contact, acetate-separated]
n_single_point_snapshots: 1000
```

- `source` is the build output root containing `systems/<system_id>/`.
- `output` defaults to the configuration directory.
- `systems` is a required, explicit list of unique semantic IDs.
- `n_single_point_snapshots` must be positive.
- `log` optionally overrides `prepare-reference.log`.

For each system the command writes:

```text
systems/<system_id>/
  system.yaml
  system.top
  system.gro
  system.xyz
  md/
  snapshots/
    md.inp
    sp.inp
    xyz/snapshot-0000.xyz
  single-atoms/<element>/
```

`system.yaml` contains physical metadata, the snapshot count, and the sorted
element set. It contains no file paths or version field. File roles follow the
documented directory contract.
