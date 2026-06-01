# Sample Configuration

Source code:

- `bff/workflows/sample/config.py`
- `bff/workflows/sample/main.py`
- `bff/workflows/_shared/campaign.py`

## Purpose

`bff sample` draws parameter vectors, stages a sampled FFMD campaign, and runs
the corresponding GROMACS jobs.

## Minimal Example

```yaml
campaign_dir: ./
systems:
  - assets: ../02-assets/ffmd/system-000
    n_steps: 1000
bounds:
  charge C2: [0.0, 1.0]
  charge O1 O2: [-0.8, -0.3]
charge_constraints:
  - selection: "resname ACE"
    target: -0.8
    scope: residue
    implicit: "charge C2"
n_samples: 10
gmx_cmd: gmx
job_scheduler: local
```

## Top-Level Keys

- `campaign_dir`
  Output directory for sampled topologies, metadata, and trajectories.
- `systems`
  Non-empty list of staged asset directories plus system-specific MD lengths.
- `bounds`
  Mapping from parameter label to lower and upper bounds.
- `charge_constraints`
  Charge equations compiled from MDAnalysis selections. Each constraint defines
  `selection`, `target`, `scope`, and a distinct bounded `implicit` parameter.
- `n_samples`
  Number of force-field vectors sampled for the campaign.
- `gmx_cmd`
  GROMACS executable.
- `job_scheduler`
  Either `local` or `slurm`.
- `dispatch`
  If `true`, launch jobs immediately after staging them.
- `compress`
  If `true`, compress finished simulation outputs.
- `cleanup`
  If `true`, remove temporary files after successful runs.
- `store`
  Which trajectory outputs to keep. Defaults to `['xtc']`.
- `slurm`
  Slurm-only runtime settings.

## `systems[]` Keys

- `assets`
  Directory created by `bff prepare-assets`, for example `ffmd/system-000`.
- `n_steps`
  Production MD length for this prepared system within the sampled campaign.

## `charge_constraints[]` Keys

- `selection`
  MDAnalysis atom selection. Selections must be disjoint or strictly nested;
  partial overlaps are rejected.
- `target`
  Required total charge for the selected group.
- `scope`
  Either `system` for one complete-system sum or `residue` to apply the target
  independently to each selected residue.
- `implicit`
  Parameter reconstructed to satisfy this equation. It must exist in `bounds`,
  belong to this selection, and not occur in a descendant selection.

Nested constraints are reconstructed from the smallest selected groups outward.

## Parameter Labels

The `bounds` keys determine which GROMACS force-field parameters are sampled
and subsequently learned. BFF currently supports:

| Parameter | Label syntax | Example | GROMACS quantity |
| --- | --- | --- | --- |
| Partial charge | `charge <name-or-type> [<name-or-type> ...]` | `charge O1 O2` | Atomic charge |
| Lennard-Jones sigma | `sigma <atom-type> [<atom-type> ...]` | `sigma OW` | LJ sigma |
| Lennard-Jones epsilon | `epsilon <atom-type> [<atom-type> ...]` | `epsilon OW` | LJ epsilon |
| Function-9 dihedral force constant | `dihedraltype9_<multiplicity>_<phase>` | `dihedraltype9_3_180` | Periodic-dihedral force constant |

Values use the native units of the GROMACS topology: elementary charge for
partial charges, nm for sigma, kJ mol^-1 for epsilon, degrees for the
dihedral phase, and kJ mol^-1 for the dihedral force constant.

### Charges

Charge labels resolve each token by atom name first and fall back to atom type
when no atom has that name. Multiple tokens in one label tie all matching atoms
to one sampled value:

```yaml
bounds:
  charge O1 O2: [-0.8, -0.3]
  charge HW: [0.1, 0.6]
```

Here, `O1` and `O2` share one charge parameter. If there is no atom named
`HW`, all atoms of type `HW` share the second parameter. Charge labels must not
overlap: one topology atom cannot be controlled by two entries in `bounds`.

Charge parameters may participate in the hierarchical `charge_constraints`
described above. Parameters of the other supported families are sampled
directly.

### Lennard-Jones Parameters

Sigma and epsilon labels address GROMACS atom types. Multiple atom types in one
label tie those types to one sampled value:

```yaml
bounds:
  sigma OW: [0.25, 0.38]
  epsilon OW: [0.58, 0.72]
  sigma NA CL: [0.20, 0.45]
```

### Function-9 Dihedrals

To sample a GROMACS function-9 dihedral force constant, use
`dihedraltype9_<multiplicity>_<phase>`:

```yaml
bounds:
  dihedraltype9_3_180: [0.0, 10.0]
```

This updates the force constant of every matching function-9 dihedral term in
the topology while preserving its multiplicity and phase. If the topology has
several function-9 terms with the same multiplicity and phase, the label ties
all of them to the same sampled value.

## Outputs

`bff sample` writes:

- `campaign_dir/specs.yaml`
- `campaign_dir/samples.yaml`
- per-sample job configs
- per-sample modified topologies
- stored trajectories and energy files
