# CALVADOS2 data provenance

Upstream: [KULL-Centre/papers, CG-cutoffs-Tesei-et-al](https://github.com/KULL-Centre/papers/tree/85979b1b4123b6b5391b617d16551969eda9f56e/2022/CG-cutoffs-Tesei-et-al).
Pinned commit: `85979b1b4123b6b5391b617d16551969eda9f56e`.
Downloaded subset: `optimization/M1_24`, approximately 23 MB.

Source study: Tesei and Lindorff-Larsen, *Improved predictions of phase behaviour
of intrinsically disordered proteins by tuning the interaction range*,
DOI: [10.12688/openreseurope.14967.2](https://doi.org/10.12688/openreseurope.14967.2).
Additional archive: [Zenodo](https://zenodo.org/records/6901205).

`fetch_data.py` pins the Git blob SHA-1 of every downloaded file. Existing
files are verified and never silently replaced. Raw bytes remain unchanged.
The converter verifies the same hashes before loading the upstream pandas
pickles; pickle loading is only appropriate for trusted sources.

## Included files

- `1_chi2.pkl` through `11_chi2.pkl`: 1,318 distinct accepted/initial parameter
  records, with `chi2_rg`, `chi2_pre`, `theta_prior`, `xi`, and `cost`.
  These are optimization histories, not posterior samples. The training script
  uses the parameter vector embedded in each row (not a cycle-end vector),
  and retains the cycle and iteration in each sample ID.
- `residues.csv`: 20 residue stickiness parameters and optimization-cycle history.
- `1_init_proteinsRgs.pkl` through `11_init_proteinsRgs.pkl`: 51 proteins,
  their conditions, experimental targets/errors, mean Rg and per-frame Rg.
- `selHPS.csv`: original development hydrophobicity scales; `prepare_priors.py`
  uses all 70 scale vectors for the joint KDE and independent beta alternatives.
  Residue IDs are mapped through `residues.csv` into the original GP column order.

## Mapping and units

In upstream `optimization/code/optimize.py`, the initial Rg snapshots are saved
before reweighting. The initial parameter vector is stored as
`lambdas_{cycle-1}`, while the optimized vector is stored as `lambdas_{cycle}`.
Thus snapshot `n_init_proteinsRgs.pkl` pairs with `lambdas_{n-1}`, not
`lambdas_n`. This subset supplies 11 parameter vectors, columns 0 through 10.
Column 11 is not paired with an initial cycle-12 snapshot here.

Upstream `analyse.py:calcRg` calculates mass-weighted Rg from MDTraj coordinates
in nm, and averages Rg across frames (not the square root of mean squared Rg).
Temperatures are K, ionic strengths M; the M1_24 interaction cutoff is 2.4 nm
(see upstream `submit.py`). The converter checks saved Rg against the frame
mean, matches proteins and residues by ID, and checks sequences, conditions,
targets, and cycle paths across snapshots.

The original `expRgErr` becomes each scalar dataset's fixed observation-noise
standard deviation. This does not estimate simulation sampling uncertainty or
model discrepancy. Review the error interpretation before scientific inference.

## Local collaborator data

`c2_rg.csv` was supplied privately by Giulio and contains unpublished CALVADOS2
predictions alongside experimental values. It is left untouched and is not
automatically merged: protein identity, construct sequence, conditions, units,
and experimental provenance require reconciliation first. No publication or
redistribution permission is inferred from its presence here.
