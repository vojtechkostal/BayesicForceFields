## Unreleased

### Changed

- Built-in `rdf` and `hydrogen_bonds` routines now use the same interface as
  custom routines and validate their own options when they run. Their outputs
  are unchanged; hydrogen-bond analysis is substantially faster.
- Moved `bff.qoi.data` to `bff.qoi.dataset` and `bff.tools.get_unitcell` to
  `bff.qoi`. Custom routines import `QoI`, `get_unitcell`, and `select_atoms`
  from `bff.qoi`.

## `0.4.2` - 2026-10-02

### Changed

- Bounded the posterior sample counts used for default corner, standard
  marginal, and QoI marginal plots. The standard marginal cap accepts `-1` or
  `null` to use every sample. QoI likelihood attribution now runs in adaptive
  batches to avoid CUDA memory exhaustion after learning.
- Centered default LGP hyperparameter priors on the observed input and residual
  target scales, avoiding noise-dominated fits for scalar QoIs whose natural
  parameter or output scales differ substantially from one.
- Reported the mode of each posterior marginal instead of its mean, limited
  charge modes to three decimal places, and formatted other parameter modes
  with three significant digits.
- Balanced marginal plots across rows of at most five parameters, placed each
  arbitrary `define` parameter on an independent axis, and added configurable
  axis labels through `plots.plot_metadata`. Marginal figures now reserve
  measured space for legends so they cannot overlap single- or multi-panel
  plots, align y-axis labels within subplot columns, and leave sufficient
  gutters between neighboring panel sections.
- Made local validation progress visible immediately, including campaigns with
  a single sample, and advanced the displayed count only after each MD job has
  completed.

## `0.4.1` - 2026-08-25

### Changed

- Simplified RDF calculation so selection handling and atom-type expansion
  occur in the QoI adapter while the numerical kernel operates directly on
  MDAnalysis AtomGroups. Dynamic selections and numerical behavior are
  preserved.
