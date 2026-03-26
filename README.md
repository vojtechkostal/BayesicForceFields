# Bayesic Force Fields (BFF)

## Description

Bayesic Force Fields (BFF) is a Python package for optimizing
fixed-charge force-field parameters for molecular dynamics,
specifically `charges` and Lennard-Jones parameters `sigma` and
`epsilon`. The optimization uses structural data derived from reference
trajectories, typically obtained through *ab initio* molecular dynamics
simulations, in order to account for `electronic polarization effects`.

The optimization uses `Bayesian inference` driven by Markov Chain Monte
Carlo (MCMC) sampling with a `Local Gaussian process (LGP)` surrogate
for the usually costly MD simulations. The LGP hyperparameters are
optimized and their uncertainty can be propagated to the posterior.

## Dependencies (non-standard)

* [SciPy](https://scipy.org/)
* [MDAnalysis](https://www.mdanalysis.org) (trajectory analysis)
* [PyTorch](https://pytorch.org/get-started/locally/)
  (surrogate modeling with optional GPU/CUDA acceleration)
* [PyYAML](https://pypi.org/project/PyYAML/) (handling .yaml files)
* [Matplotlib](https://matplotlib.org/) (Plotting)
* [gmxtop](https://github.com/vojtechkostal/gmxtop) (Topology handling)

## Installation
1. Clone this repository and create a fresh environment with `mamba`
   or `conda`:
```sh
cd BayesicForceFields
mamba env create -f ./environment.yaml
mamba activate bff
```
2. Install prerequisites:
   - `PyTorch`: install the CPU or GPU build following
     <https://pytorch.org/get-started/locally/>.
   - `GROMACS`: make sure it is available from the command line via
     `gmx` (or a compatible alternative configured in BFF).
   - `gmxtop`: follow the installation instructions at
     <https://github.com/vojtechkostal/gmxtop>.

3. Install BFF as:
```sh
pip install -e .
```

## Usage
Here, we'll go through the acetate example from `examples/acetate`.
Each stage includes a working example config, but production runs will
typically require edits for local paths, compute resources, and target
simulation lengths.

Partial charges on acetate are optimized. The parameterization uses data
from three systems: 1. aqueous acetate, 2. acetate with calcium kept at
contact distance from the carboxyl group, and 3. acetate with calcium
kept in a solvent-shared position relative to the carboxyl group.

1. Prepare the systems
```sh
cd ./examples/acetate/01-prepare
bff prepare config.yaml
```
This generates the files and directories needed for both reference
(CP2K) and training (GROMACS) trajectories. The step also performs the
initial equilibration needed to stage the systems. Reference trajectories
may later be replaced by any user-provided set of simulations with the
same system layout.

2. Run the reference calculations using CP2K.
This step can take a long time on a cluster with the default DFT setup.
For the example, reference trajectories are already provided in
compressed `.xtc` format after discarding the first 5 ps of
equilibration and saving every 50 fs.

3. Run training simulations.
For each sampled force-field parameter vector, a triplet of trajectories
is calculated. This step can be run either locally via
`config-local.yaml` or on a cluster via `config-slurm.yaml`.
```sh
cd ../03-training-trjs
bff simulate config-local.yaml
```

4. Analyze quantities of interest.
Analyze the QoIs for both the training and reference datasets.
```sh
cd ../04-qoi
bff analyze config.yaml
``` 

5. Learn the posterior distribution.
Infer the posterior distribution of force-field parameters that best
reproduces the reference data. This step can be executed from the
command line, which is convenient for batch runs, or from the
`05-learning/learn.ipynb` notebook.
```sh
cd ../05-learning
bff learn config.yaml
```
The notebook can export posterior draws to `posterior-samples.yaml`,
which can then be passed directly to the validation workflow.

6. Validate posterior samples.
Rerun selected force-field samples from the learned posterior using
`bff validate` and a validation config such as
`03-training-trjs/config-validate-local.yaml`.
```sh
cd ../03-training-trjs
bff validate config-validate-local.yaml
```

7. Visualization
Use the notebook in `06-visualize` to inspect the posterior samples and
produce the standard plots from step 5.
