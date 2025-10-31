# Bayesic Force Fields (BFF)

## Description

Bayesic Force Fields (BFF) is a Python package designed for optimizing parameters of fixed-charge force fields for molecular dynamics, specifically the `charges`, and van der Waals parameters `sigma` and `epsilon`.
The optimization is based on structural data derived from reference trajectories, typically obtained through *ab initio* molecular dynamics simulations in order to effectively account for the `electronic polarization effects`.

The optimization uses `Bayesian inference` driven by Markov Chain Monte Carlo (MCMC) sampling with a `Local Gaussian process (LGP)` as a surrogate model for the (usually) costly MD simulations.
The LGP hyperparameters are optimized and their uncertainty can be propagated to the posterior.

## Dependencies (non-standard)

* [SciPy](https://scipy.org/)
* [MDAnalysis](https://www.mdanalysis.org) (analyzing MD trajectories)
* [emcee](https://emcee.readthedocs.io/en/stable/#) (MCMC sampling within the Bayesian inference)
* [ParmEd](https://parmed.github.io/ParmEd/html/index.html) (topology handling)
* [PyTorch](https://pytorch.org/get-started/locally/) (Efficient surrogate modeling using GPU and CUDA)
* [PyYAML](https://pypi.org/project/PyYAML/) (handling .yaml files)
* [Matplotlib](https://matplotlib.org/) (Plotting)

## Installation
1. Clone this repository and (preferably) create and activate a new environment using `mamba` or `conda` as:
```sh
cd BayesicForceFields
mamba env create -f ./environment.yml
mamba activate bff
```

3. install the package:
 > [!WARNING]
The package requires `PyTorch` to be installed and it benefits hugely from the GPU acceleration.
However, the GPU-enabled PyTorch installation cannot be automatized so at this point, please install PyTorch using one of the options here: https://pytorch.org/get-started/locally/.

 > [!WARNING]
Also, the code needs `Gromacs` installation to be accesible to run force-field MDs during the initial setup and during generation of the training set.

```sh
pip install .
```

## Usage
Here, we'll go through an example that can be found in the `examples` directory.
For each stage, there is a config file provided, however, they should be altered for production level calculations.

Partial charges on acetete are optimized.
The parameterization uses data from a triplet of simulations: 1. aqueous species, 2. aqueous species with calcium cation placed at contant distance with carboxyl group, 3. aqueous species with calcium cation placed at solvent-shared position with respect to the carboxyl.

1. Create the systems
```sh
cd ./examples/acetate/01-prepare
bff initialize config.yml
```
Generates the necessary files and directories for both reference (CP2K) and training (GROMACS) trajectories.
This step also includes initial NpT equilibration for all systems.
Later, user can actually use an arbitrary set of reference simulations.

2. Run the reference calculations using CP2k.
This step will take a long time (weeks on a cluster) when using the default DFT level of theory.
It can be speeded up by using semiempirical method at a cost of accuracy or by using some flavor of neural network potentials (NNPs).
At this moment, the latter is not implemented and for example purpose, we provide trajectories in compressed .xtc format after discarding the first 5ps for equilibration and saved every 50 fs.

3. Run training simulations.
For each set of parameter distribution, a triplet of trajectories is calculated.
This step can be run either locally using the `config-local.yaml` config or the simulations can be sent to a cluster in parallel using `config-slurm.yaml`.
```sh
cd ../03-training-trjs
bff runsims config-local.yaml
```

4. Analyze quantities of interest.
Analyze the QoIs for the training and reference dataset.
```sh
cd ../04-qoi
bff analyze config.yaml
``` 

5. Learn the posterior distribution.
Infer the most likely set of force-field parameters that reproduce the reference data.
This step can be executed either via command line as the steps above (usefull when working on a cluster) but it also can be done in a jupyter notebook which is convenient for the subsequent visualization and analysis.
```sh
cd ../05-learning
bff learn config.yaml
```

or you can open a jupyter notebook (`analysis.ipynb`) in the `05-learning-interactive` directory and run the optimization interactively.

6. Visualization
Using the afforementioned jupyter notebook, you can either use the interactive code there or you can load the generated files and visualize results.
