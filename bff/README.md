# Bayesic Force Fields (BFF)

## Description

Bayesic Force Fields (BFF) is a Python package designed for optimizing parameters of fixed-charge force fields for molecular dynamics, specifically the `charges`, and van der Waals parameters `sigma` and `epsilon`.
The optimization is based on structural data derived from reference trajectories, typically obtained through *ab initio* molecular dynamics simulations in order to effectively account for the `electronic polarization effects`.

The optimization uses the `Bayesian inference` driven by Markov Chain Monte Carlo (MCMC) sampling with `Local Gaussian process (LGP)` as a surrogate model replacing running the (usually) costly MD simulations.
LGP's hyperparameters can be also infered and their uncertainty propagated into the final force field optimization. 

## Workflow

1. **Create System**  
   Generates the necessary files and directories for both reference (CP2K) and training (GROMACS) trajectories. This step also includes NpT equilibration for all systems.
   Later, user can actually use an arbitrary set of reference simulations
   
2. **Generate Reference Simulations**  
   Submit *ab initio* molecular dynamics (AIMD) simulations using CP2K (which may take several weeks to complete).
   In the furture, this should be speeded up by leveraging NNPs.

3. **Generate Training Data**  
   Submits training simulations (either locally or at a cluster - currently, only SLURM submission system is supported) by sampling parameters from user-defined ranges to generate a set of diverse data for model training.

4. **Optimization**
   - Load & analyze both training and reference trajectories
   - Train the LGP surrogate model to predict the quantities of interest (rdfs, hbonds, restrained distance distributions) given a set of parameters
   - Run the MCMC-based Bayesian optimization.

5. **Visualize the results**  
   In the examples, we demonstrate how the results can be presented

## Dependencies (non-standard)

* [SciPy](https://scipy.org/)
* [MDAnalysis](https://www.mdanalysis.org) (analyzing MD trajectories)
* [emcee](https://emcee.readthedocs.io/en/stable/#) (MCMC sampling within the Bayesian inference)
* [ParmEd](https://parmed.github.io/ParmEd/html/index.html) (topology handling)
* [PyTorch](https://pytorch.org/get-started/locally/) (Efficient surrogate evaluation using GPU and CUDA)
* [PyYAML](https://pypi.org/project/PyYAML/) (handling .yaml files)
* [Matplotlib](https://matplotlib.org/) (Plotting)
