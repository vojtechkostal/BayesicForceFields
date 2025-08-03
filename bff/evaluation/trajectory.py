import MDAnalysis as mda
import inspect

from .hbonds import compute_all_hbonds, compute_hbonds
from .rdf import compute_all_rdfs, compute_rdf
from .restraints import compute_all_restraints, compute_probability_density
from ..structures import TrajectoryData

from ..topology import prepare_universe


def analyze_trajectory(
    universe: mda.Universe,
    mol_resname: str,
    restraints: list,
    rdf_kwargs: dict = None,
    hbond_kwargs: dict = None,
    restraint_kwargs: dict = None,
) -> TrajectoryData:
    """Analyze a single trajectory."""

    rdf_kwargs = rdf_kwargs or {}
    hbond_kwargs = hbond_kwargs or {}
    restraint_kwargs = restraint_kwargs or {}

    # Analyze RDFs, Hbonds and restraint coordinates
    rdfs = compute_all_rdfs(universe, mol_resname, **rdf_kwargs)
    hbonds = compute_all_hbonds(universe, mol_resname, **hbond_kwargs)
    restraints = compute_all_restraints(
        universe, restraints, **restraint_kwargs
    )

    return TrajectoryData(rdfs, hbonds, restraints)


def analyze_all_trajectories(
        fn_topol: list[str],
        fn_coord: list[str],
        fn_trjs: list[str],
        restraints: list[list],
        mol_resname: str,
        start: int = 0,
        stop: int = -1,
        step: int = 1,
        **kwargs
) -> list[TrajectoryData]:
    """Analyze all trajectories in the dataset."""
    results = []
    for t, c, trj, r in zip(fn_topol, fn_coord, fn_trjs, restraints):
        universe = prepare_universe(t, c, dt=1)

        # Load trajectory into the universe and set the unitcell
        unitcell = universe.dimensions
        universe.load_new(trj)
        universe.transfer_to_memory(start=start, stop=stop, step=step)
        if universe.dimensions is None:
            for ts in universe.trajectory:
                ts.dimensions = unitcell

        trj_results = analyze_trajectory(
            universe=universe,
            mol_resname=mol_resname,
            restraints=r,
            **kwargs
        )
        results.append(trj_results)

    return results


def analyze_trajectories_wrapper(args):
    """Helper function for multiprocessing that unpacks arguments."""
    # Unpack the tuple of arguments
    (fn_topol, fn_coords, trj, restraints, mol_resname,
     start, stop, step, kwargs) = args

    return analyze_all_trajectories(
        fn_topol=fn_topol,
        fn_coord=fn_coords,
        fn_trjs=trj,
        restraints=restraints,
        mol_resname=mol_resname,
        start=start,
        stop=stop,
        step=step,
        **kwargs
    )


def extract_defaults(fn):
    """
    Extracts default values from the function's signature.

    Parameters
    ----------
    fn : callable
        Function from which to extract defaults.

    Returns
    -------
    dict
        Dictionary of default parameter values.
    """
    sig = inspect.signature(fn)
    return {
        k: v.default
        for k, v in sig.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def get_all_settings(kwargs: dict) -> dict:

    all_kwargs = {'rdf_kwargs': extract_defaults(compute_rdf),
                  'hbond_kwargs': extract_defaults(compute_hbonds),
                  'restraint_kwargs': extract_defaults(compute_probability_density)}
    # Update with user-provided kwargs
    for key, value in kwargs.items():
        if key in all_kwargs:
            all_kwargs[key].update(value)
        else:
            all_kwargs[key] = value

    return all_kwargs
