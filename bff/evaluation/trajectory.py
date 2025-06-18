import MDAnalysis as mda

from .hbonds import compute_all_hbonds
from .rdf import compute_all_rdfs
from .restraints import compute_all_restraints
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


# def score_sample(
#     trj_sample: list[TrajectoryData],
#     trj_reference: list[TrajectoryData],
#     components: list
# ) -> list:
#     """Evaluate the trajectory data."""
#     results = []
#     for c in components:
#         analyzer = ANALYZERS[c]
#         for r, s in zip(trj_reference, trj_sample):
#             features_true = getattr(r, c)
#             features_pred = getattr(s, c)
#             results.extend(analyzer(features_true, features_pred))
#     return results


# def feature_splits(results: list, components: list) -> tuple:
#     """Splits the reference output into segments
#     according to the results' components."""
#     i = 0
#     output_reference = []
#     output_segments = {}
#     for c in components:

#         # Calculte length of the expected output
#         n = sum(len(getattr(ref, c)) for ref in results)
#         output_segments[c] = [i, i + n]
#         i += n

#         # Update the reference output
#         if c in ["rdf", "restr"]:
#             out = np.zeros(n)
#         elif c in ["rdf_fpp", "rdf_fph", "hb"]:
#             out = [v for ref in results for v in getattr(ref, c).values()]
#         output_reference.append(out)

#     return output_segments, np.concatenate(output_reference)
