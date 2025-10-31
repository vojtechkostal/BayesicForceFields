import MDAnalysis as mda
import multiprocessing as mp
import inspect
import time
import warnings

from pathlib import Path
from .hbonds import compute_all_hbonds, compute_hbonds
from .rdf import compute_all_rdfs, compute_rdf
from .restraints import compute_all_restraints, compute_probability_density

from ..structures import TrainSetInfo, QoI
from ..topology import prepare_universe
from ..io.logs import Logger, print_progress, format_time

warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="MDAnalysis.coordinates.DCD"
)
warnings.filterwarnings(
    "ignore", category=UserWarning, module="MDAnalysis.coordinates.XDR"
)


def analyze_trajectory(
    universe: mda.Universe,
    mol_resname: str,
    restraints: list,
    rdf_kwargs: dict = None,
    hbond_kwargs: dict = None,
    restraint_kwargs: dict = None,
) -> QoI:
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

    return QoI(rdf=rdfs, hb=hbonds, restr=restraints)


def analyze_all_trajectories(
        fn_topol: list[str],
        fn_coord: list[str],
        fn_trj: list[str],
        restraints: list[list],
        mol_resname: str,
        start: int = 0,
        stop: int = -1,
        step: int = 1,
        **kwargs
) -> list[QoI]:
    """Analyze all trajectories in the dataset."""
    results = []
    for t, c, trj, r in zip(fn_topol, fn_coord, fn_trj, restraints):
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


def _wrapper(args):
    """Helper function for multiprocessing that unpacks arguments."""
    return analyze_all_trajectories(**args)


def analyze_trainset(
    trainset_dir: str | Path,
    start: int = 1,
    stop: int = None,
    step: int = 1,
    workers: int = -1,
    progress_stride: int = 10,
    logger: Logger = None,
    **settings
) -> tuple[list, TrainSetInfo]:

    """Analyze the training set to compute quantities of interest (QoI).

    Parameters
    ----------
    trainset_dir : str or Path
        Directory containing the training set data.
    start : int, optional
        Starting frame for trajectory analysis (default is 1).
    stop : int, optional
        Ending frame for trajectory analysis (default is None, meaning the last frame).
    step : int, optional
        Step size for frame selection (default is 1).
    workers : int, optional
        Number of parallel workers to use (default is -1, meaning all available cores).
    progress_stride : int, optional
        Frequency of progress updates (default is every 10 samples).
    logger : Logger, optional
        Logger for reporting progress (default is a new Logger instance).
    settings : dict, optional
        Additional settings for analysis functions
        (e.g., rdf_kwargs, hbond_kwargs, restraint_kwargs).

    Returns
    -------
    tuple
        A tuple containing:
        - qoi : list
            Computed quantities of interest.
        - trainset_info : TrainSetInfo
            Information about the training set.
    """

    logger = logger or Logger('training QoI')

    trainset_info = TrainSetInfo.from_dir(trainset_dir)

    t0 = time.time()

    n_samples = trainset_info.n_samples

    args = [
        {
            'fn_topol': trainset_info.fn_topol,
            'fn_coord': trainset_info.fn_coord,
            'fn_trj': trj_file,  # single trajectory as a list
            'restraints': trainset_info.restraints,
            'mol_resname': trainset_info.specs['mol_resname'],
            'start': start,
            'stop': stop,
            'step': step,
            **settings
        }
        for trj_file in trainset_info.fn_trj
    ]

    workers = mp.cpu_count() if workers == -1 else workers
    if workers > 1:
        with mp.Pool(workers, maxtasksperchild=1) as pool:
            iterator = pool.imap(_wrapper, args)
            qoi = list(
                print_progress(
                    iterator,
                    n_samples,
                    progress_stride,
                    logger
                )
            )
    else:
        iterator = print_progress(
            args,
            n_samples,
            progress_stride,
            logger
        )
        qoi = [_wrapper(arg) for arg in iterator]

    t1 = time.time()
    logger.info(
        f"Training QoI: {n_samples}/{n_samples} "
        f"(100%) | Done. ({format_time(t1 - t0)})",
        level=1
    )

    return qoi, trainset_info


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
