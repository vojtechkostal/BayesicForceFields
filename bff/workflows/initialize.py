import argparse
import subprocess
import numpy as np
import parmed as pmd

import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.selections.gromacs import SelectionWriter
from MDAnalysis import transformations as trans

from pathlib import Path
from collections import defaultdict

from ..topology import create_box
from ..io.cp2k import make_cp2k_input
from ..io.mdp import MDP
from ..io.utils import load_yaml
from ..io.logs import Logger


def check_gmx_available(gmx_cmd='gmx') -> None:
    """Check if the 'gmx' command is available in the system PATH."""
    try:
        subprocess.run(
            [gmx_cmd, '--version'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError(
            f"GROMACS ({gmx_cmd}) is not available.\n"
            "Please make sure that your Gromacs executable is available "
            "from the command line."
        )


def load_config(config: str | Path) -> None:
    """Ensure the config file contains necessary fields with valid types."""

    config = load_yaml(config)

    fn_log = config.get('fn_log')
    config['fn_log'] = fn_log

    required_keys = ['fn_topol', 'fn_mol', 'charge', 'mult',
                     'nsteps_nvt', 'project', 'gmx_cmd']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key in config: {key}")

    # Check if gromacs executable is available
    check_gmx_available(config['gmx_cmd'])

    # Ensure lists have the same length
    list_keys = ['fn_topol', 'fn_mol', 'restraints', 'charge', 'mult']
    length = len(config['fn_topol'])
    for key in list_keys:
        if key == 'restraints':
            if not config.get(key):
                config[key] = [None] * length
        if key == 'fn_mol':
            if isinstance(config[key], str):
                config[key] = [config[key]] * length
        if key in ['charge', 'mult']:
            if isinstance(config[key], int):
                config[key] = [config[key]] * length
        if len(config[key]) != length:
            raise ValueError(
                f"Inconsistent list length for {key}: expected {length}, "
                f"got {len(config[key])}"
            )

    # Ensure `fn_mol` is a list
    if isinstance(config['fn_mol'], str):
        config['fn_mol'] = [config['fn_mol']] * length

    if config.get('nsteps_npt') is None:
        config['nsteps_npt'] = [0] * length
    elif isinstance(config.get('nsteps_npt'), int):
        config['nsteps_npt'] = [config['nsteps_npt']] * length
    else:
        if len(config['nsteps_npt']) != length:
            raise ValueError(
                f"Inconsistent list length for nsteps_npt: expected {length}, "
                f"got {len(config['nsteps_npt'])}"
            )

    # Ensure `box` has the correct structure
    if 'box' not in config or config['box'] is None:
        config['box'] = [None] * length
    elif len(config['box']) != length:
        raise ValueError(
            f"Inconsistent list length for box: expected {length}, "
            f"got {len(config['box'])}"
        )
    else:
        for i, box in enumerate(config['box']):
            if isinstance(box, list) and len(box) in {3, 6}:
                config['box'][i] = box + [90, 90, 90] if len(box) == 3 else box
                config['nsteps_npt'][i] = 0
            else:
                raise ValueError(
                    f"Invalid box dimensions: {box}. "
                    f"Must contain either 3 or 6 values.")

    if config.get('nsteps_npt') is None:
        config['nsteps_npt'] = [0] * length
    elif isinstance(config.get('nsteps_npt'), int):
        config['nsteps_npt'] = [config['nsteps_npt']] * length
    else:
        if len(config['nsteps_npt']) != length:
            raise ValueError(
                f"Inconsistent list length for nsteps_npt: expected {length}, "
                f"got {len(config['nsteps_npt'])}"
            )

    if config.get('nsteps_nvt') is None:
        config['nsteps_nvt'] = [100000] * length
    elif isinstance(config.get('nsteps_nvt'), int):
        config['nsteps_nvt'] = [config['nsteps_nvt']] * length
    else:
        if len(config['nsteps_nvt']) != length:
            raise ValueError(
                f"Inconsistent list length for nsteps_nvt: expected {length}, "
                f"got {len(config['nsteps_nvt'])}"
            )

    # Ensure MDP files are lists of correct length
    mdp_keys = ['fn_mdp_em', 'fn_mdp_npt', 'fn_mdp_nvt']
    for key in mdp_keys:
        if key not in config or config[key] is None:
            config[key] = [get_fn_mdp(key.split('_')[-1] + '.mdp')] * length
        elif isinstance(config[key], str):
            config[key] = [Path(config[key]).resolve()] * length
        elif len(config[key]) != length:
            raise ValueError(
                f"Inconsistent list length for {key}: expected {length}, "
                f"got {len(config[key])}"
            )
        else:
            config[key] = [Path(f).resolve() for f in config[key]]

    return config


def setup_directories(main_dir: Path = Path('./')) -> tuple:
    """Create required directories for the workflow."""
    main_dir = Path('./').resolve()
    prep_dir = main_dir / 'prepare'
    train_dir = main_dir / 'train'
    cp2k_dir = main_dir / 'cp2k'

    for directory in [prep_dir, train_dir, cp2k_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    return prep_dir, train_dir, cp2k_dir


def process_topologies(config: dict):
    """Group restraints by topology file."""
    grouped = defaultdict(lambda: {'sel': [], 'x0': [], 'k': []})
    for fn, sel, x0, k in zip(
        config['fn_topol'], config['restr_sel'], config['restr_x0'], config['restr_k']
    ):
        grouped[fn]['sel'].append(sel)
        grouped[fn]['x0'].append(x0)
        grouped[fn]['k'].append(k)
    return grouped


def make_ndx(universe: mda.Universe, selections: list, fn_out: str) -> None:
    with SelectionWriter(fn_out, mode='w') as ndx:
        ndx.write(universe.atoms, name='System')
        if selections is not None:
            for sel in selections:
                for atom in sel.split():
                    group = universe.select_atoms('name ' + atom)
                    ndx.write(group, name=atom)


def run_md(
    name: str,
    fn_mdp: str,
    fn_topol: str,
    fn_coord: str,
    fn_ndx: str,
    n_steps: int = -2,
    maxwarn: int = 0,
    fn_log: str = 'gmx.log'
) -> None:

    fn_tpr = str(name) + '.tpr'

    grompp_cmd = [
        'gmx', 'grompp', '-f', str(fn_mdp), '-c', str(fn_coord),
        '-p', str(fn_topol), '-o', fn_tpr, '-maxwarn', str(maxwarn)
    ]
    if fn_ndx:
        grompp_cmd.extend(['-n', str(fn_ndx)])

    mdrun_cmd = [
        'gmx', 'mdrun', '-s', fn_tpr, '-deffnm', str(name),
        '-nsteps', str(n_steps)
    ]

    with open(fn_log, "a") as f:
        subprocess.run(grompp_cmd, stdout=f, stderr=f, check=True)
        subprocess.run(mdrun_cmd, stdout=f, stderr=f, check=True)


def insert_pull_code(
    fn_mdp: str,
    groups: list,
    positions: list,
    force_constants: list,
    fn_out: str
) -> MDP:

    mdp = MDP(str(fn_mdp))
    pull = {
        'pull': 'yes',
        'pull-ncoords': len(groups),
        'pull-ngroups': 2 * len(groups)
    }

    for i, (group, init_dist, force_const) in enumerate(
            zip(groups, positions, force_constants), start=1):

        pull[f'pull-group{i*2-1}-name'] = group.split()[0]
        pull[f'pull-group{i*2}-name'] = group.split()[1]
        pull[f'pull-coord{i}-type'] = 'umbrella'
        pull[f'pull-coord{i}-geometry'] = 'distance'
        pull[f'pull-coord{i}-dim'] = 'Y Y Y'
        pull[f'pull-coord{i}-groups'] = f'{i*2-1} {i*2}'
        pull[f'pull-coord{i}-start'] = 'no'
        pull[f'pull-coord{i}-init'] = init_dist
        pull[f'pull-coord{i}-rate'] = 0.0
        pull[f'pull-coord{i}-k'] = force_const

    mdp.content.update(pull)
    mdp.write(fn_out)

    return mdp


def get_average_box(
    universe: mda.Universe,
    start: int = 0,
    stop: int = None,
    step: int = 1
) -> np.ndarray:
    """Compute the average box size over a trajectory."""
    sl = slice(start, stop or universe.trajectory.n_frames, step)

    box = np.zeros((len(universe.trajectory[sl]), 6))
    for i, ts in enumerate(universe.trajectory[sl]):
        box[i] = ts.dimensions
    box_avg = np.round(np.mean(box, axis=0), 4)

    return box_avg


def create_restraint_window(
    universe: mda.Universe,
    restr_sel: list,
    restr_x0: list,
    box: np.ndarray,
    fn_out=None
) -> None:
    restr_x0 = np.array(restr_x0, dtype=float)
    ags = [[universe.select_atoms('name ' + x)
            for x in a.split()] for a in restr_sel]
    distances = np.array([
        [distance_array(ag_pair[0], ag_pair[1], box=ts.dimensions)[0, 0]
         for ag_pair in ags]
        for ts in universe.trajectory
    ])

    # win_names = []
    restr_x0_scaled = np.array(restr_x0) * 10
    name, suffix = str(fn_out).split('.')

    for pos in restr_x0_scaled.T:
        idx = np.argmin(np.sum(np.abs(distances - pos), axis=1))
        # print(i, pos, idx, np.argmin(np.sum(np.abs(distances - pos), axis=1)))
        fn_out_idx = f"{name}.{suffix}"
        # win_names.append(fn_out_idx)
        with mda.Writer(fn_out_idx, 'w') as w:
            ts = universe.trajectory[idx]
            ts.dimensions = box
            w.write(universe.atoms)


def remove_vsites(universe: mda.Universe, fn_out=None) -> mda.AtomGroup:
    """Remove virtual sites from the universe and save the coordinates."""

    # Remove virtual sites and save .xyz file
    atoms = universe.select_atoms('not mass -1 to 0.5')

    if fn_out:
        atoms.write(fn_out, frames=universe.trajectory[[-1]])
    else:
        return atoms


def strip_topol(
        fn_topol: str, fn_coords: str, fn_out_topol: str, fn_out_coords: str) -> None:
    top = pmd.load_file(fn_topol, xyz=fn_coords)

    # Remove dummy atoms
    mask = np.array([atom.element for atom in top.atoms]) == 0
    top.strip(mask)
    top.write(fn_out_topol)
    top.save(fn_out_coords)


def get_restraint_atom_indices(fn_system: str, names: list) -> np.ndarray:
    u = mda.Universe(fn_system)
    indices = []
    for name in names:
        if name:
            a1, a2 = name.split()
            atoms = u.select_atoms(f'name {a1} {a2}')
            indices.append(atoms.indices + 1)
    return np.array(indices)


def get_fn_mdp(fn: str) -> Path:
    fn_mdp = Path(__file__).parent.parent / 'data' / 'mdp' / fn
    return fn_mdp.resolve()


def main(fn_config: str) -> None:
    config = load_config(fn_config)
    logger = Logger("initialize", config['fn_log'])
    logger.info("", level=0)
    logger.info(f"=== Initializing project: {config['project']} ===\n", level=0)

    prep_dir, train_dir, cp2k_dir = setup_directories()

    # Storage for unique topologies -> cached data
    done_topologies: dict[str, dict] = {fn: None for fn in set(config['fn_topol'])}

    n_total = len(config['fn_topol'])
    for i, (fn_topol, fn_mol, charge, mult, restraint, box) in enumerate(
        zip(config['fn_topol'], config['fn_mol'], config['charge'],
            config['mult'], config['restraints'], config['box'])
    ):
        logger.info(f"System: {i + 1}/{n_total}", level=1)

        # --- EM + NpT equilibration (only once per unique topology) ---
        if done_topologies[fn_topol] is None:
            j = config['fn_topol'].index(fn_topol)
            fn_coord = prep_dir / f'topol-{j:02d}.gro'

            logger.info("Creating box: in progress...", overwrite=True, level=2)
            u, topol = create_box(fn_topol, fn_mol, fn_out=fn_coord, box=box)
            logger.info("Creating box: Done.", level=2)

            fn_topol_processed = fn_coord.with_suffix('.top')
            topol.save(str(fn_topol_processed))
            q = sum(atom.charge for atom in topol.atoms)
            maxwarn = 1 if not np.isclose(q, 0, atol=1e-4) else 0

            # Energy minimization
            deffnm_em = prep_dir / f'0-em-{j:02d}'
            fn_mdp_em = config['fn_mdp_em'][i]
            logger.info("Energy minimization: in progress...", overwrite=True, level=2)
            run_md(deffnm_em, fn_mdp_em, fn_topol_processed, fn_coord,
                   fn_ndx=None, n_steps=-2, maxwarn=maxwarn)
            logger.info("Energy minimization: Done.", level=2)
            (train_dir / 'em.mdp').write_text(fn_mdp_em.read_text())

            # NpT equilibration
            deffnm_npt = prep_dir / f'1-npt-{j:02d}'
            n_steps = config['nsteps_npt'][i]
            if n_steps > 0:
                fn_mdp_npt = config['fn_mdp_npt'][i]
                logger.info(
                    "NpT equilibration: in progress...", overwrite=True, level=2)
                run_md(
                    deffnm_npt, fn_mdp_npt, fn_topol_processed,
                    deffnm_em.with_suffix('.gro'),
                    fn_ndx=None, n_steps=n_steps, maxwarn=maxwarn)
                u = mda.Universe(deffnm_npt.with_suffix('.tpr'),
                                 deffnm_npt.with_suffix('.xtc'),
                                 to_guess=('elements', 'masses'))
                u.trajectory.add_transformations(trans.unwrap(u.atoms))
                discard = int(u.trajectory.n_frames * 0.2)
                box_avg = get_average_box(u, start=discard)
                logger.info("NpT equilibration: Done.", level=2)
                fn_coord = deffnm_npt.with_suffix('.gro')
            else:
                logger.info("NpT equilibration: skipped (box defined)", level=2)
                box_avg = np.array(box)
                u = mda.Universe(
                    deffnm_em.with_suffix('.tpr'),
                    deffnm_em.with_suffix('.gro'),
                    to_guess=('elements', 'masses')
                )
                fn_coord = deffnm_em.with_suffix('.gro')

            done_topologies[fn_topol] = {
                'fn_topol_processed': fn_topol_processed,
                'fn_coord': fn_coord,
                'universe': u,
                'box': box_avg,
                'maxwarn': maxwarn,
            }

        # --- NVT equilibration (for all systems) ---
        topol_data = done_topologies[fn_topol]
        fn_topol_processed = topol_data['fn_topol_processed']
        u = topol_data['universe']
        box = topol_data['box']
        maxwarn = topol_data['maxwarn']

        fn_mdp_nvt = config['fn_mdp_nvt'][i]
        fn_nvt_local = prep_dir / f'nvt-{i:03d}.mdp'
        fn_ndx = prep_dir / f'index-{i:03d}.ndx'
        fn_coord = prep_dir / f'system-{i:03d}.gro'
        fn_topol_local = fn_topol_processed.with_name(f'system-{i:02d}.top')
        fn_topol_local.write_text(fn_topol_processed.read_text())

        if restraint:
            atoms, x0, k = restraint['atoms'], restraint['x0'], restraint['k']
            create_restraint_window(u, atoms, x0, box, fn_coord)
            insert_pull_code(fn_mdp_nvt, atoms, x0, k, fn_nvt_local)
            make_ndx(u, atoms, fn_out=fn_ndx)
        else:
            atoms, x0, k = [], [], []
            make_ndx(u, None, fn_out=fn_ndx)
            with mda.Writer(fn_coord, 'w') as w:
                ts = u.trajectory[-1]
                ts.dimensions = box
                w.write(u.atoms)
            fn_nvt_local.write_text(fn_mdp_nvt.read_text())

        deffnm_nvt = prep_dir / f'2-nvt-{i:03d}'
        logger.info("NVT equilibration: in progress...", overwrite=True, level=2)
        run_md(deffnm_nvt, fn_mdp_nvt, fn_topol_local, fn_coord,
               fn_ndx, n_steps=config['nsteps_nvt'][i], maxwarn=maxwarn)
        logger.info("NVT equilibration: Done.", level=2)

        # Save training artifacts
        for src, dst in [
            (fn_nvt_local, train_dir / f'nvt-{i:03d}.mdp'),
            (fn_coord, train_dir / f'system-{i:03d}.gro'),
            (fn_ndx, train_dir / fn_ndx.name),
            (fn_topol_local, train_dir / f'topol-{i:03d}.top'),
        ]:
            dst.write_text(src.read_text())

        # --- CP2K input preparation ---
        logger.info("CP2k input: in progress...", overwrite=True, level=2)
        cp2k_win_dir = cp2k_dir / f'win-{i:03d}'
        cp2k_win_dir.mkdir(exist_ok=True)

        remove_vsites(u, cp2k_win_dir / 'pos.xyz')
        strip_topol(str(fn_topol_processed), str(deffnm_nvt) + '.gro',
                    str(cp2k_win_dir / 'topol.top'), str(cp2k_win_dir / 'pos.gro'))

        atom_indices = get_restraint_atom_indices(
            str(cp2k_win_dir / 'pos.gro'), atoms)
        restraint_info = [
            {'atoms': ' '.join(map(str, idx)), 'target': x0_, 'k': k_ / 2}
            for idx, x0_, k_ in zip(atom_indices, x0, k)
        ]

        for specs in [
            {'eq': True, 'restart': False, 'fn': 'md-eq-start.inp'},
            {'eq': True, 'restart': True,  'fn': 'md-eq-restart.inp'},
            {'eq': False, 'restart': True, 'fn': 'md-prod.inp'},
        ]:
            make_cp2k_input(config['project'], charge, mult,
                            box[:3].astype(str), cp2k_win_dir / 'pos.xyz',
                            restraint_info, specs['eq'], specs['restart'],
                            cp2k_win_dir / specs['fn'])

        logger.info("CP2k input: Done.", level=2)
        logger.info("", level=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create the system.')
    parser.add_argument('fn_config', help='Path to the config file [YAML].')
    args = parser.parse_args()
    main(args.fn_config)
