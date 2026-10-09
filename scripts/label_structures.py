#!/usr/bin/env python3
"""Label trajectory snapshots with CP2K single points on Slurm.

Selects frames from one or more trajectories, runs one CP2K energy-and-force
calculation per frame (plus one per isolated element) as a Slurm job array,
and collects the results as EXTXYZ files for training or fine-tuning a
machine-learned interatomic potential.

This script is distributed with BFF but does not import it. It needs Python
3.10+, PyYAML, NumPy, and MDAnalysis (or ASE for ``.traj`` input).

Subcommands, all reading the same CONFIG:

    run     CONFIG             stage every task, submit the Slurm array(s),
                               wait for them, and collect the results
    prepare CONFIG             only stage the tasks (inspect before running)
    collect CONFIG             (re-)assemble the datasets from finished tasks
    task    CONFIG INDEX       run one task; called by every array element

``run`` is safe to repeat: tasks whose structure, cell, and CP2K input are
unchanged since they last completed are skipped.

Configuration (paths are relative to the config file)::

    output_dir: ./labels           # default ./labels
    cp2k_cmd: cp2k.psmp            # may include arguments
    cell_labels: none              # or stress_and_virials (NPT training)
    seed: 2026                     # frame sampling and dataset split
    split: {train: 0.8, valid: 0.1, test: 0.1}
    systems:
      - id: water                  # [A-Za-z0-9_.-]+, unique
        topology: system.gro       # omit for an ASE .traj trajectory
        trajectory: md.xtc
        atom_selection: all        # MDAnalysis selection
        elements: {CAL: Ca}        # optional element per atom name, else
                                   # taken from the topology or guessed
        first_frame: 0             # optional
        stop_frame: null           # optional, exclusive
        n_snapshots: 100           # or stride: N, or n_random: N
        periodic: true             # require cell.inc in the CP2K input
        cp2k_input: sp.inp         # must use structure.xyz (and cell.inc)
    single_atom_inputs:            # one CP2K input per element present
      H: atom-h.inp
      O: atom-o.inp
    slurm:
      max_parallel_jobs: 50        # running tasks per array
      max_array_size: 1000         # tasks per array (Slurm MaxArraySize)
      sbatch: {time: "04:00:00", ntasks: 8}
      setup: [module load cp2k]    # shell lines before CP2K runs

Outputs in ``output_dir``: ``train.extxyz``, ``valid.extxyz``,
``test.extxyz`` (energy in eV, forces in eV/angstrom, optional stress in
eV/angstrom^3 and virials in eV; ``config_type`` is the system id),
``energies.yaml`` (isolated-atom energies in eV keyed by atomic number), and
``label-results.yaml`` (provenance, split membership, and failures).
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903
FORCE_TO_EV_ANGSTROM = HARTREE_TO_EV / BOHR_TO_ANGSTROM
GPA_TO_EV_ANGSTROM3 = 0.006241509074460763
FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
ENERGY_RE = re.compile(rf"ENERGY\| Total FORCE_EVAL .*?({FLOAT})\s*$", re.MULTILINE)
PIPE_FORCE_RE = re.compile(
    rf"^\s*FORCES\|\s+\d+\s+({FLOAT})\s+({FLOAT})\s+({FLOAT})\s+{FLOAT}\s*$",
    re.MULTILINE,
)
LEGACY_FORCE_RE = re.compile(
    rf"^\s*\d+\s+\d+\s+\S+\s+({FLOAT})\s+({FLOAT})\s+({FLOAT})\s*$"
)
STRESS_ROW_RE = re.compile(
    rf"^\s*STRESS\|\s*[xyzXYZ]\s+({FLOAT})\s+({FLOAT})\s+({FLOAT})\s*$",
    re.MULTILINE,
)
ELEMENTS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni "
    "Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I "
    "Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt "
    "Au Hg Tl Pb Bi Po At Rn"
).split()
ATOMIC_NUMBERS = {symbol: number for number, symbol in enumerate(ELEMENTS, start=1)}
CONFIG_KEYS = {
    "output_dir", "cp2k_cmd", "cell_labels", "seed", "split", "systems",
    "single_atom_inputs", "slurm",
}
SYSTEM_KEYS = {
    "id", "topology", "trajectory", "atom_selection", "first_frame",
    "stop_frame", "n_snapshots", "stride", "n_random", "periodic", "cp2k_input",
    "elements",
}
SLURM_KEYS = {"max_parallel_jobs", "max_array_size", "sbatch", "setup"}
ISOLATED_ATOM_CELL = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]


def resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def existing_file(base: Path, value, where: str) -> Path:
    path = resolve(base, str(value))
    if not path.is_file():
        raise FileNotFoundError(f"{where}: file not found: {path}")
    return path


def check_keys(raw: dict, allowed: set[str], where: str) -> None:
    unknown = set(raw) - allowed
    if unknown:
        keys = ", ".join(sorted(unknown))
        raise ValueError(f"{where} has unsupported key(s): {keys}")


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text()) or {}
    if not isinstance(cfg, dict):
        raise TypeError(f"{path} must contain a mapping, not a task list")
    check_keys(cfg, CONFIG_KEYS, str(path))
    systems = cfg.get("systems")
    if not isinstance(systems, list) or not systems:
        raise ValueError(f"{path}: systems must be a non-empty list")
    for index, system in enumerate(systems):
        check_keys(system, SYSTEM_KEYS, f"{path}: systems[{index}]")
    slurm = cfg.get("slurm")
    if not isinstance(slurm, dict) or not isinstance(slurm.get("sbatch"), dict):
        raise ValueError(f"{path}: slurm.sbatch must be a mapping of sbatch options")
    check_keys(slurm, SLURM_KEYS, f"{path}: slurm")
    if "array" in slurm["sbatch"]:
        raise ValueError(f"{path}: slurm.sbatch.array is set by this script")
    if cfg.get("cell_labels", "none") not in {"none", "stress_and_virials"}:
        raise ValueError(f"{path}: cell_labels must be none or stress_and_virials")
    split = cfg.get("split", {"train": 0.8, "valid": 0.1, "test": 0.1})
    if not isinstance(split, dict):
        raise ValueError(f"{path}: split must be a mapping")
    check_keys(split, {"train", "valid", "test"}, f"{path}: split")
    split = {key: float(split.get(key, 0.0)) for key in ("train", "valid", "test")}
    if not math.isclose(sum(split.values()), 1.0):
        raise ValueError(f"{path}: split.train + split.valid + split.test must be 1")

    base = path.parent
    cfg["split"] = split
    cfg["_base"] = base
    cfg["output_dir"] = resolve(base, str(cfg.get("output_dir", "./labels")))
    return cfg


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_frame_indices(raw: dict, n_frames: int, seed: int) -> list[int]:
    """Frames of one trajectory to label, set by exactly one selection key.

    ``n_snapshots`` picks evenly spaced frames, ``stride`` every Nth frame,
    and ``n_random`` a seeded random subset, all within
    ``[first_frame, stop_frame)``.
    """
    first = int(raw.get("first_frame", 0))
    stop = n_frames if raw.get("stop_frame") is None else int(raw["stop_frame"])
    stop = min(stop, n_frames)
    modes = [key for key in ("n_snapshots", "stride", "n_random") if key in raw]
    if len(modes) != 1:
        raise ValueError(
            f"system {raw['id']!r}: set exactly one of n_snapshots, stride, "
            f"n_random (got {modes or 'none'})"
        )
    mode, value = modes[0], int(raw[modes[0]])
    available = stop - first
    if value < 1:
        raise ValueError(f"system {raw['id']!r}: {mode} must be positive")
    if mode == "stride":
        indices = list(range(first, stop, value))
    elif value > available:
        raise ValueError(
            f"system {raw['id']!r}: cannot choose {value} frames from "
            f"[{first}, {stop})"
        )
    elif mode == "n_random":
        indices = sorted(random.Random(seed).sample(range(first, stop), value))
    elif value == 1:
        indices = [first + (available - 1) // 2]
    else:
        indices = [
            first + round(i * (available - 1) / (value - 1)) for i in range(value)
        ]
    if not indices:
        raise ValueError(f"system {raw['id']!r}: no frames in [{first}, {stop})")
    return indices


def cell_vectors(dimensions) -> list[list[float]]:
    """Cell vectors (angstrom) from ``a, b, c, alpha, beta, gamma``."""
    a, b, c, alpha, beta, gamma = (float(value) for value in dimensions[:6])
    alpha, beta, gamma = map(math.radians, (alpha, beta, gamma))
    sin_gamma = math.sin(gamma)
    if a <= 0 or b <= 0 or c <= 0 or abs(sin_gamma) < 1e-12:
        raise ValueError(f"degenerate unit cell {list(dimensions[:6])}")
    cx = c * math.cos(beta)
    cy = c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / sin_gamma
    vectors = [
        [a, 0.0, 0.0],
        [b * math.cos(gamma), b * sin_gamma, 0.0],
        [cx, cy, math.sqrt(max(c * c - cx * cx - cy * cy, 0.0))],
    ]
    return [[round(value, 4) for value in row] for row in vectors]


def open_trajectory(
    topology: Path | None,
    trajectory: Path,
    atom_selection: str,
    elements: dict[str, str],
):
    """Return ``(n_frames, get_frame)``; ``get_frame(i)`` gives
    ``(symbols, positions, cell)``.

    ASE ``.traj`` files are read with ASE and always use every atom; anything
    else is read with MDAnalysis and requires a topology. ``elements`` maps
    atom names to element symbols where the topology has none or the guess
    from the name is wrong (for example ``CAL`` for calcium).
    """
    if trajectory.suffix == ".traj":
        if atom_selection != "all" or elements:
            raise ValueError(
                f"{trajectory}: atom_selection and elements require MDAnalysis input"
            )
        from ase.io.trajectory import Trajectory

        frames = Trajectory(str(trajectory))

        def get_ase_frame(index: int):
            atoms = frames[index]
            if atoms.cell.rank != 3:
                raise ValueError(f"frame {index} of {trajectory} has no 3D cell")
            symbols = [symbol.capitalize() for symbol in atoms.get_chemical_symbols()]
            return symbols, atoms.get_positions(), cell_vectors(atoms.cell.cellpar())

        return len(frames), get_ase_frame

    import MDAnalysis as mda

    if topology is None:
        raise ValueError(f"{trajectory}: topology is required for non-.traj input")
    universe = mda.Universe(str(topology), str(trajectory))
    if not hasattr(universe.atoms, "elements"):
        universe.guess_TopologyAttrs(to_guess=["elements"])
    atoms = universe.select_atoms(atom_selection)
    if len(atoms) == 0:
        raise ValueError(f"{trajectory}: atom_selection {atom_selection!r} is empty")
    symbols = [
        str(elements.get(name, element)).capitalize()
        for name, element in zip(atoms.names, atoms.elements, strict=True)
    ]

    def get_mda_frame(index: int):
        timestep = universe.trajectory[index]
        if timestep.dimensions is None:
            raise ValueError(f"frame {index} of {trajectory} has no unit cell")
        return symbols, atoms.positions, cell_vectors(timestep.dimensions)

    return universe.trajectory.n_frames, get_mda_frame


def write_structure(path: Path, symbols, positions, cell) -> None:
    lattice = " ".join(f"{value:.12g}" for row in cell for value in row)
    lines = [
        str(len(symbols)),
        f'Lattice="{lattice}" Properties=species:S:1:pos:R:3 pbc="T T T"',
    ]
    for symbol, xyz in zip(symbols, positions, strict=True):
        lines.append(f"{symbol} " + " ".join(f"{float(v):.12g}" for v in xyz))
    path.write_text("\n".join(lines) + "\n")


def write_cell(path: Path, cell) -> None:
    lines = [
        f"{key} " + " ".join(f"{value:.12g}" for value in vector)
        for key, vector in zip("ABC", cell, strict=True)
    ]
    path.write_text("\n".join(lines) + "\n")


def stage(run_dir: Path, symbols, positions, cell, cp2k_input: Path) -> str:
    """Write one task's inputs and return its content signature."""
    run_dir.mkdir(parents=True, exist_ok=True)
    write_structure(run_dir / "structure.xyz", symbols, positions, cell)
    write_cell(run_dir / "cell.inc", cell)
    shutil.copy2(cp2k_input, run_dir / "input.inp")
    return ":".join(
        sha256(run_dir / name) for name in ("structure.xyz", "cell.inc", "input.inp")
    )


def prepare(config_path: Path) -> list[dict]:
    """Stage every snapshot and isolated atom and write ``tasks.yaml``."""
    cfg = load_config(config_path)
    output: Path = cfg["output_dir"]
    seed = int(cfg.get("seed", 2026))
    cell_labels = str(cfg.get("cell_labels", "none"))
    tasks: list[dict] = []
    systems: dict[str, dict] = {}
    elements: set[str] = set()
    for raw in cfg["systems"]:
        system_id = str(raw.get("id"))
        if system_id in systems or not re.fullmatch(r"[A-Za-z0-9_.-]+", system_id):
            raise ValueError(f"invalid or duplicate system id {system_id!r}")
        base, where = cfg["_base"], f"system {system_id!r}"
        topology = (
            existing_file(base, raw["topology"], f"{where} topology")
            if "topology" in raw
            else None
        )
        trajectory = existing_file(base, raw["trajectory"], f"{where} trajectory")
        cp2k_input = existing_file(base, raw["cp2k_input"], f"{where} cp2k_input")
        text = cp2k_input.read_text()
        if "structure.xyz" not in text:
            raise ValueError(f"{system_id!r}: {cp2k_input} must read structure.xyz")
        if raw.get("periodic", True) and "cell.inc" not in text:
            raise ValueError(f"{system_id!r}: {cp2k_input} must include cell.inc")

        n_frames, get_frame = open_trajectory(
            topology,
            trajectory,
            str(raw.get("atom_selection", "all")),
            {str(name): str(value) for name, value in raw.get("elements", {}).items()},
        )
        indices = select_frame_indices(raw, n_frames, seed)
        for ordinal, frame in enumerate(indices):
            run_dir = output / "systems" / system_id / f"snapshot-{ordinal:05d}"
            symbols, positions, cell = get_frame(frame)
            unknown = set(symbols) - set(ATOMIC_NUMBERS)
            if unknown:
                raise ValueError(
                    f"system {system_id!r}: unknown element(s) "
                    + ", ".join(sorted(unknown))
                    + "; map atom names to elements with `elements`"
                )
            elements.update(symbols)
            signature = stage(run_dir, symbols, positions, cell, cp2k_input)
            tasks.append({
                "kind": "snapshot",
                "system": system_id,
                "frame": frame,
                "run_dir": str(run_dir),
                "cell": cell,
                "signature": f"{signature}:{cell_labels}",
            })
        systems[system_id] = {
            "topology": str(topology) if topology else None,
            "trajectory": str(trajectory),
            "topology_sha256": sha256(topology) if topology else None,
            "trajectory_sha256": sha256(trajectory),
            "cp2k_input_sha256": sha256(cp2k_input),
            "frame_indices": indices,
        }

    atom_inputs = {
        str(key).capitalize(): value
        for key, value in (cfg.get("single_atom_inputs") or {}).items()
    }
    missing = elements - set(atom_inputs)
    if missing:
        raise ValueError(
            "single_atom_inputs is missing CP2K inputs for: "
            + ", ".join(sorted(missing))
        )
    for element in sorted(elements):
        cp2k_input = existing_file(
            cfg["_base"], atom_inputs[element], f"single_atom_inputs.{element}"
        )
        if "structure.xyz" not in cp2k_input.read_text():
            raise ValueError(f"{cp2k_input} must read structure.xyz")
        run_dir = output / "single-atoms" / element.lower()
        signature = stage(
            run_dir, [element], [[5.0, 5.0, 5.0]], ISOLATED_ATOM_CELL, cp2k_input
        )
        tasks.append({
            "kind": "atom",
            "element": element,
            "run_dir": str(run_dir),
            "signature": signature,
        })

    (output / "tasks.yaml").write_text(yaml.safe_dump(tasks, sort_keys=False))
    manifest = {
        "complete": False,
        "systems": systems,
        "units": {
            "energy": "eV",
            "forces": "eV/angstrom",
            "stress": "eV/angstrom^3",
            "virials": "eV",
        },
    }
    manifest_text = yaml.safe_dump(manifest, sort_keys=False)
    (output / "label-results.yaml").write_text(manifest_text)
    print(f"Prepared {len(tasks)} tasks in {output}")
    return tasks


def to_float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def read_energy(text: str) -> float:
    matches = ENERGY_RE.findall(text)
    if not matches:
        raise ValueError("CP2K energy was not found")
    return to_float(matches[-1]) * HARTREE_TO_EV


def read_forces(text: str) -> list[list[float]]:
    """Last force table; supports the ``FORCES|`` and older table formats."""
    rows = PIPE_FORCE_RE.findall(text)
    if not rows:
        tables: list[list[tuple[str, str, str]]] = []
        table: list[tuple[str, str, str]] | None = None
        for line in text.splitlines():
            if re.match(r"^\s*ATOMIC FORCES\b", line):
                table = []
            elif table is not None and re.match(r"^\s*SUM OF ATOMIC FORCES\b", line):
                tables.append(table)
                table = None
            elif table is not None and (match := LEGACY_FORCE_RE.match(line)):
                table.append(match.group(1, 2, 3))
        rows = tables[-1] if tables else []
    if not rows:
        raise ValueError("CP2K forces were not found")
    return [[to_float(value) * FORCE_TO_EV_ANGSTROM for value in row] for row in rows]


def read_stress(text: str) -> list[list[float]]:
    """Last stress tensor, converted to tension-positive eV/angstrom^3."""
    rows = STRESS_ROW_RE.findall(text)
    if len(rows) < 3:
        raise ValueError("CP2K stress tensor was not found")
    if re.search(r"STRESS.*\[GPa\]", text, re.IGNORECASE):
        factor = GPA_TO_EV_ANGSTROM3
    elif re.search(r"STRESS.*\[bar\]", text, re.IGNORECASE):
        factor = GPA_TO_EV_ANGSTROM3 / 1e4
    else:
        raise ValueError("CP2K stress unit was not recognized; print it in GPa or bar")
    # CP2K prints pressure-positive stress; ASE and MACE use tension-positive.
    return [[-to_float(value) * factor for value in row] for row in rows[-3:]]


def read_structure(path: Path) -> tuple[list[str], list[list[float]]]:
    lines = path.read_text().splitlines()
    symbols: list[str] = []
    positions: list[list[float]] = []
    for line in lines[2:2 + int(lines[0])]:
        symbol, *xyz = line.split()
        symbols.append(symbol)
        positions.append([float(value) for value in xyz])
    return symbols, positions


def volume(cell: list[list[float]]) -> float:
    a, b, c = cell
    return abs(
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def write_label(task: dict, energy: float, forces, stress) -> None:
    run_dir = Path(task["run_dir"])
    cell = task["cell"]
    symbols, positions = read_structure(run_dir / "structure.xyz")
    if len(symbols) != len(forces):
        raise ValueError(
            f"{run_dir}: expected {len(symbols)} forces, got {len(forces)}"
        )

    def flat(matrix) -> str:
        return " ".join(
            f"{0.0 if abs(value) < 1e-15 else value:.16g}"
            for row in matrix
            for value in row
        )

    header = [
        f'Lattice="{flat(cell)}"',
        'pbc="T T T"',
        "Properties=species:S:1:pos:R:3:forces:R:3",
        f"energy={energy:.16g}",
        f"config_type={task['system']}",
    ]
    if stress is not None:
        virials = [[-volume(cell) * value for value in row] for row in stress]
        header += [f'stress="{flat(stress)}"', f'virials="{flat(virials)}"']
    lines = [str(len(symbols)), " ".join(header)]
    for symbol, xyz, force in zip(symbols, positions, forces, strict=True):
        lines.append(f"{symbol} " + " ".join(f"{v:.12g}" for v in [*xyz, *force]))
    (run_dir / "label.extxyz").write_text("\n".join(lines) + "\n")


def result_path(task: dict) -> Path:
    name = "label.extxyz" if task["kind"] == "snapshot" else "energy.yaml"
    return Path(task["run_dir"]) / name


def is_done(task: dict) -> bool:
    signature = Path(task["run_dir"]) / ".task-signature"
    return (
        result_path(task).exists()
        and signature.exists()
        and signature.read_text().strip() == task["signature"]
    )


def read_tasks(cfg: dict) -> list[dict]:
    path = cfg["output_dir"] / "tasks.yaml"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `prepare` or `run` first")
    return yaml.safe_load(path.read_text())


def run_task(config_path: Path, index: int) -> None:
    """Run CP2K for one task and write its label unless it is already done."""
    cfg = load_config(config_path)
    task = read_tasks(cfg)[index]
    if is_done(task):
        print(f"Task {index} is up to date: {task['run_dir']}")
        return
    run_dir = Path(task["run_dir"])
    result_path(task).unlink(missing_ok=True)
    command = [
        *shlex.split(str(cfg.get("cp2k_cmd", "cp2k.psmp"))),
        "-i", "input.inp",
        "-o", "output.out",
    ]
    if os.environ.get("SLURM_JOB_ID") and shutil.which("srun"):
        command = ["srun", *command]
    with (run_dir / "driver.log").open("w") as log:
        status = subprocess.run(
            command, cwd=run_dir, stdout=log, stderr=subprocess.STDOUT, check=False
        ).returncode
    if status != 0:
        raise RuntimeError(
            f"CP2K exited with status {status} in {run_dir}; see driver.log and "
            "output.out there"
        )
    text = (run_dir / "output.out").read_text(errors="ignore")
    energy = read_energy(text)
    if task["kind"] == "snapshot":
        stress = (
            read_stress(text)
            if cfg.get("cell_labels", "none") == "stress_and_virials"
            else None
        )
        write_label(task, energy, read_forces(text), stress)
    else:
        result_path(task).write_text(
            yaml.safe_dump({"element": task["element"], "energy": energy})
        )
    (run_dir / ".task-signature").write_text(task["signature"] + "\n")


def write_array_script(cfg: dict, config_path: Path, pending: list[int]) -> Path:
    """Write the array script; element ``i`` runs task ``pending[offset + i]``."""
    output: Path = cfg["output_dir"]
    slurm = cfg["slurm"]
    sbatch = {"output": output / "slurm" / "%A_%a.out"} | slurm["sbatch"]
    (output / "slurm").mkdir(exist_ok=True)
    fn_pending = output / "pending-tasks.txt"
    fn_pending.write_text("".join(f"{index}\n" for index in pending))
    lines = [
        "#!/usr/bin/env bash",
        *(
            f"#SBATCH --{str(key).replace('_', '-')}={value}"
            for key, value in sbatch.items()
        ),
        "",
        # No `set -u`: setup lines often source environments that are not
        # nounset-safe.
        "set -eo pipefail",
        *(str(line) for line in slurm.get("setup", [])),
        "LINE=$((SLURM_ARRAY_TASK_ID + TASK_OFFSET + 1))",
        f'TASK_INDEX=$(sed -n "${{LINE}}p" {shlex.quote(str(fn_pending))})',
        " ".join([
            shlex.quote(sys.executable),
            shlex.quote(str(Path(__file__).resolve())),
            "task",
            shlex.quote(str(config_path)),
            '"$TASK_INDEX"',
        ]),
    ]
    path = output / "label-array.sbatch"
    path.write_text("\n".join(lines) + "\n")
    return path


def sbatch_commands(n_tasks: int, slurm: dict, script: Path) -> list[list[str]]:
    """One blocking ``sbatch`` call per array of at most ``max_array_size``."""
    max_parallel = int(slurm.get("max_parallel_jobs", 100))
    max_size = int(slurm.get("max_array_size", 1000))
    if max_parallel < 1 or max_size < 1:
        raise ValueError(
            "slurm.max_parallel_jobs and slurm.max_array_size must be positive"
        )
    commands = []
    for offset in range(0, n_tasks, max_size):
        size = min(max_size, n_tasks - offset)
        commands.append([
            "sbatch", "--wait", "--parsable",
            f"--array=0-{size - 1}%{max_parallel}",
            f"--export=ALL,TASK_OFFSET={offset}",
            str(script),
        ])
    return commands


def run_all(config_path: Path) -> None:
    """Stage, submit the unfinished tasks, wait, and collect."""
    tasks = prepare(config_path)
    cfg = load_config(config_path)
    pending = [index for index, task in enumerate(tasks) if not is_done(task)]
    if pending:
        print(f"Submitting {len(pending)} of {len(tasks)} tasks")
        script = write_array_script(cfg, config_path, pending)
        # Arrays run concurrently; each sbatch --wait returns when its array ends.
        processes = [
            subprocess.Popen(command)
            for command in sbatch_commands(len(pending), cfg["slurm"], script)
        ]
        if any([process.wait() != 0 for process in processes]):
            print(
                "Some Slurm tasks failed; collecting the finished ones.",
                file=sys.stderr,
            )
    collect(config_path)


def allocate_splits(count: int, fractions: dict[str, float]) -> dict[str, int]:
    """Split sizes by largest remainder, so they add up to ``count``."""
    raw = {key: count * fraction for key, fraction in fractions.items()}
    sizes = {key: math.floor(value) for key, value in raw.items()}
    by_remainder = sorted(raw, key=lambda key: raw[key] - sizes[key], reverse=True)
    for key in by_remainder[:count - sum(sizes.values())]:
        sizes[key] += 1
    return sizes


def collect(config_path: Path) -> None:
    """Write the train/valid/test sets and isolated-atom energies."""
    cfg = load_config(config_path)
    output: Path = cfg["output_dir"]
    tasks = read_tasks(cfg)
    snapshots: list[int] = []
    energies: dict[int, float] = {}
    failures: list[dict] = []
    for index, task in enumerate(tasks):
        if not is_done(task):
            failures.append(
                {"task": index, "kind": task["kind"], "run_dir": task["run_dir"]}
            )
        elif task["kind"] == "snapshot":
            snapshots.append(index)
        else:
            data = yaml.safe_load(result_path(task).read_text())
            energies[ATOMIC_NUMBERS[data["element"]]] = float(data["energy"])

    random.Random(int(cfg.get("seed", 2026))).shuffle(snapshots)
    membership: dict[str, list[int]] = {}
    offset = 0
    for split, size in allocate_splits(len(snapshots), cfg["split"]).items():
        membership[split] = sorted(snapshots[offset:offset + size])
        offset += size
        (output / f"{split}.extxyz").write_text("".join(
            result_path(tasks[index]).read_text() for index in membership[split]
        ))

    atoms_complete = not any(item["kind"] == "atom" for item in failures)
    # Incomplete isolated-atom energies must not be mistaken for usable E0s.
    energy_file = output / "energies.yaml"
    stale_file = output / "energies.partial.yaml"
    if not atoms_complete:
        energy_file, stale_file = stale_file, energy_file
    stale_file.unlink(missing_ok=True)
    energy_file.write_text(yaml.safe_dump(energies, sort_keys=True))

    manifest_path = output / "label-results.yaml"
    manifest = {}
    if manifest_path.exists():
        manifest = yaml.safe_load(manifest_path.read_text())
    manifest.update({
        "complete": not failures,
        "successful_snapshots": len(snapshots),
        "split_membership": membership,
        "single_atom_energies": energies,
        "single_atom_energy_file": energy_file.name,
        "failures": failures,
    })
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    print(f"Collected {len(snapshots)} snapshots; {len(failures)} tasks incomplete.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("run", "stage, submit to Slurm, wait, and collect"),
        ("prepare", "only stage the tasks and write tasks.yaml"),
        ("collect", "assemble datasets from finished tasks"),
        ("task", "run one staged task (called by the Slurm array)"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("config", type=Path, help="labeling config YAML")
        if name == "task":
            command.add_argument("index", type=int, help="0-based index in tasks.yaml")

    args = parser.parse_args(argv)
    config = args.config.expanduser().resolve()
    if args.command == "run":
        run_all(config)
    elif args.command == "prepare":
        prepare(config)
    elif args.command == "collect":
        collect(config)
    else:
        run_task(config, args.index)


if __name__ == "__main__":
    main()
