import importlib.util
import sys
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest
import yaml

SCRIPT = Path(__file__).parents[2] / "scripts" / "label_structures.py"
FAKE_CP2K = """\
import sys
from pathlib import Path

n_atoms = int(Path("structure.xyz").read_text().split()[0])
lines = [" ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]   -1.5"]
lines += [
    f" FORCES| {i + 1} 0.01 -0.02 0.03 0.04" for i in range(n_atoms)
]
Path(sys.argv[sys.argv.index("-o") + 1]).write_text("\\n".join(lines) + "\\n")
"""


@pytest.fixture(scope="module")
def label():
    spec = importlib.util.spec_from_file_location("label_structures", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_water(tmp_path: Path, n_frames: int) -> tuple[Path, Path]:
    universe = mda.Universe.empty(3, trajectory=True)
    universe.add_TopologyAttr("names", ["OW", "HW1", "HW2"])
    universe.add_TopologyAttr("elements", ["O", "H", "H"])
    universe.add_TopologyAttr("resnames", ["SOL"])
    universe.add_TopologyAttr("resids", [1])
    fn_gro = tmp_path / "water.gro"
    fn_xtc = tmp_path / "water.xtc"
    universe.dimensions = [12.0, 12.0, 12.0, 90.0, 90.0, 90.0]
    universe.atoms.positions = np.array(
        [[1.0, 1.0, 1.0], [1.9, 1.0, 1.0], [1.0, 1.9, 1.0]]
    )
    universe.atoms.write(fn_gro)
    with mda.Writer(str(fn_xtc), n_atoms=3) as writer:
        for frame in range(n_frames):
            universe.atoms.positions += 0.1 * frame
            writer.write(universe.atoms)
    return fn_gro, fn_xtc


def write_config(tmp_path: Path, **overrides) -> Path:
    fn_gro, fn_xtc = write_water(tmp_path, n_frames=10)
    (tmp_path / "sp.inp").write_text(
        "COORD_FILE_NAME structure.xyz\n@INCLUDE cell.inc\n"
    )
    for element in ("h", "o"):
        (tmp_path / f"atom-{element}.inp").write_text("COORD_FILE_NAME structure.xyz\n")
    (tmp_path / "fake_cp2k.py").write_text(FAKE_CP2K)
    config = {
        "output_dir": "labels",
        "cp2k_cmd": f"{sys.executable} {tmp_path / 'fake_cp2k.py'}",
        "split": {"train": 0.5, "valid": 0.25, "test": 0.25},
        "systems": [
            {
                "id": "water",
                "topology": fn_gro.name,
                "trajectory": fn_xtc.name,
                "n_snapshots": 4,
                "cp2k_input": "sp.inp",
            }
        ],
        "single_atom_inputs": {"H": "atom-h.inp", "O": "atom-o.inp"},
        "slurm": {"max_parallel_jobs": 2, "sbatch": {"time": "01:00:00"}},
        **overrides,
    }
    fn_config = tmp_path / "config.yaml"
    fn_config.write_text(yaml.safe_dump(config))
    return fn_config


def test_prepare_task_and_collect_write_labeled_datasets(
    label, tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    fn_config = write_config(tmp_path)
    tasks = label.prepare(fn_config)
    assert [task["kind"] for task in tasks] == ["snapshot"] * 4 + ["atom"] * 2
    assert [task["frame"] for task in tasks[:4]] == [0, 3, 6, 9]
    run_dir = Path(tasks[0]["run_dir"])
    assert (run_dir / "cell.inc").read_text().startswith("A 12 0 0\n")

    for index in range(len(tasks)):
        label.run_task(fn_config, index)
    label.collect(fn_config)

    output = tmp_path / "labels"
    counts = {
        split: (output / f"{split}.extxyz").read_text().count("energy=")
        for split in ("train", "valid", "test")
    }
    assert counts == {"train": 2, "valid": 1, "test": 1}
    header = (output / "train.extxyz").read_text().splitlines()[1]
    assert f"energy={-1.5 * label.HARTREE_TO_EV:.16g}" in header
    assert "config_type=water" in header
    assert set(yaml.safe_load((output / "energies.yaml").read_text())) == {1, 8}
    manifest = yaml.safe_load((output / "label-results.yaml").read_text())
    assert manifest["complete"] is True
    assert manifest["systems"]["water"]["frame_indices"] == [0, 3, 6, 9]

    # Finished tasks are skipped, so re-running does not call CP2K again.
    (tmp_path / "fake_cp2k.py").write_text("raise SystemExit(1)\n")
    label.prepare(fn_config)
    label.run_task(fn_config, 0)


def test_incomplete_atoms_write_partial_energies(label, tmp_path) -> None:
    fn_config = write_config(tmp_path)
    label.prepare(fn_config)
    label.collect(fn_config)

    output = tmp_path / "labels"
    assert not (output / "energies.yaml").exists()
    assert (output / "energies.partial.yaml").exists()
    manifest = yaml.safe_load((output / "label-results.yaml").read_text())
    assert manifest["complete"] is False
    assert len(manifest["failures"]) == 6


def test_run_submits_only_pending_tasks_in_bounded_arrays(
    label, tmp_path, monkeypatch
) -> None:
    fn_config = write_config(
        tmp_path,
        slurm={
            "max_parallel_jobs": 2,
            "max_array_size": 4,
            "sbatch": {"time": "01:00:00", "ntasks": 8},
            "setup": ["module load cp2k"],
        },
    )
    submitted: list[list[str]] = []

    class FakePopen:
        def __init__(self, command):
            submitted.append(command)

        def wait(self):
            return 0

    monkeypatch.setattr(label.subprocess, "Popen", FakePopen)
    label.run_all(fn_config)

    assert [command[3] for command in submitted] == ["--array=0-3%2", "--array=0-1%2"]
    assert submitted[1][4] == "--export=ALL,TASK_OFFSET=4"
    output = tmp_path / "labels"
    assert (output / "pending-tasks.txt").read_text().split() == [
        str(index) for index in range(6)
    ]
    script = (output / "label-array.sbatch").read_text()
    assert "#SBATCH --ntasks=8" in script
    assert "module load cp2k" in script
    assert script.rstrip().endswith('task {} "$TASK_INDEX"'.format(fn_config))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"scheduler": "local"}, "unsupported key"),
        ({"slurm": {"sbatch": {"array": "0-3"}}}, "set by this script"),
        ({"split": {"train": 0.5, "test": 0.1}}, "must be 1"),
    ],
)
def test_invalid_configs_are_rejected(label, tmp_path, overrides, message) -> None:
    fn_config = write_config(tmp_path, **overrides)
    with pytest.raises(ValueError, match=message):
        label.load_config(fn_config)


def test_stress_is_converted_to_tension_positive_ev_per_cubic_angstrom(
    label,
) -> None:
    text = "\n".join(
        [
            " STRESS| Analytical stress tensor [GPa]",
            " STRESS|      x   1.0   0.0   0.0",
            " STRESS|      y   0.0   2.0   0.0",
            " STRESS|      z   0.0   0.0   3.0",
        ]
    )
    stress = label.read_stress(text)
    assert stress[1][1] == pytest.approx(-2.0 * label.GPA_TO_EV_ANGSTROM3)


def test_elements_override_guess_from_atom_names(label, tmp_path) -> None:
    fn_gro, fn_xtc = write_water(tmp_path, n_frames=2)
    fn_gro.write_text(fn_gro.read_text().replace("HW2", "CAL"))

    _, get_frame = label.open_trajectory(fn_gro, fn_xtc, "all", {"CAL": "Ca"})

    assert get_frame(0)[0] == ["O", "H", "Ca"]
