from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from gmxtopology import Topology

from bff.domain.charge_constraints import compile_specs as compile_domain_specs
from bff.workflows.campaign.job import write_sample_topology
from bff.workflows.sample_parameters.config import ChargeConstraintConfig

ROOT = Path(__file__).parents[2]


def _write(path: Path, text: str = "data\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


ACE_TOP = ROOT / "examples/acetate/inputs/topol.top"
ACE_IONS_TOP = ROOT / "examples/acetate/inputs/topol-ions.top"


def compile_specs(config: SimpleNamespace):
    return compile_domain_specs(
        config.bounds,
        config.charge_constraints,
        [system.topology_path for system in config.systems],
    )


def _config(tmp_path: Path, bounds: dict, constraints: list[dict]) -> SimpleNamespace:
    return SimpleNamespace(
        systems=[SimpleNamespace(topology_path=ACE_TOP)],
        bounds=bounds,
        charge_constraints=tuple(
            ChargeConstraintConfig(**constraint) for constraint in constraints
        ),
    )


def test_compile_specs_serializes_reconstructable_charge_constraints(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        bounds={
            "charge C2": [0.0, 1.0],
            "charge O1 O2": [-0.8, -0.3],
            "charge C1": [-1.0, 0.3],
            "charge H1 H2 H3": [-0.3, 0.3],
        },
        constraints=[
            {
                "selection": "resname ACE",
                "target": -0.8,
                "scope": "residue",
                "implicit": "charge C2",
            }
        ],
    )

    specs = compile_specs(config)
    explicit = [-0.37, 0.09, -0.76]

    assert specs.explicit_names == (
        "charge C1",
        "charge H1 H2 H3",
        "charge O1 O2",
    )
    np.testing.assert_allclose(
        specs.complete([explicit]),
        [[-0.37, 0.82, 0.09, -0.76]],
    )

    fn_out = tmp_path / "modified.top"
    write_sample_topology(ACE_TOP, specs, explicit, fn_out)
    ace = Topology(fn_out).moleculetype("ACE")
    assert sum(atom.charge for atom in ace.atoms) == pytest.approx(-0.8)
    assert ace.atoms[1].charge == pytest.approx(0.82)

    with pytest.raises(ValueError, match="violate"):
        write_sample_topology(ACE_TOP, specs, [0.3, 0.3, -0.3], fn_out)


def test_system_constraint_can_reconstruct_charge_across_molecule_types(
    tmp_path: Path,
) -> None:
    config = SimpleNamespace(
        systems=[SimpleNamespace(topology_path=ACE_IONS_TOP)],
        bounds={"charge C2": [0.0, 1.0], "charge CAL": [0.0, 2.0]},
        charge_constraints=(
            ChargeConstraintConfig(
                selection="resname ACE or resname CAL",
                target=0.0,
                scope="system",
                implicit="charge CAL",
            ),
        ),
    )

    specs = compile_specs(config)
    np.testing.assert_allclose(specs.complete([[0.5]]), [[0.5, 1.12]])

    fn_out = tmp_path / "modified-ions.top"
    write_sample_topology(ACE_IONS_TOP, specs, [0.5], fn_out)
    topol = Topology(fn_out)
    assert sum(atom.charge for atom in topol.atoms) == pytest.approx(0.0)


def test_compile_specs_solves_overlapping_constraints_together(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        bounds={
            "charge C1": [-2.0, 2.0],
            "charge C2": [-2.0, 2.0],
            "charge O1": [-2.0, 2.0],
        },
        constraints=[
            {
                "selection": "resname ACE and name C1 C2",
                "target": 0.0,
                "scope": "residue",
                "implicit": "charge C1",
            },
            {
                "selection": "resname ACE and name C2 O1",
                "target": 0.0,
                "scope": "residue",
                "implicit": "charge O1",
            },
        ],
    )

    specs = compile_specs(config)
    c1, c2, o1 = specs.complete([[0.3]])[0]
    assert c2 == pytest.approx(0.3)
    assert c1 + c2 == pytest.approx(0.0)
    assert c2 + o1 == pytest.approx(0.0)


def test_compile_specs_rejects_duplicate_charge_parameter_tokens(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        bounds={"charge C1 C1": [-2.0, 2.0]},
        constraints=[
            {
                "selection": "resname ACE and name C1",
                "target": 0.0,
                "scope": "residue",
                "implicit": "charge C1 C1",
            }
        ],
    )

    with pytest.raises(ValueError, match="Duplicate atom name or type"):
        compile_specs(config)


def test_compile_specs_rejects_constraints_that_cannot_be_solved(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        bounds={"charge C1": [-2.0, 2.0], "charge C2": [-2.0, 2.0]},
        constraints=[
            {
                "selection": "resname ACE and name C1 C2",
                "target": 0.0,
                "scope": "residue",
                "implicit": "charge C1",
            },
            {
                "selection": "resname ACE",
                "target": -0.8,
                "scope": "residue",
                "implicit": "charge C2",
            },
        ],
    )

    # Both equations contain C1 and C2 once each: they cannot fix both.
    with pytest.raises(ValueError, match="uniquely"):
        compile_specs(config)


def test_sampling_records_a_reproducible_seed(tmp_path: Path) -> None:
    from bff.workflows.sample_parameters.main import main

    inputs = {
        "topology": str(ACE_TOP),
        "coordinates": str(_write(tmp_path / "system.gro")),
        "mdp_production": str(_write(tmp_path / "production.mdp")),
        "index": str(_write(tmp_path / "index.ndx")),
    }
    config = {
        "campaign_dir": "./campaign",
        "systems": [{"system_id": "acetate", "inputs": inputs, "n_steps": 10}],
        "bounds": {"charge C2": [0.0, 1.0]},
        "n_samples": 4,
        "dispatch": False,
        "gmx_cmd": "gmx",
        "job_scheduler": "local",
    }
    fn_config = tmp_path / "sample.yaml"

    def run(**options) -> dict:
        fn_config.write_text(yaml.safe_dump(config | options))
        main(fn_config)
        return yaml.safe_load((tmp_path / "campaign" / "samples.yaml").read_text())

    first = run()
    seed = first["provenance"]["seed"]
    assert first["provenance"]["source"] == "latin_hypercube"
    repeated = run(seed=seed, overwrite=True)
    assert repeated["samples"] == first["samples"]
