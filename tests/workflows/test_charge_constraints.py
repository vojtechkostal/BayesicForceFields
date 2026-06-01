from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from gmxtopology import Topology

from bff.domain.specs import Specs
from bff.workflows._shared.campaign import build_specs
from bff.workflows.md.main import modify_topology
from bff.workflows.sample.config import ChargeConstraintConfig

ROOT = Path(__file__).parents[2]
ACE_TOP = ROOT / "examples/acetate/inputs/common/topol.top"
ACE_IONS_TOP = ROOT / "examples/acetate/inputs/common/topol-ions.top"


def _config(tmp_path: Path, bounds: dict, constraints: list[dict]) -> SimpleNamespace:
    return SimpleNamespace(
        campaign_dir=tmp_path,
        systems=[SimpleNamespace(fn_topol=ACE_TOP)],
        bounds=bounds,
        charge_constraints=tuple(
            ChargeConstraintConfig(**constraint) for constraint in constraints
        ),
    )


def test_build_specs_serializes_reconstructable_charge_constraints(
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

    specs = Specs(build_specs(config))
    explicit = [-0.37, 0.09, -0.76]

    assert specs.parameter_names(explicit_only=True) == (
        "charge C1",
        "charge H1 H2 H3",
        "charge O1 O2",
    )
    np.testing.assert_allclose(
        specs.with_implicit_charges([explicit]),
        [[-0.37, 0.82, 0.09, -0.76]],
    )

    fn_out = tmp_path / "modified.top"
    modify_topology(ACE_TOP, specs, explicit, True, fn_out)
    ace = Topology(fn_out).molecules["ACE"][0]
    assert sum(atom.charge for atom in ace.atoms) == pytest.approx(-0.8)
    assert ace.atoms[1].charge == pytest.approx(0.82)

    with pytest.raises(ValueError, match="violate"):
        modify_topology(ACE_TOP, specs, [0.3, 0.3, -0.3], True, fn_out)


def test_system_constraint_can_reconstruct_charge_across_molecule_types(
    tmp_path: Path,
) -> None:
    config = SimpleNamespace(
        campaign_dir=tmp_path,
        systems=[SimpleNamespace(fn_topol=ACE_IONS_TOP)],
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

    specs = Specs(build_specs(config))
    np.testing.assert_allclose(specs.with_implicit_charges([[0.5]]), [[0.5, 1.12]])

    fn_out = tmp_path / "modified-ions.top"
    modify_topology(ACE_IONS_TOP, specs, [0.5], True, fn_out)
    topol = Topology(fn_out)
    assert sum(atom.charge for atom in topol.atoms) == pytest.approx(0.0)


def test_build_specs_rejects_partially_overlapping_constraints(tmp_path: Path) -> None:
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

    with pytest.raises(ValueError, match="partially overlap"):
        build_specs(config)


def test_build_specs_rejects_duplicate_charge_parameter_tokens(tmp_path: Path) -> None:
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
        build_specs(config)


def test_build_specs_rejects_parent_implicit_parameter_in_child(
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

    with pytest.raises(ValueError, match="must not appear"):
        build_specs(config)
