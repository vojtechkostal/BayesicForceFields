import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
from gmxtopology import Topology

from bff.domain.specs import Specs
from bff.topology import TopologyModifier, prepare_universe
from bff.workflows.md.main import modify_topology

ROOT = Path(__file__).parents[1]
ACE_TOP = ROOT / "examples/acetate/inputs/common/topol.top"


def test_prepare_universe_suppresses_expected_topology_only_warnings() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prepare_universe(ACE_TOP)

    assert caught == []


def test_dihedraltype9_parameter_uses_multiplicity_and_phase_suffix() -> None:
    dihedrals = [
        SimpleNamespace(func=9, params={"phi_s": 180.0, "kphi": 1.0, "mult": 3}),
        SimpleNamespace(func=9, params={"phi_s": 180.0, "kphi": 2.0, "mult": 3}),
        SimpleNamespace(func=9, params={"phi_s": 0.0, "kphi": 3.0, "mult": 3}),
    ]
    for dihedral in dihedrals:
        dihedral.update = dihedral.params.update

    modifier = object.__new__(TopologyModifier)
    modifier.moleculetypes = [SimpleNamespace(dihedrals=dihedrals)]
    modifier.apply_parameters({"dihedraltype9_3_180": 4.2})

    assert [dihedral.params["kphi"] for dihedral in dihedrals] == [4.2, 4.2, 3.0]


def test_modify_topology_updates_all_matching_gromacs_dihedrals(
    tmp_path: Path,
) -> None:
    specs = Specs({
        "bounds": {"dihedraltype9_6_180": [0.0, 10.0]},
        "charge_constraints": [],
    })
    fn_out = tmp_path / "modified.top"

    modify_topology(ACE_TOP, specs, [4.2], True, fn_out)

    ace = Topology(fn_out).moleculetypes[0]
    dihedrals = [dihedral for dihedral in ace.dihedrals if dihedral.func == 9]
    assert len(dihedrals) == 6
    assert all(dihedral.params["kphi"] == pytest.approx(4.2) for dihedral in dihedrals)
    assert all(
        dihedral.params["phi_s"] == pytest.approx(180.0) for dihedral in dihedrals
    )
    assert all(dihedral.params["mult"] == 6 for dihedral in dihedrals)


def test_dihedraltype9_parameter_rejects_invalid_name() -> None:
    modifier = object.__new__(TopologyModifier)
    modifier.moleculetypes = []

    with pytest.raises(ValueError, match="expected"):
        modifier.apply_parameters({"dihedraltype9 3 180": 4.2})


def test_dihedraltype9_parameter_rejects_missing_term() -> None:
    modifier = object.__new__(TopologyModifier)
    modifier.moleculetypes = []

    with pytest.raises(ValueError, match="not found"):
        modifier.apply_parameters({"dihedraltype9_3_180": 4.2})
