from pathlib import Path
from types import SimpleNamespace

import pytest

from bff.domain.charge_constraints import check_specs_topologies, compile_specs

ROOT = Path(__file__).parents[2]
ACE_TOP = ROOT / "examples/acetate/inputs/topol.top"
ACE_IONS_TOP = ROOT / "examples/acetate/inputs/topol-ions.top"


def _constraint(selection: str, scope: str, implicit: str, target: float):
    return SimpleNamespace(
        selection=selection, scope=scope, implicit=implicit, target=target
    )


def test_specs_apply_to_other_systems_with_the_same_equation() -> None:
    specs = compile_specs(
        {"charge C2": [0.0, 1.0], "charge O1 O2": [-0.8, -0.3]},
        [_constraint("resname ACE", "residue", "charge C2", -0.8)],
        [ACE_TOP],
    )

    # The calcium system contains the same acetate residue.
    check_specs_topologies(specs, [ACE_IONS_TOP])


def test_specs_tolerate_parameters_and_constraints_absent_from_a_system() -> None:
    specs = compile_specs(
        {"charge C2": [0.0, 1.0], "charge CAL": [0.0, 2.0]},
        [_constraint("resname CAL", "residue", "charge CAL", 1.6)],
        [ACE_IONS_TOP],
    )

    # Neither the CAL parameter nor its constraint exists without calcium.
    check_specs_topologies(specs, [ACE_TOP])


def test_specs_reject_a_constraint_that_changes_meaning_in_a_system() -> None:
    specs = compile_specs(
        {"charge C2": [0.0, 1.0]},
        [_constraint("resname ACE or resname CAL", "system", "charge C2", 0.0)],
        [ACE_IONS_TOP],
    )

    # Without calcium, the same selection sums to a different fixed charge.
    with pytest.raises(ValueError, match="different charge equation"):
        check_specs_topologies(specs, [ACE_TOP])
