from pathlib import Path
from types import SimpleNamespace

import pytest

from bff.domain.charge_constraints import check_specs_topologies, compile_specs

ROOT = Path(__file__).parents[2]
ACE_TOP = ROOT / "examples/acetate/inputs/acetate.top"
ACE_IONS_TOP = ROOT / "examples/acetate/inputs/calcium-acetate.top"


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


def test_a_define_parameter_must_exist_in_a_topology(tmp_path: Path) -> None:
    inputs = ACE_TOP.parent
    top = tmp_path / "with-define.top"
    top.write_text(
        "#define VSA 120\n"
        + "".join(
            f'#include "{inputs / name}"\n'
            for name in (
                "ff/defaults.itp",
                "ff/ace-atomtypes.itp",
                "ff/water-atomtypes.itp",
                "ff/ace.itp",
                "ff/water.itp",
            )
        )
        + "\n[ system ]\nx\n\n[ molecules ]\nACE 1\nSOL 128\n"
    )
    bounds = {"charge C2": [0.0, 1.0], "define VSA": [60.0, 150.0]}
    constraint = [_constraint("resname ACE", "residue", "charge C2", -0.8)]

    compile_specs(bounds, constraint, [top])
    with pytest.raises(ValueError, match="#define directive that no configured"):
        compile_specs(bounds, constraint, [ACE_TOP])
    with pytest.raises(ValueError, match="define VSA"):
        check_specs_topologies(compile_specs(bounds, constraint, [top]), [ACE_TOP])
