from pathlib import Path

import pytest

from bff.qoi.routines import run_routine
from bff.workflows.build_qoi_datasets.config import load_routines
from bff.workflows.config import ConfigSection


def load_routine_configs(routines: list[dict]):
    config = ConfigSection(
        {"routines": routines}, "", base_dir=Path.cwd(), allowed=("routines",)
    )
    return load_routines(config)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_custom_file_routine_receives_roles_and_authoritative_name(
    tmp_path: Path,
) -> None:
    profile = _write(tmp_path / "profile.pmf", "0 1\n")
    module = _write(
        tmp_path / "routine.py",
        "from bff.qoi import QoI\n"
        "def load_profile(*, inputs, options):\n"
        "    assert inputs == {'pmf': inputs['pmf']}\n"
        "    assert inputs['pmf'].name == 'profile.pmf'\n"
        "    assert options == {'scale': 2}\n"
        "    return QoI('ignored', [1.0, 2.0])\n",
    )
    (routine,) = load_routine_configs(
        [
            {
                "name": "contact-pmf",
                "callable": f"{module}:load_profile",
                "systems": ["contact"],
                "inputs": ["pmf"],
                "options": {"scale": 2},
            }
        ]
    )
    result = run_routine(
        routine,
        universe=None,
        frames=slice(None),
        inputs={"pmf": profile, "other": profile},
        system_id="contact",
        sample_id="reference",
    )
    assert not routine.uses_trajectory
    assert result.name == "contact-pmf"


def test_custom_trajectory_routine_receives_universe_and_frames(
    tmp_path: Path,
) -> None:
    module = _write(
        tmp_path / "trajectory_routine.py",
        "from bff.qoi import QoI\n"
        "def calculate(u, *, frames, options):\n"
        "    assert u == 'universe'\n"
        "    assert frames == slice(2, 8, 2)\n"
        "    return QoI('ignored', [options['value']])\n",
    )
    (routine,) = load_routine_configs(
        [
            {
                "name": "distance",
                "callable": f"{module}:calculate",
                "systems": ["contact"],
                "options": {"value": 3.0},
            }
        ]
    )

    result = run_routine(
        routine,
        universe="universe",
        frames=slice(2, 8, 2),
        inputs={},
        system_id="contact",
        sample_id="sample-0",
    )

    assert routine.uses_trajectory
    assert result.name == "distance"
    assert result.values.tolist() == [3.0]


def test_builtin_routine_receives_selections_as_options() -> None:
    (routine,) = load_routine_configs(
        [
            {
                "name": "rdf-ow",
                "type": "rdf",
                "systems": ["acetate"],
                "selections": {"group_a": "resname ACE", "group_b": "name OW"},
                "options": {"bins": 50},
            }
        ]
    )
    assert routine.options == {
        "group_a": "resname ACE",
        "group_b": "name OW",
        "bins": 50,
    }


def test_routine_errors_name_the_routine_system_and_sample(tmp_path: Path) -> None:
    module = _write(
        tmp_path / "broken.py",
        "def calculate(universe, *, frames, options):\n"
        "    raise ValueError('bad option')\n",
    )
    (routine,) = load_routine_configs(
        [{"name": "broken", "callable": f"{module}:calculate", "systems": ["s"]}]
    )
    with pytest.raises(
        ValueError, match="Routine 'broken', system 's', sample '0': bad"
    ):
        run_routine(
            routine,
            universe=None,
            frames=slice(None),
            inputs={},
            system_id="s",
            sample_id="0",
        )


@pytest.mark.parametrize(
    ("routine", "message"),
    [
        (
            {"name": "x", "type": "rdf", "systems": ["s"], "loader": "a"},
            "unsupported key",
        ),
        ({"name": "x", "type": "adf", "systems": ["s"]}, "type must be one of"),
        ({"name": "x", "systems": ["s"]}, "exactly one of type or callable"),
        ({"name": "x", "type": "rdf", "systems": ["s"], "inputs": ["pmf"]}, "inputs"),
        (
            {
                "name": "x",
                "type": "rdf",
                "systems": ["s"],
                "selections": {"group_a": "a"},
                "options": {"group_a": "b"},
            },
            "both selections and options",
        ),
    ],
)
def test_routine_configs_are_validated(routine, message) -> None:
    with pytest.raises(ValueError, match=message):
        load_routine_configs([routine])
