import numpy as np
import pytest
import torch

from bff.domain.specs import Specs, latin_hypercube, parameter_kind


def _spec_data() -> dict:
    return {
        "bounds": {
            "charge A": [-1.0, 1.0],
            "charge B": [-0.5, 0.5],
            "sigma C": [0.1, 2.0],
        },
        "charge_constraints": [
            {
                "selection": "name A B",
                "target": 0.0,
                "scope": "residue",
                "implicit": "charge B",
                "coefficients": {"charge A": 1.0, "charge B": 1.0},
            }
        ],
    }


def test_bounds_are_sorted_and_validate_limits() -> None:
    specs = Specs(
        {"bounds": {"z": (0.0, 1.0), "a": (-1.0, 1.0)}, "charge_constraints": []}
    )

    assert specs.names == ("a", "z")
    assert specs.bounds == {"a": (-1.0, 1.0), "z": (0.0, 1.0)}
    assert specs.explicit_bounds.tolist() == [[-1.0, 1.0], [0.0, 1.0]]

    with pytest.raises(ValueError, match="Lower bound"):
        Specs({"bounds": {"x": (2.0, 1.0)}, "charge_constraints": []})


def test_specs_charge_helpers() -> None:
    specs = Specs(_spec_data())

    assert specs.implicit_names == ("charge B",)
    assert specs.explicit_names == ("charge A", "sigma C")
    assert specs.complete([[0.2, 1.0]]).tolist() == [[0.2, -0.2, 1.0]]
    assert specs.as_dict([0.2, -0.2, 1.0]) == {
        "charge A": 0.2,
        "charge B": -0.2,
        "sigma C": 1.0,
    }
    assert Specs(specs.to_dict()) == specs


def test_parameter_kind_groups_dihedral_labels() -> None:
    assert parameter_kind("charge O1 O2") == "charge"
    assert parameter_kind("dihedraltype9_3_180") == "dihedraltype9"
    assert parameter_kind("define VSA") == "define"


def test_specs_rejects_missing_required_fields() -> None:
    with pytest.raises(ValueError, match="Missing required"):
        Specs({"bounds": {}})


def test_specs_solve_dependent_constraints_together() -> None:
    specs = Specs(
        {
            "bounds": {
                "charge A": [-1.0, 1.0],
                "charge B": [-1.0, 1.0],
                "charge C": [0.0, 2.0],
            },
            "charge_constraints": [
                {
                    "selection": "name A B C",
                    "target": 1.0,
                    "scope": "residue",
                    "implicit": "charge C",
                    "coefficients": {
                        "charge A": 1.0,
                        "charge B": 1.0,
                        "charge C": 1.0,
                    },
                },
                {
                    "selection": "name A B",
                    "target": 0.0,
                    "scope": "residue",
                    "implicit": "charge B",
                    "coefficients": {"charge A": 1.0, "charge B": 1.0},
                },
            ],
        }
    )

    np.testing.assert_allclose(specs.complete([[0.2]]), [[0.2, -0.2, 1.0]])


def test_specs_reject_constraints_that_do_not_fix_their_implicit_charges() -> None:
    constraint = {
        "selection": "name A B",
        "target": 0.0,
        "scope": "system",
        "coefficients": {"charge A": 1.0, "charge B": 1.0},
    }
    with pytest.raises(ValueError, match="uniquely"):
        Specs(
            {
                "bounds": {"charge A": [-1.0, 1.0], "charge B": [-1.0, 1.0]},
                "charge_constraints": [
                    constraint | {"implicit": "charge A"},
                    constraint | {"implicit": "charge B", "selection": "name B A"},
                ],
            }
        )


def test_specs_reject_infeasible_constraints() -> None:
    with pytest.raises(ValueError, match="incompatible"):
        Specs(
            {
                "bounds": {"charge A": [0.0, 1.0], "charge B": [0.0, 1.0]},
                "charge_constraints": [
                    {
                        "selection": "name A B",
                        "target": 3.0,
                        "scope": "system",
                        "implicit": "charge B",
                        "coefficients": {"charge A": 1.0, "charge B": 1.0},
                    }
                ],
            }
        )


def test_specs_reconstruct_multi_atom_implicit_parameter_per_atom() -> None:
    specs = Specs(
        {
            "bounds": {"charge A": [-1.0, 1.0], "charge B C": [-0.5, 0.5]},
            "charge_constraints": [
                {
                    "selection": "name A B C",
                    "target": 0.0,
                    "scope": "residue",
                    "implicit": "charge B C",
                    "coefficients": {"charge A": 1.0, "charge B C": 2.0},
                }
            ],
        }
    )

    assert specs.complete([[0.6]]).tolist() == [[0.6, -0.3]]
    assert specs.is_valid([[0.6]]).tolist() == [True]


def test_is_valid_accepts_numpy_and_torch_inputs() -> None:
    specs = Specs(_spec_data())

    assert specs.is_valid(np.array([[0.2, 1.0], [0.8, 1.0]])).tolist() == [True, False]
    assert specs.is_valid(torch.tensor([[0.0, 1.0]])).tolist() == [True]

    with pytest.raises(ValueError, match="columns"):
        specs.is_valid(np.zeros((1, 3)))


def test_violations_describe_bound_violations() -> None:
    message = Specs(_spec_data()).violations([[0.8, 1.0]])

    assert "charge B=-0.8" in message
    assert "below" in message


def test_latin_hypercube_respects_bounds_and_implicit_charges() -> None:
    specs = Specs(_spec_data())

    samples = latin_hypercube(specs, 50, seed=1)

    assert samples.shape == (50, 2)
    assert specs.is_valid(samples).all()
    # charge B = -charge A must stay within [-0.5, 0.5].
    assert np.abs(samples[:, 0]).max() <= 0.5
    np.testing.assert_allclose(latin_hypercube(specs, 50, seed=1), samples)


def test_latin_hypercube_handles_zero_and_negative_counts() -> None:
    specs = Specs({"bounds": {"sigma A": [0.0, 1.0]}, "charge_constraints": []})

    assert latin_hypercube(specs, 0).shape == (0, 1)
    with pytest.raises(ValueError, match="non-negative"):
        latin_hypercube(specs, -1)


def test_latin_hypercube_handles_fully_constrained_parameters() -> None:
    specs = Specs(
        {
            "bounds": {"charge A": [-1.0, 1.0]},
            "charge_constraints": [
                {
                    "selection": "name A",
                    "target": 0.0,
                    "scope": "system",
                    "implicit": "charge A",
                    "coefficients": {"charge A": 1.0},
                }
            ],
        }
    )

    samples = latin_hypercube(specs, 3)

    assert samples.shape == (3, 0)
    assert specs.complete(samples).tolist() == [[0.0], [0.0], [0.0]]


def test_specs_can_be_copied_and_pickled() -> None:
    import copy
    import pickle

    specs = Specs({"bounds": {"charge A": [0.0, 1.0]}, "charge_constraints": []})
    assert copy.deepcopy(specs) == specs
    assert pickle.loads(pickle.dumps(specs)) == specs
