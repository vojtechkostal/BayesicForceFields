import numpy as np
import pytest
import torch

from bff.domain.specs import Bounds, ChargeConstraint, RandomParamsGenerator, Specs


def _spec_data() -> dict:
    return {
        "mol_resname": "ACE",
        "bounds": {
            "charge A": [-1.0, 1.0],
            "charge B": [-0.5, 0.5],
            "sigma C": [0.1, 2.0],
        },
        "total_charge": 0.0,
        "charge_target": 0.0,
        "implicit_atoms": ["B"],
    }


def test_bounds_are_sorted_and_validate_limits() -> None:
    bounds = Bounds({"z": (0.0, 1.0), "a": (-1.0, 1.0)})

    assert bounds.names.tolist() == ["a", "z"]
    assert bounds.array.tolist() == [[-1.0, 1.0], [0.0, 1.0]]
    assert bounds.index("z") == 1

    with pytest.raises(ValueError, match="Lower bound"):
        Bounds({"x": (2.0, 1.0)})


def test_specs_charge_helpers() -> None:
    specs = Specs(_spec_data())

    assert specs.implicit_param == "charge B"
    assert specs.implicit_param_index == 1
    assert specs.parameter_names(explicit_only=True) == ("charge A", "sigma C")
    assert specs.explicit_charge_coefficients.tolist() == [1, 0]
    assert specs.implicit_charge([[0.2, 1.0]]).tolist() == [-0.2]
    assert specs.with_implicit_charge([[0.2, 1.0]]).tolist() == [[0.2, -0.2, 1.0]]


def test_specs_rejects_missing_required_fields() -> None:
    with pytest.raises(ValueError, match="Missing required"):
        Specs({"bounds": {}, "implicit_atoms": []})


def test_charge_constraint_accepts_numpy_and_torch_inputs() -> None:
    constraint = ChargeConstraint(_spec_data())

    assert constraint.is_valid(np.array([[0.2, 1.0], [0.8, 1.0]])).tolist() == [
        True,
        False,
    ]
    assert constraint(torch.tensor([[0.0, 1.0]])).tolist() == [True]

    with pytest.raises(ValueError, match="shape"):
        constraint(np.zeros((1, 3)))


def test_random_params_generator_respects_bounds_and_constraint() -> None:
    def constraint(x: np.ndarray) -> np.ndarray:
        return x[:, 0] > 0.0

    generator = RandomParamsGenerator(
        np.array([[-1.0, 1.0], [2.0, 3.0]]),
        constraint=constraint,
    )

    samples = generator(5)

    assert samples.shape == (5, 2)
    assert np.all(samples[:, 0] > 0.0)
    assert np.all((samples[:, 1] >= 2.0) & (samples[:, 1] <= 3.0))


def test_random_params_generator_handles_zero_and_negative_counts() -> None:
    generator = RandomParamsGenerator(np.array([[0.0, 1.0]]))

    assert generator(0).shape == (0, 1)
    with pytest.raises(ValueError, match="non-negative"):
        generator(-1)
