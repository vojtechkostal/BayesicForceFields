"""Parameter specification: bounds and the charge equations of implicit charges.

``specs.yaml`` holds the bounds of every parameter and one linear equation per
charge constraint (see :mod:`bff.domain.charge_constraints`)::

    bounds: {charge C1: [-1, 0.3], charge C2: [0, 1], ...}
    charge_constraints:
      - {selection: resname ACE, target: -0.8, scope: residue,
         implicit: charge C2, coefficients: {...}, fixed_charge: 0.0}

Parameters are the *sampled* (explicit) ones plus the *implicit* charges that
the equations compute from them. All name tuples are sorted by name; arrays of
parameter values have one column per name of ``names`` (all parameters) or
``explicit_names`` (sampled parameters only).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Union

import numpy as np
import torch
from scipy.optimize import linprog
from scipy.stats.qmc import LatinHypercube

from ..io.utils import load_yaml, save_yaml

PathLike = Union[str, Path]


def parameter_kind(name: str) -> str:
    """Kind of a parameter label: charge, sigma, epsilon, dihedraltype9, define."""
    head = name.split()[0]
    return "dihedraltype9" if head.startswith("dihedraltype9") else head


@dataclass(frozen=True)
class ChargeConstraintSpec:
    """One charge equation: ``sum(coefficients * charges) + fixed = target``."""

    selection: str
    target: float
    scope: str
    implicit: str
    coefficients: Mapping[str, float]
    fixed_charge: float = 0.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ChargeConstraintSpec:
        """Validate and read a ``charge_constraints`` entry of ``specs.yaml``."""
        required = {"selection", "target", "scope", "implicit", "coefficients"}
        missing = required - set(data)
        if missing:
            fields = ", ".join(sorted(repr(key) for key in missing))
            raise ValueError(f"Charge constraint is missing field(s): {fields}")
        if data["scope"] not in {"system", "residue"}:
            raise ValueError(
                f"Unsupported charge-constraint scope {data['scope']!r}; "
                "expected 'system' or 'residue'."
            )
        if not isinstance(data["coefficients"], Mapping):
            raise ValueError("Charge-constraint 'coefficients' must be a mapping.")
        return cls(
            selection=str(data["selection"]),
            target=float(data["target"]),
            scope=str(data["scope"]),
            implicit=str(data["implicit"]),
            coefficients={str(k): float(v) for k, v in data["coefficients"].items()},
            fixed_charge=float(data.get("fixed_charge", 0.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        """The ``charge_constraints`` entry of ``specs.yaml``."""
        return {
            "selection": self.selection,
            "target": self.target,
            "scope": self.scope,
            "implicit": self.implicit,
            "coefficients": dict(self.coefficients),
            "fixed_charge": self.fixed_charge,
        }


class Specs:
    """Bounds of all parameters and the equations that fix the implicit charges.

    ``source`` is a mapping or a ``specs.yaml`` path with ``bounds`` (name to
    ``[lower, upper]``) and ``charge_constraints`` (a list of equations, see
    :class:`ChargeConstraintSpec`).

    Attributes
    ----------
    bounds : dict
        ``name -> (lower, upper)``, sorted by name.
    names : tuple of str
        All parameters (``bounds`` keys).
    explicit_names, implicit_names : tuple of str
        Sampled parameters and the charges the constraints solve for.
    charge_constraints : tuple of ChargeConstraintSpec
    """

    def __init__(self, source: dict[str, Any] | PathLike) -> None:
        if isinstance(source, dict):
            data = dict(source)
        elif isinstance(source, (str, Path)):
            data = load_yaml(source)
        else:
            raise TypeError(f"Unsupported specs source: {type(source)}")
        missing = {"bounds", "charge_constraints"} - set(data)
        if missing:
            fields = ", ".join(sorted(repr(key) for key in missing))
            raise ValueError(f"Missing required specs field(s): {fields}")
        if not isinstance(data["charge_constraints"], list):
            raise ValueError("'charge_constraints' must be a list.")

        bounds = {}
        for name, (lower, upper) in data["bounds"].items():
            if lower > upper:
                raise ValueError(
                    f"Lower bound {lower} is greater than upper bound {upper} "
                    f"for parameter {name!r}."
                )
            bounds[str(name)] = (float(lower), float(upper))
        self.bounds: dict[str, tuple[float, float]] = dict(sorted(bounds.items()))
        self.charge_constraints = tuple(
            ChargeConstraintSpec.from_dict(item) for item in data["charge_constraints"]
        )

        self.names = tuple(self.bounds)
        self.implicit_names = tuple(c.implicit for c in self.charge_constraints)
        self.explicit_names = tuple(
            name for name in self.names if name not in self.implicit_names
        )
        self._check_equations()
        self._explicit = [self.names.index(name) for name in self.explicit_names]
        self._implicit = [self.names.index(name) for name in self.implicit_names]
        self._matrix = np.asarray(
            [[c.coefficients.get(name, 0.0) for name in self.names]
             for c in self.charge_constraints],
            dtype=float,
        ).reshape(len(self.charge_constraints), len(self.names))
        self._targets = np.asarray(
            [c.target - c.fixed_charge for c in self.charge_constraints], dtype=float
        )
        self._bounds_array = np.asarray(list(self.bounds.values()), dtype=float)
        self._bounds_array = self._bounds_array.reshape(-1, 2)
        self._cache: dict = {}
        self._check_solvable()

    def _check_equations(self) -> None:
        if len(self.implicit_names) != len(set(self.implicit_names)):
            raise ValueError(
                "Each charge constraint must own a distinct implicit parameter."
            )
        for c in self.charge_constraints:
            if c.implicit not in self.bounds:
                raise ValueError(
                    f"Implicit parameter {c.implicit!r} is not defined in bounds."
                )
            unknown = sorted(set(c.coefficients) - set(self.bounds))
            if unknown:
                raise ValueError(
                    f"Charge constraint {c.selection!r} references unknown bounded "
                    f"parameter(s): {', '.join(map(repr, unknown))}."
                )
            not_charges = [n for n in c.coefficients if parameter_kind(n) != "charge"]
            if not_charges or parameter_kind(c.implicit) != "charge":
                raise ValueError(
                    f"Charge constraint {c.selection!r} may only involve charge "
                    f"parameters, got {not_charges or [c.implicit]}."
                )
            if np.isclose(c.coefficients.get(c.implicit, 0.0), 0.0):
                raise ValueError(
                    f"Implicit parameter {c.implicit!r} is not selected by its "
                    f"owning constraint {c.selection!r}."
                )

    def _check_solvable(self) -> None:
        """The equations must fix the implicit charges, within the bounds."""
        n = len(self.charge_constraints)
        if not n:
            return
        if np.linalg.matrix_rank(self._matrix[:, self._implicit]) < n:
            raise ValueError(
                "The charge constraints do not determine their implicit "
                "parameters uniquely; give each constraint an implicit atom that "
                "the other constraints do not fix as well."
            )
        result = linprog(
            np.zeros(len(self.names)),
            A_eq=self._matrix,
            b_eq=self._targets,
            bounds=self._bounds_array,
            method="highs",
        )
        if not result.success:
            raise ValueError(
                "Charge constraints are incompatible with each other or with the "
                f"configured bounds: {result.message}"
            )

    # -- serialization -------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Specs({len(self.names)} parameters: {len(self.explicit_names)} "
            f"sampled, {len(self.implicit_names)} implicit)"
        )

    @classmethod
    def load(cls, source: PathLike) -> Specs:
        """Read a ``specs.yaml`` file."""
        return cls(source)

    def to_dict(self) -> dict[str, Any]:
        """The mapping :class:`Specs` is built from."""
        return {
            "bounds": {name: list(bounds) for name, bounds in self.bounds.items()},
            "charge_constraints": [c.to_dict() for c in self.charge_constraints],
        }

    def write(self, fn_out: PathLike) -> None:
        """Write ``specs.yaml``."""
        save_yaml(self.to_dict(), fn_out)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Specs) and self.to_dict() == other.to_dict()

    __hash__ = None  # type: ignore[assignment]

    # -- parameter values ----------------------------------------------------

    @property
    def explicit_bounds(self) -> np.ndarray:
        """Bounds of the sampled parameters, shape ``(n_explicit, 2)``."""
        return self._bounds_array[self._explicit]

    def complete(self, X: Any) -> Any:
        """Add the implicit charges to sampled parameter vectors.

        ``X`` has one row per vector and one column per ``explicit_names``;
        the result has one column per ``names``. NumPy input gives NumPy
        output. A torch tensor is completed where it lives, in its own dtype,
        without any transfer.
        """
        is_torch = isinstance(X, torch.Tensor)
        x = X if is_torch else torch.as_tensor(np.asarray(X, dtype=float))
        x = x.reshape(1, -1) if x.dim() < 2 else x
        if x.shape[1] != len(self._explicit):
            raise ValueError(
                f"Expected {len(self._explicit)} columns for {self.explicit_names}, "
                f"got shape {tuple(x.shape)}."
            )
        explicit, implicit, solve, coefficients, targets, _ = self._on(
            x.device, x.dtype
        )
        full = x.new_zeros((len(x), len(self.names)))
        full[:, explicit] = x
        if self.charge_constraints:
            full[:, implicit] = (targets - x @ coefficients.T) @ solve.T
        return full if is_torch else full.numpy()

    def _on(self, device: torch.device, dtype: torch.dtype) -> tuple:
        """Index and coefficient tensors for ``device`` and ``dtype``, cached."""
        key = (str(device), dtype)
        if key not in self._cache:
            matrix = torch.as_tensor(self._matrix)
            solve = (
                torch.linalg.inv(matrix[:, self._implicit])
                if self.charge_constraints
                else torch.empty((0, 0), dtype=matrix.dtype)
            )
            self._cache[key] = tuple(
                tensor.to(device, dtype if tensor.is_floating_point() else None)
                for tensor in (
                    torch.tensor(self._explicit, dtype=torch.long),
                    torch.tensor(self._implicit, dtype=torch.long),
                    solve,
                    matrix[:, self._explicit],
                    torch.as_tensor(self._targets),
                    torch.as_tensor(self._bounds_array),
                )
            )
        return self._cache[key]

    def is_valid(self, X: Any) -> Any:
        """Whether each sampled vector keeps every parameter, implicit charges
        included, within its bounds. Returns a mask of the type and device of
        ``X``."""
        full = self.complete(X)
        is_torch = isinstance(full, torch.Tensor)
        full = full if is_torch else torch.as_tensor(full)
        bounds = self._on(full.device, full.dtype)[5]
        valid = ((full >= bounds[:, 0]) & (full <= bounds[:, 1])).all(dim=1)
        return valid if is_torch else valid.numpy()

    def violations(self, X: Any, max_items: int = 5) -> str:
        """Which parameters of which vectors leave their bounds."""
        X = X.detach().cpu() if isinstance(X, torch.Tensor) else X
        full = np.asarray(self.complete(X))
        lower, upper = self._bounds_array.T
        messages = []
        for i, row in enumerate(full):
            for j in np.flatnonzero((row < lower) | (row > upper)):
                side = "below" if row[j] < lower[j] else "above"
                messages.append(
                    f"sample {i}: {self.names[j]}={row[j]:.8g} is {side} "
                    f"[{lower[j]:.8g}, {upper[j]:.8g}]"
                )
                if len(messages) >= max_items:
                    return "; ".join(messages)
        return "; ".join(messages) or "all samples satisfy bounds"

    def as_dict(self, values: Sequence[float] | np.ndarray) -> dict[str, float]:
        """Name every value of one vector with all ``names``."""
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.size != len(self.names):
            raise ValueError(f"Expected {len(self.names)} values, got {values.size}.")
        return dict(zip(self.names, map(float, values)))


def latin_hypercube(
    specs: Specs, n: int, seed: int | np.random.Generator | None = None
) -> np.ndarray:
    """``n`` valid sampled parameter vectors from one Latin hypercube.

    Vectors whose implicit charges would leave their bounds are redrawn.
    Columns follow ``specs.explicit_names``.
    """
    if n < 0:
        raise ValueError("Number of samples must be non-negative.")
    lower, upper = specs.explicit_bounds.T
    if lower.size == 0:
        if n and not specs.is_valid(np.empty((n, 0))).all():
            raise RuntimeError("Fully constrained parameter values are invalid.")
        return np.empty((n, 0))
    sampler = LatinHypercube(lower.size, seed=seed)
    collected: list[np.ndarray] = []
    n_valid = failures = 0
    while n_valid < n:
        batch = sampler.random(max(2 * (n - n_valid), 1)) * (upper - lower) + lower
        valid = batch[specs.is_valid(batch)]
        failures = 0 if len(valid) else failures + 1
        if failures >= 1000:
            raise RuntimeError(
                "Failed to generate valid parameter samples within 1000 "
                "consecutive Latin-hypercube batches."
            )
        collected.append(valid)
        n_valid += len(valid)
    return np.vstack(collected)[:n] if collected else np.empty((0, lower.size))
