"""Typed reading of stage configuration files.

Every stage loader reads its YAML through :class:`ConfigSection`, so all
stages reject unknown keys, treat ``null`` like an omitted key, check types
and ranges the same way, and name the offending key in every error, for
example ``mcmc.warmup must be an integer >= 0, got -1``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Collection, Iterable, Mapping
from pathlib import Path
from typing import Any

from ..io.utils import load_yaml

PathLike = str | Path
REQUIRED: Any = object()


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_number(value: Any) -> Any:
    """Numbers unchanged, numeric strings as floats, anything else unchanged.

    PyYAML reads exponents without a dot, such as ``1e-3``, as strings.
    """
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    return value


class ConfigSection:
    """One mapping of a configuration file and its location for messages."""

    def __init__(
        self,
        raw: Any,
        where: str,
        *,
        base_dir: Path,
        allowed: Iterable[str],
        required: Iterable[str] = (),
        label: str | None = None,
    ) -> None:
        # ``where`` prefixes key names in messages ("" at the top level);
        # ``label`` names the mapping itself.
        self.where = where
        self.base_dir = base_dir
        label = label or where
        if not isinstance(raw, Mapping):
            raise ValueError(f"{label} must be a mapping, got {raw!r}.")
        unknown = set(raw) - set(allowed)
        if unknown:
            raise ValueError(
                f"{label} contains unsupported key(s): "
                + ", ".join(sorted(map(str, unknown)))
            )
        self.raw = dict(raw)
        missing = [key for key in required if key not in self]
        if missing:
            raise ValueError(
                f"{label} is missing required key(s): "
                + ", ".join(repr(key) for key in missing)
            )

    def __contains__(self, key: str) -> bool:
        return self.raw.get(key) is not None

    def field(self, key: str) -> str:
        return f"{self.where}.{key}" if self.where else key

    def _value(self, key: str, default: Any) -> tuple[Any, bool]:
        """The raw value and whether it was given (``null`` means omitted)."""
        value = self.raw.get(key)
        if value is not None:
            return value, True
        if default is REQUIRED:
            raise ValueError(f"{self.field(key)} is required.")
        return default, False

    def get(self, key: str, default: Any = None) -> Any:
        """The raw value of an untyped key."""
        return self._value(key, default)[0]

    def integer(
        self,
        key: str,
        default: Any = REQUIRED,
        *,
        minimum: int | None = None,
        special: Collection[int] = (),
    ) -> Any:
        """An integer ``>= minimum``; values in ``special`` are also accepted."""
        value, given = self._value(key, default)
        if not given:
            return value
        value = _as_number(value)
        if _is_number(value) and math.isfinite(value) and float(value).is_integer():
            value = int(value)
            if value in special or minimum is None or value >= minimum:
                return value
        expected = "an integer" + ("" if minimum is None else f" >= {minimum}")
        if special:
            expected += " or " + " or ".join(map(str, special))
        raise ValueError(f"{self.field(key)} must be {expected}, got {value!r}.")

    def number(
        self,
        key: str,
        default: Any = REQUIRED,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
        exclusive: bool = False,
    ) -> Any:
        """A finite number within the bounds (exclusive bounds if requested)."""
        value, given = self._value(key, default)
        if not given:
            return value
        value = _as_number(value)
        if _is_number(value) and math.isfinite(value):
            number = float(value)
            above = minimum is None or (
                number > minimum if exclusive else number >= minimum
            )
            below = maximum is None or (
                number < maximum if exclusive else number <= maximum
            )
            if above and below:
                return number
        bounds = []
        if minimum is not None:
            bounds.append(f"{'>' if exclusive else '>='} {minimum:g}")
        if maximum is not None:
            bounds.append(f"{'<' if exclusive else '<='} {maximum:g}")
        expected = "a finite number" + (" " + " and ".join(bounds) if bounds else "")
        raise ValueError(f"{self.field(key)} must be {expected}, got {value!r}.")

    def boolean(self, key: str, default: Any = REQUIRED) -> Any:
        value, given = self._value(key, default)
        if given and not isinstance(value, bool):
            raise ValueError(
                f"{self.field(key)} must be true or false, got {value!r}."
            )
        return value

    def string(
        self,
        key: str,
        default: Any = REQUIRED,
        *,
        choices: Collection[str] | None = None,
    ) -> Any:
        """A non-empty string, optionally one of ``choices``."""
        value, given = self._value(key, default)
        if not given:
            return value
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{self.field(key)} must be a non-empty string, got {value!r}."
            )
        if choices is not None and value not in choices:
            raise ValueError(
                f"{self.field(key)} must be one of "
                + ", ".join(repr(choice) for choice in sorted(choices))
                + f"; got {value!r}."
            )
        return value

    def device(self, key: str, default: Any = REQUIRED) -> Any:
        """A PyTorch device: ``auto``, ``cpu``, ``cuda``, ``cuda:<index>``, or
        ``mps``."""
        value = self.string(key, default)
        if value is not None and not re.fullmatch(
            r"auto|cpu|cuda(:\d+)?|mps", value
        ):
            raise ValueError(
                f"{self.field(key)} must be auto, cpu, cuda, cuda:<index>, or "
                f"mps; got {value!r}."
            )
        return value

    def strings(self, key: str, default: Any = REQUIRED) -> Any:
        """A list of non-empty strings, returned as a tuple."""
        value, given = self._value(key, default)
        if not given:
            return value
        if not isinstance(value, list) or not all(
            isinstance(item, str) and item for item in value
        ):
            raise ValueError(
                f"{self.field(key)} must be a list of strings, got {value!r}."
            )
        return tuple(value)

    def path(
        self,
        key: str,
        default: Any = REQUIRED,
        *,
        must_exist: bool = True,
    ) -> Any:
        """A path relative to the configuration file's directory."""
        value = self._value(key, default)[0]
        if value is None:
            return None
        return resolve_path(
            self.base_dir, value, must_exist=must_exist, field=self.field(key)
        )

    def mapping(self, key: str, default: Any = REQUIRED) -> Any:
        """A free-form mapping, such as Slurm ``sbatch`` options."""
        value, given = self._value(key, default)
        if given and not isinstance(value, Mapping):
            raise ValueError(f"{self.field(key)} must be a mapping, got {value!r}.")
        return dict(value) if isinstance(value, Mapping) else value

    def section(
        self,
        key: str,
        *,
        allowed: Iterable[str],
        required: Iterable[str] = (),
    ) -> ConfigSection:
        """A nested mapping; an omitted one is read as empty."""
        return ConfigSection(
            self.raw.get(key) or {},
            self.field(key),
            base_dir=self.base_dir,
            allowed=allowed,
            required=required,
        )

    def sections(
        self,
        key: str,
        *,
        allowed: Iterable[str],
        required: Iterable[str] = (),
    ) -> list[ConfigSection]:
        """A non-empty list of mappings."""
        value = self._value(key, REQUIRED)[0]
        if not isinstance(value, list) or not value:
            raise ValueError(f"{self.field(key)} must be a non-empty list.")
        allowed = tuple(allowed)
        return [
            ConfigSection(
                item,
                f"{self.field(key)}[{index}]",
                base_dir=self.base_dir,
                allowed=allowed,
                required=required,
            )
            for index, item in enumerate(value)
        ]

    def named_sections(
        self,
        key: str,
        *,
        allowed: Iterable[str],
        required: Iterable[str] = (),
    ) -> dict[str, ConfigSection]:
        """A non-empty mapping from names to mappings."""
        value = self._value(key, REQUIRED)[0]
        if not isinstance(value, Mapping) or not value:
            raise ValueError(f"{self.field(key)} must be a non-empty mapping.")
        names = [name for name in value if not isinstance(name, str) or not name]
        if names:
            raise ValueError(
                f"{self.field(key)} names must be non-empty strings, got {names!r}."
            )
        allowed = tuple(allowed)
        return {
            str(name): ConfigSection(
                item,
                f"{self.field(key)}.{name}",
                base_dir=self.base_dir,
                allowed=allowed,
                required=required,
            )
            for name, item in value.items()
        }


def load_config(
    fn_config: PathLike,
    *,
    stage: str,
    allowed: Iterable[str],
    required: Iterable[str] = (),
) -> ConfigSection:
    """Read a stage configuration file; paths resolve from its directory."""
    fn_config = Path(fn_config).resolve()
    return ConfigSection(
        load_yaml(fn_config),
        "",
        base_dir=fn_config.parent,
        allowed=allowed,
        required=required,
        label=f"{stage} configuration {fn_config}",
    )


def resolve_path(
    base_dir: Path,
    path: PathLike,
    *,
    must_exist: bool = True,
    field: str = "path",
) -> Path:
    """Resolve ``path`` relative to the configuration directory."""
    if not isinstance(path, (str, Path)) or not str(path):
        raise ValueError(f"{field} must be a path, got {path!r}.")
    resolved = (base_dir / Path(path).expanduser()).resolve()
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"{field}: {resolved} does not exist.")
    return resolved
