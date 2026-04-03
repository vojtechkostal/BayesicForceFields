from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

PathLike = str | Path


@dataclass(frozen=True, slots=True)
class BiasSpec:
    """Bias specification for one simulation window.

    Parameters
    ----------
    kind
        Bias type. Supported values are ``"none"``, ``"colvars"``, and
        ``"plumed"``.
    colvars_file
        Path to the user-supplied COLVARS input file.
    plumed_file
        Path to the user-supplied PLUMED input file.
    """

    kind: str = "none"
    colvars_file: Path | None = None
    plumed_file: Path | None = None

    def __post_init__(self) -> None:
        colvars_file = None if self.colvars_file is None else Path(self.colvars_file)
        plumed_file = None if self.plumed_file is None else Path(self.plumed_file)

        if self.kind not in {"none", "colvars", "plumed"}:
            raise ValueError(
                f"Unsupported bias kind {self.kind!r}. "
                "Supported values are 'none', 'colvars', and 'plumed'."
            )
        if colvars_file is not None and plumed_file is not None:
            raise ValueError(
                "Bias specification must define at most one of 'colvars_file' "
                "and 'plumed_file'."
            )
        if self.kind == "colvars" and colvars_file is None:
            raise ValueError("COLVARS bias requires 'colvars_file'.")
        if self.kind == "plumed" and plumed_file is None:
            raise ValueError("PLUMED bias requires 'plumed_file'.")
        if self.kind == "none" and (
            colvars_file is not None or plumed_file is not None
        ):
            raise ValueError("Unbiased systems cannot define bias input files.")
        if colvars_file is not None and not colvars_file.exists():
            raise FileNotFoundError(f"COLVARS file not found: {colvars_file}")
        if plumed_file is not None and not plumed_file.exists():
            raise FileNotFoundError(f"PLUMED file not found: {plumed_file}")

        object.__setattr__(self, "colvars_file", colvars_file)
        object.__setattr__(self, "plumed_file", plumed_file)

    @classmethod
    def from_any(
        cls,
        value: "BiasSpec | Mapping[str, Any] | PathLike | None",
        *,
        base_dir: Path | None = None,
    ) -> "BiasSpec":
        """Normalize a raw config value into a ``BiasSpec``.

        Parameters
        ----------
        value
            Existing ``BiasSpec`` instance, raw mapping, or ``None``.
        base_dir
            Base directory used to resolve relative COLVARS file paths.

        Returns
        -------
        BiasSpec
            Normalized bias specification.
        """
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        if isinstance(value, (str, Path)):
            if base_dir is not None:
                value = Path(value)
                if not value.is_absolute():
                    value = (base_dir / value).resolve()
            return cls.load(value)
        if not isinstance(value, Mapping):
            raise ValueError(f"Invalid bias specification: {value!r}")

        colvars_raw = value.get("colvars_file", value.get("fn_colvars"))
        plumed_raw = value.get("plumed_file", value.get("fn_plumed"))
        colvars_file = None
        plumed_file = None
        if colvars_raw is not None:
            colvars_file = Path(colvars_raw)
            if base_dir is not None and not colvars_file.is_absolute():
                colvars_file = (base_dir / colvars_file).resolve()
        if plumed_raw is not None:
            plumed_file = Path(plumed_raw)
            if base_dir is not None and not plumed_file.is_absolute():
                plumed_file = (base_dir / plumed_file).resolve()

        if value.get("kind") is not None:
            kind = str(value["kind"])
        elif colvars_file is not None:
            kind = "colvars"
        elif plumed_file is not None:
            kind = "plumed"
        else:
            kind = "none"
        return cls(
            kind=kind,
            colvars_file=colvars_file,
            plumed_file=plumed_file,
        )

    @property
    def is_biased(self) -> bool:
        """Whether this specification defines any simulation bias."""
        return self.kind != "none"

    @property
    def input_file(self) -> Path | None:
        """Bias input file for the configured engine, if any."""
        return self.colvars_file if self.kind == "colvars" else self.plumed_file

    @property
    def input_filename(self) -> str | None:
        """Canonical filename used when copying a bias input file."""
        if self.kind == "colvars":
            return "bias.colvars.dat"
        if self.kind == "plumed":
            return "bias.plumed.dat"
        return None

    @classmethod
    def load(cls, fn_in: PathLike) -> "BiasSpec":
        """Load a bias specification from a direct bias-input file."""
        fn_in = Path(fn_in).resolve()
        name = fn_in.name
        if name.endswith(".colvars.dat"):
            return cls(kind="colvars", colvars_file=fn_in)
        if name.endswith(".plumed.dat"):
            return cls(kind="plumed", plumed_file=fn_in)
        raise ValueError(
            f"Unsupported bias file {fn_in}. Expected '.colvars.dat' "
            "or '.plumed.dat'."
        )
