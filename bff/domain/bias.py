from dataclasses import dataclass
from pathlib import Path

PathLike = str | Path


@dataclass(frozen=True, slots=True)
class BiasSpec:
    """Bias specification for one simulation system.

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
