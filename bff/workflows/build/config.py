from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ...domain.bias import BiasSpec
from ...domain.systems import validate_system_id, validate_unique_system_ids
from ..config import ConfigSection, PathLike, load_config, resolve_path


@dataclass(frozen=True)
class BuildSystemConfig:
    system_id: str
    system_name: str | None
    topology_path: Path
    templates: dict[str, Path]
    box: list[float] | None
    bias: BiasSpec
    nsteps_npt: int
    nsteps_prod: int
    mdp_em_path: Path
    mdp_npt_path: Path
    mdp_production_path: Path


def _load_box(system: ConfigSection) -> list[float] | None:
    box = system.get("box")
    if box is None:
        return None
    if (
        not isinstance(box, list)
        or len(box) not in {3, 6}
        or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in box
        )
        or any(value <= 0 for value in box)
    ):
        raise ValueError(
            f"{system.field('box')} must be 3 or 6 positive numbers, got {box!r}."
        )
    return [float(value) for value in (box if len(box) == 6 else [*box, 90, 90, 90])]


def _load_bias(system: ConfigSection) -> BiasSpec:
    bias = system.section("bias", allowed=("colvars_file", "plumed_file"))
    if "colvars_file" in bias and "plumed_file" in bias:
        raise ValueError(
            f"{system.field('bias')} must define colvars_file or plumed_file, "
            "not both."
        )
    if "colvars_file" in bias:
        return BiasSpec(kind="colvars", colvars_file=bias.path("colvars_file"))
    if "plumed_file" in bias:
        return BiasSpec(kind="plumed", plumed_file=bias.path("plumed_file"))
    return BiasSpec()


@dataclass(frozen=True)
class BuildConfig:
    fn_config: Path
    project_dir: Path
    gmx_cmd: str
    fn_log: Optional[Path]
    systems: list[BuildSystemConfig]

    @classmethod
    def load(cls, fn_config: PathLike) -> BuildConfig:
        config = load_config(
            fn_config,
            stage="build",
            allowed=("project", "gromacs", "systems"),
            required=("project", "gromacs", "systems"),
        )
        project = config.section(
            "project", allowed=("directory", "log"), required=("directory",)
        )
        project_dir = project.path("directory", must_exist=False)
        gromacs = config.section("gromacs", allowed=("command",), required=("command",))

        systems: list[BuildSystemConfig] = []
        for system in config.sections(
            "systems",
            allowed=(
                "system_id",
                "system_name",
                "topology",
                "templates",
                "box",
                "bias",
                "nsteps",
                "mdp",
            ),
            required=("system_id", "topology", "nsteps", "mdp"),
        ):
            nsteps = system.section(
                "nsteps", allowed=("npt", "prod"), required=("npt", "prod")
            )
            mdp = system.section(
                "mdp", allowed=("em", "npt", "prod"), required=("em", "npt", "prod")
            )
            systems.append(
                BuildSystemConfig(
                    system_id=validate_system_id(
                        system.get("system_id"), field=system.field("system_id")
                    ),
                    system_name=system.string("system_name", None),
                    topology_path=system.path("topology"),
                    templates={
                        str(residue): resolve_path(
                            config.base_dir,
                            path,
                            field=f"{system.field('templates')}.{residue}",
                        )
                        for residue, path in system.mapping("templates", {}).items()
                    },
                    box=_load_box(system),
                    bias=_load_bias(system),
                    nsteps_npt=nsteps.integer("npt", minimum=0),
                    nsteps_prod=nsteps.integer("prod", minimum=1),
                    mdp_em_path=mdp.path("em"),
                    mdp_npt_path=mdp.path("npt"),
                    mdp_production_path=mdp.path("prod"),
                )
            )
        validate_unique_system_ids(
            [system.system_id for system in systems], field="systems"
        )

        return cls(
            fn_config=Path(fn_config).resolve(),
            project_dir=project_dir,
            gmx_cmd=gromacs.string("command"),
            fn_log=project.path("log", project_dir / "build.log", must_exist=False),
            systems=systems,
        )
