from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...domain.systems import validate_system_id
from ...qoi.routines import load_custom_routine
from ..config import PathLike, load_config


@dataclass(frozen=True)
class FitLGPDatasetConfig:
    name: str
    fn_data: Path
    mean: str | float = "data"
    nuisance: float | None = None
    fn_model: Path | None = None


@dataclass(frozen=True)
class FitLGPOptionsConfig:
    model_dir: Path
    reuse_models: bool = True
    n_hyper_max: int = 200
    committee_size: int = 1
    test_fraction: float = 0.2
    opt_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FitLGPConfig:
    fn_config: Path
    datasets: tuple[FitLGPDatasetConfig, ...]
    fit: FitLGPOptionsConfig
    log: Path

    @classmethod
    def load(cls, fn_config: PathLike) -> FitLGPConfig:
        config = load_config(
            fn_config,
            stage="fit-lgp",
            allowed=("datasets", "fit", "log"),
            required=("datasets",),
        )
        options = config.section(
            "fit",
            allowed=(
                "model_dir",
                "reuse_models",
                "n_hyper_max",
                "committee_size",
                "test_fraction",
                "max_iter",
                "tol_grad",
            ),
        )
        model_dir = options.path("model_dir", "./models", must_exist=False)
        opt_kwargs = {
            "max_iter": options.integer("max_iter", None, minimum=1),
            "tol_grad": options.number("tol_grad", None, minimum=0, exclusive=True),
        }
        fit = FitLGPOptionsConfig(
            model_dir=model_dir,
            reuse_models=options.boolean("reuse_models", True),
            n_hyper_max=options.integer("n_hyper_max", 200, minimum=1),
            committee_size=options.integer("committee_size", 1, minimum=1),
            test_fraction=options.number(
                "test_fraction", 0.2, minimum=0, maximum=1, exclusive=True
            ),
            opt_kwargs={k: v for k, v in opt_kwargs.items() if v is not None},
        )

        datasets: list[FitLGPDatasetConfig] = []
        for name, dataset in config.named_sections(
            "datasets",
            allowed=("data", "mean", "nuisance", "model"),
            required=("data",),
        ).items():
            validate_system_id(name, field=dataset.where)
            mean = dataset.get("mean", "data")
            if isinstance(mean, str) and ":" in mean:
                # A custom mean function; import it now so errors show early.
                mean = load_custom_routine(mean, config.base_dir)[0]
            elif mean not in ("data", "sigmoid") and not (
                isinstance(mean, (int, float)) and not isinstance(mean, bool)
            ):
                raise ValueError(
                    f"{dataset.field('mean')} must be 'data', 'sigmoid', a number, "
                    f"or 'path/to/file.py:function'; got {mean!r}."
                )
            datasets.append(
                FitLGPDatasetConfig(
                    name=name,
                    fn_data=dataset.path("data"),
                    mean=mean,
                    nuisance=dataset.number(
                        "nuisance", None, minimum=0, exclusive=True
                    ),
                    fn_model=dataset.path(
                        "model", model_dir / f"{name}.lgp", must_exist=False
                    ),
                )
            )
        return cls(
            fn_config=Path(fn_config).resolve(),
            datasets=tuple(datasets),
            fit=fit,
            log=config.path("log", model_dir.parent / "fit-lgp.log", must_exist=False),
        )
