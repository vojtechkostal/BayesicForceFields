from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ...domain.systems import validate_system_id
from ...io.utils import load_yaml
from ..config import PathLike, check_keys, resolve_path, strict_bool


@dataclass(frozen=True)
class FitLGPDatasetConfig:
    name: str
    fn_data: Path
    mean: Any = 0
    nuisance: float | None = None
    fn_model: Path | None = None


@dataclass(frozen=True)
class FitLGPOptionsConfig:
    model_dir: Path
    reuse_models: bool = True
    n_hyper_max: int = 200
    committee_size: int = 1
    test_fraction: float = 0.2
    device: str = "cuda"
    opt_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FitLGPConfig:
    fn_config: Path
    datasets: tuple[FitLGPDatasetConfig, ...]
    fit: FitLGPOptionsConfig
    log: Path

    @classmethod
    def load(cls, fn_config: PathLike) -> "FitLGPConfig":
        fn_config = Path(fn_config).resolve()
        base_dir = fn_config.parent
        config = check_keys(
            load_yaml(fn_config),
            where="Fit-LGP configuration",
            allowed={"datasets", "fit", "log"},
            required=("datasets", "fit"),
        )

        datasets_raw = config["datasets"]
        if not isinstance(datasets_raw, Mapping) or not datasets_raw:
            raise ValueError("'datasets' must be a non-empty mapping.")
        fixed_options = {
            "model_dir",
            "reuse_models",
            "n_hyper_max",
            "committee_size",
            "test_fraction",
            "device",
        }
        optimizer_options = {"lr", "max_iter", "tol_grad"}
        options = check_keys(
            config["fit"], where="fit", allowed=fixed_options | optimizer_options
        )
        model_dir = resolve_path(
            base_dir,
            options.get("model_dir", "./models"),
            must_exist=False,
            kind="model directory",
        )
        fit = FitLGPOptionsConfig(
            model_dir=model_dir,
            reuse_models=strict_bool(
                options.get("reuse_models", True),
                field="fit.reuse_models",
            ),
            n_hyper_max=int(options.get("n_hyper_max", 200)),
            committee_size=int(options.get("committee_size", 1)),
            test_fraction=float(options.get("test_fraction", 0.2)),
            device=str(options.get("device", "cuda")),
            opt_kwargs={
                key: value
                for key, value in options.items()
                if key in optimizer_options
            },
        )
        if not 0 < fit.test_fraction < 1:
            raise ValueError("'fit.test_fraction' must be between 0 and 1.")
        if fit.n_hyper_max < 1:
            raise ValueError("'fit.n_hyper_max' must be positive.")
        if fit.committee_size < 1:
            raise ValueError("'fit.committee_size' must be positive.")

        datasets: list[FitLGPDatasetConfig] = []
        for raw_name, dataset in datasets_raw.items():
            name = validate_system_id(raw_name, field="datasets key")
            dataset = check_keys(
                dataset,
                where=f"Dataset {name!r}",
                allowed={"data", "mean", "nuisance", "model"},
                required=("data",),
            )
            nuisance = dataset.get("nuisance")
            if nuisance is not None:
                nuisance = float(nuisance)
                if nuisance <= 0:
                    raise ValueError(
                        f"Dataset {name!r} nuisance must be a positive standard "
                        "deviation."
                    )
            fn_model = (
                model_dir / f"{name}.lgp"
                if dataset.get("model") is None
                else resolve_path(
                    base_dir,
                    dataset["model"],
                    must_exist=False,
                    kind=f"dataset {name!r} model file",
                )
            )
            datasets.append(
                FitLGPDatasetConfig(
                    name=str(name),
                    fn_data=resolve_path(
                        base_dir,
                        dataset["data"],
                        kind=f"dataset {name!r} data file",
                    ),
                    mean=dataset.get("mean", 0),
                    nuisance=nuisance,
                    fn_model=fn_model,
                )
            )
        return cls(
            fn_config=fn_config,
            datasets=tuple(datasets),
            fit=fit,
            log=resolve_path(
                base_dir,
                config.get("log", model_dir.parent / "fit-lgp.log"),
                must_exist=False,
                kind="fit-lgp log file",
            ),
        )
