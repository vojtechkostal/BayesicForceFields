from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ...domain.systems import validate_system_id
from ...io.utils import load_yaml
from .._shared.config import PathLike, _resolve_path, _strict_bool


@dataclass(frozen=True)
class LGPFitDatasetConfig:
    name: str
    fn_data: Path
    mean: Any = 0
    nuisance: float | None = None
    fn_model: Path | None = None


@dataclass(frozen=True)
class LGPFitOptionsConfig:
    model_dir: Path
    reuse_models: bool = True
    n_hyper_max: int = 200
    committee_size: int = 1
    test_fraction: float = 0.2
    device: str = "cuda"
    opt_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LGPFitConfig:
    fn_config: Path
    datasets: tuple[LGPFitDatasetConfig, ...]
    lgpfit: LGPFitOptionsConfig
    log: Path

    @classmethod
    def load(cls, fn_config: PathLike) -> "LGPFitConfig":
        fn_config = Path(fn_config).resolve()
        base_dir = fn_config.parent
        config = load_yaml(fn_config)
        if not isinstance(config, Mapping):
            raise ValueError("LGP-fit configuration must contain a mapping.")
        unknown_top = set(config) - {"datasets", "lgpfit", "log"}
        if unknown_top:
            raise ValueError(
                "LGP-fit configuration contains unsupported key(s): "
                + ", ".join(sorted(unknown_top))
            )
        for key in ("datasets", "lgpfit"):
            if key not in config:
                raise ValueError(f"Missing required configuration section: {key!r}.")

        datasets_raw = config["datasets"]
        if not isinstance(datasets_raw, Mapping) or not datasets_raw:
            raise ValueError("'datasets' must be a non-empty mapping.")
        options = config["lgpfit"]
        if not isinstance(options, Mapping):
            raise ValueError("'lgpfit' must be a mapping.")
        model_dir = _resolve_path(
            base_dir,
            options.get("model_dir", "./models"),
            must_exist=False,
            kind="model directory",
        )
        fixed_options = {
            "model_dir",
            "reuse_models",
            "n_hyper_max",
            "committee_size",
            "test_fraction",
            "device",
        }
        optimizer_options = {"lr", "max_iter", "tol_grad"}
        known = fixed_options | optimizer_options
        unknown_options = set(options) - known
        if unknown_options:
            raise ValueError(
                "lgpfit contains unsupported key(s): "
                + ", ".join(sorted(unknown_options))
            )
        lgpfit = LGPFitOptionsConfig(
            model_dir=model_dir,
            reuse_models=_strict_bool(
                options.get("reuse_models", True),
                field="lgpfit.reuse_models",
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
        if not 0 < lgpfit.test_fraction < 1:
            raise ValueError("'lgpfit.test_fraction' must be between 0 and 1.")
        if lgpfit.n_hyper_max < 1:
            raise ValueError("'lgpfit.n_hyper_max' must be positive.")
        if lgpfit.committee_size < 1:
            raise ValueError("'lgpfit.committee_size' must be positive.")

        datasets: list[LGPFitDatasetConfig] = []
        for raw_name, dataset in datasets_raw.items():
            name = validate_system_id(raw_name, field="datasets key")
            if not isinstance(dataset, Mapping):
                raise ValueError(f"Dataset {name!r} must be a mapping.")
            unknown = set(dataset) - {"data", "mean", "nuisance", "model"}
            if unknown:
                raise ValueError(
                    f"Dataset {name!r} contains unsupported key(s): "
                    + ", ".join(sorted(unknown))
                )
            if "data" not in dataset:
                raise ValueError(f"Dataset {name!r} is missing required key 'data'.")
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
                else _resolve_path(
                    base_dir,
                    dataset["model"],
                    must_exist=False,
                    kind=f"dataset {name!r} model file",
                )
            )
            datasets.append(
                LGPFitDatasetConfig(
                    name=str(name),
                    fn_data=_resolve_path(
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
            lgpfit=lgpfit,
            log=_resolve_path(
                base_dir,
                config.get("log", model_dir.parent / "lgpfit.log"),
                must_exist=False,
                kind="lgpfit log file",
            ),
        )
