import sys
from pathlib import Path
from typing import Union

from ..bff import BFFLearner
from ..structures import QoIDataset, ChargeConstraint
from ..io.logs import Logger
from ..io.utils import load_yaml


PathLike = Union[str, Path]


def load_config(fn_config: PathLike) -> dict:
    fn_config = Path(fn_config).resolve()
    config = load_yaml(fn_config)
    base_dir = fn_config.parent

    required_keys = ["fn_specs", "datasets"]

    def resolve_and_check(path: PathLike) -> Path:
        """Resolve a path relative to base_dir and ensure it exists."""
        path = (base_dir / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return path

    # --- validate required keys ---
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise ValueError(
            "Missing required key(s) in configuration: "
            f"{', '.join(repr(k) for k in missing)}"
        )

    datasets = config.get("datasets")
    if not datasets:
        raise ValueError("Missing 'qoi' in 'lgp_train' configuration.")

    for qoi_name, qoi_data in datasets.items():
        try:
            qoi_data["data"] = resolve_and_check(qoi_data["data"])
            qoi_data["mean"] = qoi_data["mean"]
        except KeyError as exc:
            raise ValueError(
                f"Missing {exc.args[0]!r} for QoI '{qoi_name}' in 'lgp_train.qoi'."
            ) from None

    # --- validate fn_specs ---
    config["fn_specs"] = resolve_and_check(config["fn_specs"])

    # --- validate log file directory ---
    fn_log = config.get("fn_log", "./out.log")
    fn_log = (base_dir / fn_log).resolve()
    if not fn_log.parent.exists():
        raise ValueError(f"Directory does not exist: {fn_log.parent}")
    config["fn_log"] = fn_log

    return config


def main(fn_config: PathLike) -> None:

    config = load_config(fn_config)
    logger = Logger("BFF", config.get("fn_log"))

    # train LGP surrogates for the requested QoIs
    datasets = [
        QoIDataset.load(qoi["data"])
        for qoi in config["datasets"].values()
    ]
    qoi = config.get("qoi", None)

    charge_constraint = ChargeConstraint(config["fn_specs"])

    learner = BFFLearner(*datasets, logger=logger)
    learner.train(**config.get("lgp_train", {}))
    learner.run(qoi=qoi, constraint=charge_constraint, **config.get("mcmc", {}))

if __name__ == "__main__":
    fn_config = sys.argv[1]
    main(fn_config)
