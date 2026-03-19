import sys
from pathlib import Path
from typing import Union

from ..bff import train_lgp, learn
from ..structures import TrainData
from ..io.logs import Logger
from ..io.utils import load_yaml


PathLike = Union[str, Path]


def load_config(fn_config: PathLike) -> dict:
    fn_config = Path(fn_config).resolve()
    config = load_yaml(fn_config)
    base_dir = fn_config.parent

    required_keys = ["fn_specs", 'lgp_train', 'mcmc', 'fn_train']
    required_train_fn = ["inputs", "outputs", "outputs_ref", "observations"]

    def resolve_and_check(path: PathLike) -> Path:
        """Resolve a path relative to base_dir and ensure it exists."""
        path = (base_dir / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return path

    # --- validate required keys ---
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key in configuration: '{key}'")

    # --- validate training data entries ---
    fn_train = config["fn_train"]
    if not isinstance(fn_train, list):
        raise ValueError("'fn_train' must be a list of training data file paths.")

    for i, train_files in enumerate(fn_train):
        if not isinstance(train_files, dict):
            raise ValueError(
                f"Training entry {i} must be a dictionary of file paths."
            )
        for req_fn in required_train_fn:
            if req_fn not in train_files:
                raise ValueError(
                    f"Missing required key '{req_fn}' in training data entry {i}."
                )

            if req_fn in ["inputs", "settings", "observations"]:
                train_files[req_fn] = resolve_and_check(train_files[req_fn])
            else:  # dict of QoI → file path
                for qoi, fn in train_files[req_fn].items():
                    train_files[req_fn][qoi] = resolve_and_check(fn)

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

    logger = Logger(name='learn', fn_log=config['fn_log'])

    # train LGP surrogates for the requested QoIs
    train_datasets = config['fn_train']
    train_data = [TrainData(**files) for files in train_datasets]
    models = train_lgp(*train_data, **config.get('lgp_train', {}), logger=logger)

    # run MCMC sampling to learn the posterior distribution over parameters
    y_ref = {
        qoi: dataset.outputs_ref[qoi]
        for dataset in train_data
        for qoi in dataset.outputs_ref
    }
    mcmc = learn(y_ref=y_ref, models=models, **config.get('mcmc', {}), logger=logger)

    if 'fn_priors' in config.get('mcmc', {}):
        fn_priors = config['mcmc']['fn_priors']
        mcmc.write_priors(fn_priors)
    if 'fn_tau' in config.get('mcmc', {}):
        fn_tau = config['mcmc']['fn_tau']
        mcmc.write_tau(fn_tau)


if __name__ == "__main__":
    fn_config = sys.argv[1]
    main(fn_config)
