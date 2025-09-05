import sys
from pathlib import Path

from ..bff import BFFOptimizer
from ..structures import TrainData
from ..io.logs import Logger
from ..io.utils import load_yaml


def load_config(fn_config: str) -> dict:

    fn_config = Path(fn_config).resolve()
    config = load_yaml(fn_config)

    required_keys = ['fn_train', 'fn_specs']
    required_train_fn = ['inputs', 'outputs', 'outputs_ref', 'observations']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key in configuration: '{key}'")
        elif key == 'fn_train':
            if not isinstance(config[key], list):
                raise ValueError(
                    "'fn_train' must be a list of training data file paths.")
            for i, train_files in enumerate(config[key]):
                if not isinstance(train_files, dict):
                    raise ValueError(
                        "Each entry in 'fn_train' must be a dictionary of file paths.")
                for req_fn in required_train_fn:
                    if req_fn not in train_files:
                        raise ValueError(
                            f"Missing required key '{req_fn}'"
                            f" in training data entry {i}."
                        )
                    train_files[req_fn] = Path(train_files[req_fn]).resolve()
                    if not train_files[req_fn].exists():
                        raise FileNotFoundError(
                            f"File not found: {train_files[req_fn]}")
        elif key == 'fn_specs':
            config[key] = Path(config[key]).resolve()
            if not config[key].exists():
                raise FileNotFoundError(f"File not found: {config[key]}")

    fn_log = config.get('fn_log', './out.log')
    fn_log = Path(fn_log).resolve()
    if not fn_log.parent.exists():
        raise ValueError(f"Directory does not exist: {fn_log.parent}")
    config['fn_log'] = fn_log

    return config


def main(fn_config):

    config = load_config(fn_config)

    logger = Logger(name='optimize', fn_log=config['fn_log'])

    train_data = [TrainData(**files) for files in config['fn_train']]
    optimizer = BFFOptimizer(*train_data, specs=config['fn_specs'], logger=logger)

    optimizer.setup_LGP(**config.get('lgp', {}))
    optimizer.run(**config.get('mcmc', {}))


if __name__ == "__main__":
    fn_config = sys.argv[1]
    main(fn_config)
