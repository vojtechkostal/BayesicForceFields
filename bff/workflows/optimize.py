import sys
import os

from ..bff import Optimizer

from pathlib import Path
from ..io.utils import load_yaml


def load_config(fn_config: str) -> dict:

    fn_config = Path(fn_config).resolve()
    config = load_yaml(fn_config)

    required_keys = ['train_dir', 'results_dir', 'aimd', 'ffmd', 'QoI', 'lgp', 'mcmc']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key in configuration: {key}")
        if key in 'train_dir':
            train_dir = Path(config[key]).resolve()
            if not train_dir.exists():
                raise FileNotFoundError(f"Training directory not found: {train_dir}")
            config[key] = train_dir
        elif key == 'results_dir':
            results_dir = Path(config[key]).resolve()
            config[key] = results_dir

    fn_log = config.get('fn_log', 'out.log')
    config['fn_log'] = Path(fn_log).resolve()

    aimd_keys = ['fn_coord', 'fn_topol', 'fn_trj']
    for key in aimd_keys:
        resolved_paths = []
        if key not in config['aimd']:
            raise ValueError(f"Missing required key in AIMD configuration: {key}")
        for fn in config['aimd'][key]:
            resolved_path = Path(fn).resolve()
            if not resolved_path.exists():
                raise FileNotFoundError(f"File not found: {resolved_path}")
            resolved_paths.append(resolved_path)
    aimd_lengths = [len(config['aimd'][key]) for key in aimd_keys]
    if len(set(aimd_lengths)) != 1:
        raise ValueError("AIMD configuration lists must have the same length.")

    if config['ffmd'].get('fn_in'):
        fn_in = Path(config['ffmd']['fn_in']).resolve()
        if not fn_in.exists():
            raise FileNotFoundError(f"File not found: {fn_in}")
        config['ffmd']['fn_in'] = fn_in
    if config['ffmd'].get('fn_out'):
        fn_out = config['results_dir'] / config['ffmd']['fn_out']
        config['ffmd']['fn_out'] = fn_out

    if config['lgp'].get('fn_hyper'):
        keys = ['rdf', 'hb', 'restr']
        for key in keys:
            if config['lgp']['fn_hyper'].get(key):
                fn = Path(config['lgp']['fn_hyper'][key]).resolve()
                config['lgp']['fn_hyper'][key] = fn

    return config


def main(fn_config):

    os.environ["OMP_NUM_THREADS"] = "1"

    # Load the configuration file
    config = load_config(fn_config)

    # Modify relative to absolute paths
    config['results_dir'].mkdir(parents=True, exist_ok=True)

    # Initialize the sampler
    optimizer = Optimizer(config['train_dir'], fn_log=config['fn_log'])

    # Load reference and training set -> evaluate the training set
    optimizer.load_train(**config['ffmd'], **config['settings'])
    optimizer.load_reference(**config['aimd'], **config['settings'])

    # Train the model
    optimizer.setup_lgp(**config['lgp'])

    # Run the Bayesian inference
    optimizer.run(**config['mcmc'])


if __name__ == "__main__":
    fn_config = sys.argv[1]
    main(fn_config)
