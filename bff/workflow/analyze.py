import sys
import os

from ..amo import AMO
from sklearn.preprocessing import StandardScaler

from pathlib import Path
from ..io.utils import load_yaml


def main(fn_config):

    os.environ["OMP_NUM_THREADS"] = "1"

    # Load the configuration file
    fn_config = Path(fn_config).resolve()
    config = load_yaml(fn_config)
    main_dir = fn_config.parent

    # Modify relative to absolute paths
    config['fn_train_set'] = [main_dir / fn for fn in config['fn_train_set']]
    config['results_dir'] = main_dir / config['results_dir']
    config['results_dir'].mkdir(parents=True, exist_ok=True)
    config['aimd']['fn_coord'] = [main_dir /
                                  fn for fn in config['aimd']['fn_coord']]
    config['aimd']['fn_topol'] = [main_dir /
                                  fn for fn in config['aimd']['fn_topol']]
    config['aimd']['fn_trjs'] = [main_dir /
                                 fn for fn in config['aimd']['fn_trjs']]
    config['ffmd']['fn_in'] = (
        main_dir /
        config['ffmd']['fn_in'] if config['ffmd'].get('fn_in') else None
    )
    config['ffmd']['fn_out'] = (
        main_dir / config['ffmd']['fn_out']
        if config['ffmd'].get('fn_out')
        else None
    )

    # Initialize the sampler
    optimizer = AMO(
        *config['fn_train_set'],
        fn_log=str(config['results_dir'] / 'analyze.log')
    )

    # Load reference and training set -> evaluate the training set
    optimizer.load_reference(**config['aimd'], **config['settings'])
    if config['ffmd'].get('fn_in', None) is not None:
        optimizer.load_train_set(fn_in=config['ffmd']['fn_in'])
    else:
        optimizer.load_train_set(
            workers=config['ffmd']['workers'],
            fn_out=config['ffmd']['fn_out'],
            progress_stride=config['ffmd']['progress_stride'],
            **config['settings'])

    optimizer.eval_train_set(*config['features'])

    # Train the model
    if config.get('rf', None) is not None:
        scaler_in, scaler_out = StandardScaler(), StandardScaler()
        optimizer.train(config['rf']['n_trees'], scaler_in, scaler_out)

    # Define the file names
    if config.get('inference', None) is not None:
        results_dir = Path(config['results_dir'])
        fn_backend = Path('./backend.h5').resolve()
        fn_priors = results_dir / 'priors.yaml'
        fn_tau = results_dir / 'tau.npy'
        fn_specs = results_dir / 'specs.yaml'
        if not results_dir.exists():
            results_dir.mkdir(parents=True)

        optimizer.specs.save(fn_specs)

        # Run the Bayesian inference
        optimizer.run(
            **config['inference'],
            fn_backend=fn_backend,
            fn_priors=fn_priors,
            fn_tau=fn_tau,
            fn_specs=fn_specs)


if __name__ == "__main__":
    fn_config = sys.argv[1]
    main(fn_config)
