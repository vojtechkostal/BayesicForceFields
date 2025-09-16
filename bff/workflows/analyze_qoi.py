import time
import numpy as np

from pathlib import Path

from ..qoi.analysis import analyze_trainset, analyze_all_trajectories, get_all_settings
from ..io.logs import Logger
from ..io.utils import load_yaml, save_json
from ..structures import QoI, TrainData, Specs


def load_config(fn_config: str) -> dict:

    fn_config = Path(fn_config).resolve()
    config = load_yaml(fn_config)

    required_keys = ['aimd', 'ffmd']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key in configuration: {key}")

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

    if not config['ffmd'].get('trainset_dir'):
        raise ValueError("Missing 'trainset_dir' in FFMD configuration.")
    config['ffmd']['trainset_dir'] = Path(config['ffmd']['trainset_dir']).resolve()
    if not config['ffmd']['trainset_dir'].exists():
        raise FileNotFoundError(f"Trainset directory not found: {config['ffmd']['trainset_dir']}")

    if 'results_dir' not in config:
        config['results_dir'] = fn_config.parent
    config['results_dir'] = Path(config['results_dir']).resolve()
    config['results_dir'].mkdir(parents=True, exist_ok=True)

    if 'base_name' not in config:
        config['base_name'] = 'qoi'
    
    if 'fn_log' not in config:
        config['fn_log'] = 'out.log'
    config['fn_log'] = config['results_dir'] / config['fn_log']

    return config


def _flatten_qoi(
    sample: list[QoI], valid_qoi: list[str], valid_hb: np.ndarray[str] = None
) -> dict[str, list]:
    """Group QoI values without flattening."""

    blocks = {attr: [] for attr in valid_qoi}
    for s in sample:
        for attr in valid_qoi:
            if not hasattr(s, attr):
                continue
            val = getattr(s, attr)
            if attr in {"rdf", "restr"}:
                for _, (_, y) in val.items():
                    blocks[attr].extend(y)
            elif attr == "hb":
                blocks[attr].extend(s.hb.get(hb_name, 0) for hb_name in valid_hb)
            else:
                blocks[attr].extend(val)
    return blocks


def _infer_observations(
    qoi: list[QoI], valid_hb: np.ndarray[str] = None
) -> dict[str, int]:
    observations = {}
    for i in qoi:
        for name, n in i.observations.items():
            observations[name] = observations.get(name, 0) + n

    #if "hb" in observations:
    #    observations["hb"] = len(valid_hb)

    if "restr" in observations and "rdf" in observations:
        observations["restr"] = observations["rdf"]

    return observations


def main(fn_config: str) -> None:

    config = load_config(fn_config)
    logger = Logger(name='analyze_qoi', fn_log=config['fn_log'])

    logger.info("Analysing Quantities of Interest (QoI)", level=0)
    logger.info("======================================\n", level=0)

    settings = get_all_settings(config['settings'])

    qoi_train, trainset_info = analyze_trainset(
        **config['ffmd'], **settings, logger=logger,
    )

    logger.info("", level=0)
    logger.info("Reference QoI: in progress...", level=1, overwrite=True)
    t0 = time.time()
    qoi_ref = analyze_all_trajectories(
        **config['aimd'],
        **settings,
        restraints=trainset_info.restraints,
        mol_resname=trainset_info.specs['mol_resname'],
    )
    t1 = time.time()
    logger.info(f"Reference QoI: Done. ({t1 - t0:.1f} s)", level=1, overwrite=True)

    valid_hb = np.unique([name for s in qoi_ref if hasattr(s, "hb") for name in s.hb])
    valid_qoi = {name for trj in qoi_ref for name in trj.names}

    y_true = {
        qoi: np.array(values)
        for qoi, values in _flatten_qoi(qoi_ref, valid_qoi, valid_hb).items()
    }

    y_train = {
        qoi: np.array([
            _flatten_qoi(sample, valid_qoi, valid_hb)[qoi]
            for sample in qoi_train
        ])
        for qoi in valid_qoi
    }

    observations = _infer_observations(qoi_ref, valid_hb)

    train_data = TrainData(
        X=trainset_info.inputs,
        y=y_train,
        y_ref=y_true,
        observations=observations,
        settings=settings
    )

    fn_base = config['results_dir'] / config['base_name']
    train_data.write(fn_base)
    Specs(trainset_info.specs).save(fn_base.with_name(fn_base.name + '-specs.yml'))
    if config.get('write_raw_qoi', False):
        fn_qoi_train = fn_base.with_name(fn_base.name + '-train.raw.json')
        fn_qoi_ref = fn_base.with_name(fn_base.name + '-ref.raw.json')

        qoi_train_dict = [[trj.__dict__ for trj in sample] for sample in qoi_train]
        qoi_ref_dict = [trj.__dict__ for trj in qoi_ref]

        save_json(qoi_train_dict, fn_qoi_train)
        save_json(qoi_ref_dict, fn_qoi_ref)
