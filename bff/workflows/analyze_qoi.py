import time
import numpy as np

from pathlib import Path
from typing import List

from ..qoi.analysis import analyze_trainset, analyze_all_trajectories, get_all_settings
from ..io.logs import Logger
from ..io.utils import load_yaml, save_json
from ..structures import QoI, QoIDataset


def load_config(fn_config: str) -> dict:

    fn_config = Path(fn_config).resolve()
    base_dir = fn_config.parent
    config = load_yaml(fn_config)

    required_keys = ['aimd', 'ffmd']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key in configuration: {key}")

    aimd_keys = ['fn_coord', 'fn_topol', 'fn_trj']
    for key in aimd_keys:
        if key not in config['aimd']:
            raise ValueError(f"Missing required key in AIMD configuration: {key}")
        for i, fn in enumerate(config['aimd'][key]):
            resolved_path = (base_dir / fn).resolve()
            if not resolved_path.exists():
                raise FileNotFoundError(f"AIMD file not found: {resolved_path}")

    aimd_lengths = [len(config['aimd'][key]) for key in aimd_keys]
    if len(set(aimd_lengths)) != 1:
        raise ValueError("AIMD configuration lists must have the same length.")

    if not config['ffmd'].get('trainset_dir'):
        raise ValueError("Missing 'trainset_dir' in FFMD configuration.")
    trainset_dir = base_dir / config['ffmd']['trainset_dir']
    config['ffmd']['trainset_dir'] = trainset_dir.resolve()
    if not config['ffmd']['trainset_dir'].exists():
        raise FileNotFoundError(
            f"Trainset directory not found: {config['ffmd']['trainset_dir']}")

    if 'base_name' not in config:
        config['base_name'] = './qoi'
    config['base_name'] = (base_dir / config['base_name']).resolve()

    if 'fn_log' not in config:
        config['fn_log'] = 'out.log'
    config['fn_log'] = (base_dir / config['fn_log']).resolve()

    return config


def _get_sample_qoi(
    sample: List[QoI], qoi_name: str, valid_hb: np.ndarray[str] = None
) -> List[float]:
    sample_values = []
    for trj in sample:
        qoi_values = getattr(trj, qoi_name, None)
        if qoi_values is None:
            raise ValueError(
                f"QoI '{qoi_name}' not found in training sample trajectory."
            )
        elif qoi_name in {"rdf", "restr"}:
            for _, (_, y) in qoi_values.items():
                sample_values.extend(y)
        elif qoi_name == "hb":
            for hb_name in valid_hb:
                sample_values.append(trj.hb.get(hb_name, 0))
        else:
            pass  # TODO: handle other QoI types if needed
            sample_values.extend(qoi_values)

    return sample_values


def main(fn_config: str) -> None:

    config = load_config(fn_config)
    logger = Logger(name='analyze_qoi', fn_log=config['fn_log'])

    logger.info("=== Quantities of Interest (QoI) Analysis ===\n", level=0)

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
    logger.info("", level=0)

    valid_hb = np.unique([name for s in qoi_ref if hasattr(s, "hb") for name in s.hb])
    valid_qoi = {name for trj in qoi_ref for name in trj.names}

    logger.info("Saving QoI data: in progress...", level=1, overwrite=True)
    for qoi_name in valid_qoi:
        output_ref = _get_sample_qoi(qoi_ref, qoi_name, valid_hb)
        output_train = [
            _get_sample_qoi(sample, qoi_name, valid_hb) for sample in qoi_train
        ]

        n_obs = sum(len(getattr(trj, qoi_name, [])) for trj in qoi_ref)

        data = QoIDataset(
            name=qoi_name,
            inputs=trainset_info.inputs,
            outputs=output_train,
            outputs_ref=output_ref,
            n_observations=n_obs,
            nuisance=None,
            settings=settings
        )

        # save the data
        fn_base = config['base_name']
        fn_out = fn_base.with_name(config['base_name'].name + f'-{qoi_name}.pt')
        data.write(fn_out)

    t2 = time.time()
    logger.info(f"Saving QoI data: Done. ({t2 - t1:.1f} s)", level=1, overwrite=True)
    logger.info("", level=0)

    # save raw data
    if config.get('write_raw_qoi', False):
        logger.info("Saving raw QoI data: in progress...", level=1, overwrite=True)
        fn_out = fn_base.with_name(config['base_name'].name + '.raw.json')

        qoi_train_raw = [[trj.__dict__ for trj in sample] for sample in qoi_train]
        qoi_ref_raw = [trj.__dict__ for trj in qoi_ref]
        raw_data = {
            "train": qoi_train_raw,
            "ref": qoi_ref_raw,
        }
        save_json(raw_data, fn_out)
        t3 = time.time()
        logger.info(
            f"Saving raw QoI data: Done. ({t3 - t2:.1f} s)", level=1, overwrite=True)
