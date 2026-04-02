import numpy as np

from ..domain.trainset import TrainSetInfo, TrajectorySet
from ..io.logs import Logger
from ..io.utils import save_json
from ..qoi.analysis import analyze_trajectory_sets
from ..qoi.data import QoI, QoIDataset
from ..qoi.routines import build_analysis_routines
from .configs import QoIConfig


def _qoi_output_path(path, qoi_name: str | None = None, *, raw: bool = False):
    if raw:
        return path.with_name(path.name + ".raw.json")
    if qoi_name is None:
        raise ValueError("qoi_name is required unless raw=True.")
    return path.with_name(f"{path.name}-{qoi_name}.pt")


def _validate_qoi_blocks(
    blocks: list[QoI],
    *,
    context: str,
) -> tuple[str, ...] | None:
    if not blocks:
        return None

    first = blocks[0]
    labels = first.labels
    values_per_label = first.values_per_label
    n_values = first.n_values
    for block in blocks[1:]:
        if block.values_per_label != values_per_label:
            raise ValueError(
                f"QoI schema mismatch in {context}: expected values_per_label="
                f"{values_per_label}, got {block.values_per_label}."
            )
        if block.labels != labels:
            raise ValueError(
                f"QoI label mismatch in {context}: expected {labels}, "
                f"got {block.labels}."
            )
        if block.n_values != n_values:
            raise ValueError(
                f"QoI value-count mismatch in {context}: expected {n_values}, "
                f"got {block.n_values}."
            )
    return labels


def _stack_qoi_blocks(
    blocks: list[QoI],
    *,
    context: str,
) -> np.ndarray:
    if not blocks:
        return np.empty(0, dtype=float)
    _validate_qoi_blocks(blocks, context=context)
    return np.concatenate([block.values for block in blocks])


def _dataset_labels(
    ref_blocks: list[QoI],
    system_indices: list[int],
) -> tuple[str, ...] | None:
    if not ref_blocks:
        return None

    labels = ref_blocks[0].labels
    if labels is None:
        return None
    if len(ref_blocks) == 1:
        return labels

    return tuple(
        f"window-{system_index:03d}:{label}"
        for system_index, block in zip(system_indices, ref_blocks)
        for label in block.labels or ()
    )


def _write_qoi_datasets(
    *,
    fn_out,
    inputs,
    qoi_train: list[list[dict[str, QoI]]],
    qoi_ref: list[dict[str, QoI]],
) -> None:
    qoi_names = sorted({name for sample in qoi_ref for name in sample})

    for qoi_name in qoi_names:
        fn_qoi = _qoi_output_path(fn_out, qoi_name)
        system_indices = [i for i, sample in enumerate(qoi_ref) if qoi_name in sample]
        ref_blocks = [qoi_ref[i][qoi_name] for i in system_indices]
        train_blocks = [
            [sample[i][qoi_name] for i in system_indices]
            for sample in qoi_train
        ]

        _validate_qoi_blocks(
            ref_blocks,
            context="reference QoI blocks",
        )
        reference = ref_blocks[0] if ref_blocks else None
        outputs_ref = _stack_qoi_blocks(
            ref_blocks,
            context="reference QoI blocks",
        )
        outputs = []
        for i, blocks in enumerate(train_blocks):
            if reference is not None and blocks:
                _validate_qoi_blocks(
                    [reference, *blocks],
                    context=f"training QoI blocks for sample {i}",
                )
            outputs.append(
                _stack_qoi_blocks(
                    blocks,
                    context=f"training QoI blocks for sample {i}",
                )
            )

        if outputs_ref.size == 0:
            if fn_qoi.exists():
                fn_qoi.unlink()
            continue

        settings: dict[str, object] = {}
        metadata: dict[str, object] = {"system_indices": system_indices}
        values_per_label = 1
        if reference is not None:
            settings_by_block = [dict(block.settings) for block in ref_blocks]
            if settings_by_block:
                shared_settings = settings_by_block[0]
                if all(
                    block_settings == shared_settings
                    for block_settings in settings_by_block[1:]
                ):
                    settings = dict(shared_settings)
                else:
                    metadata["settings_by_block"] = settings_by_block

            metadata_by_block = [dict(block.metadata) for block in ref_blocks]
            if metadata_by_block:
                shared_metadata = metadata_by_block[0]
                if all(
                    block_metadata == shared_metadata
                    for block_metadata in metadata_by_block[1:]
                ):
                    metadata = dict(shared_metadata) | metadata
                else:
                    metadata["metadata_by_block"] = metadata_by_block

            labels = _dataset_labels(ref_blocks, system_indices)
            values_per_label = reference.values_per_label

        dataset = QoIDataset(
            name=qoi_name,
            inputs=inputs,
            outputs=outputs,
            outputs_ref=outputs_ref,
            labels=labels,
            values_per_label=values_per_label,
            nuisance=None,
            settings=settings,
            metadata=metadata,
        )
        dataset.write(fn_qoi)


def main(fn_config: str) -> None:
    """Run QoI analysis for a training set and its reference trajectories.

    Parameters
    ----------
    fn_config
        Path to the YAML workflow configuration.
    """
    config = QoIConfig.load(fn_config)
    train_cfg = config.trainset
    ref_cfg = config.refset
    run_cfg = config.run
    output_cfg = config.output

    routines_by_system = build_analysis_routines(
        [system.routines for system in ref_cfg.systems]
    )
    logger = Logger(name="qoi", fn_log=str(output_cfg.log), mode="w")
    trainset = TrainSetInfo.from_dir(train_cfg.dir)
    if len(routines_by_system) != len(trainset.systems):
        raise ValueError(
            "Analysis system count must match the number of staged trainset systems."
        )

    reference_set = TrajectorySet(
        sample_id="reference",
        fn_topol=tuple(system.fn_topol for system in ref_cfg.systems),
        fn_coord=tuple(system.fn_coord for system in ref_cfg.systems),
        fn_trj=tuple(system.fn_trj for system in ref_cfg.systems),
    )

    logger.info("=== Quantities of Interest (QoI) Analysis ===\n", level=0)

    qoi_ref = analyze_trajectory_sets(
        [reference_set],
        routines_by_system=routines_by_system,
        start=ref_cfg.start,
        stop=ref_cfg.stop,
        step=ref_cfg.step,
        workers=1,
        progress_stride=1,
        progress_label="Reference QoI",
        logger=logger,
        in_memory=run_cfg.in_memory,
        gc_collect=run_cfg.gc_collect,
    )[0]

    logger.info("", level=0)

    qoi_train = analyze_trajectory_sets(
        trainset.samples,
        routines_by_system=routines_by_system,
        start=train_cfg.start,
        stop=train_cfg.stop,
        step=train_cfg.step,
        workers=train_cfg.workers,
        progress_stride=train_cfg.progress_stride,
        progress_label="Training QoI",
        logger=logger,
        in_memory=run_cfg.in_memory,
        gc_collect=run_cfg.gc_collect,
        maxtasksperchild=run_cfg.maxtasksperchild,
    )

    logger.info("", level=0)
    logger.info("Saving QoI data: in progress...", level=1, overwrite=True)
    _write_qoi_datasets(
        fn_out=output_cfg.path,
        inputs=trainset.inputs,
        qoi_train=qoi_train,
        qoi_ref=qoi_ref,
    )
    logger.info("Saving QoI data: Done.", level=1, overwrite=True)
    logger.info("", level=0)

    if output_cfg.write_raw:
        logger.info("Saving raw QoI data: in progress...", level=1, overwrite=True)
        fn_out = _qoi_output_path(output_cfg.path, raw=True)
        raw_data = {
            "train": [
                [{name: qoi.to_dict() for name, qoi in trj.items()} for trj in sample]
                for sample in qoi_train
            ],
            "ref": [
                {name: qoi.to_dict() for name, qoi in trj.items()}
                for trj in qoi_ref
            ],
        }
        save_json(raw_data, fn_out)
        logger.info("Saving raw QoI data: Done.", level=1, overwrite=True)
