from ..domain.trainset import TrainSetInfo, TrajectorySet
from ..io.logs import Logger
from ..io.utils import save_json
from ..qoi.analysis import (
    analyze_trajectory_sets,
    collect_qoi_dataset,
)
from ..qoi.data import QoI, QoIDataset
from ..qoi.routines import build_analysis_routines
from .configs import AnalyzeConfig


def _build_reference_set(config: AnalyzeConfig) -> TrajectorySet:
    return TrajectorySet(
        sample_id="reference",
        fn_topol=tuple(system.fn_topol for system in config.systems),
        fn_coord=tuple(system.fn_coord for system in config.systems),
        fn_trj=tuple(system.fn_trj for system in config.systems),
    )


def _write_qoi_datasets(
    *,
    config: AnalyzeConfig,
    trainset: TrainSetInfo,
    qoi_train: list[list[dict[str, QoI]]],
    qoi_ref: list[dict[str, QoI]],
    logger: Logger,
) -> None:
    qoi_names = sorted({name for sample in qoi_ref for name in sample})
    logger.info("Saving QoI data: in progress...", level=1, overwrite=True)

    for qoi_name in qoi_names:
        fn_out = config.base_name.with_name(f"{config.base_name.name}-{qoi_name}.pt")
        system_indices = [i for i, sample in enumerate(qoi_ref) if qoi_name in sample]
        ref_blocks = [qoi_ref[i][qoi_name] for i in system_indices]
        train_blocks = [
            [sample[i][qoi_name] for i in system_indices]
            for sample in qoi_train
        ]
        output_ref, output_train, qoi_data = collect_qoi_dataset(
            ref_blocks,
            train_blocks,
        )
        if output_ref.size == 0:
            if fn_out.exists():
                fn_out.unlink()
            logger.info(
                f"Skipping QoI '{qoi_name}': no reference observations found.",
                level=2,
            )
            continue

        metadata = dict(qoi_data["metadata"])
        metadata["system_indices"] = system_indices
        dataset = QoIDataset(
            name=qoi_name,
            inputs=trainset.inputs,
            outputs=output_train,
            outputs_ref=output_ref,
            nuisance=None,
            settings_kwargs=qoi_data["settings_kwargs"],
            metadata=metadata,
        )
        dataset.write(fn_out)

    logger.info("Saving QoI data: Done.", level=1, overwrite=True)
    logger.info("", level=0)


def main(fn_config: str) -> None:
    """Run QoI analysis for a training set and its reference trajectories.

    Parameters
    ----------
    fn_config
        Path to the YAML workflow configuration.
    """
    config = AnalyzeConfig.load(fn_config)
    routines_by_system = build_analysis_routines(config.analysis)
    logger = Logger(name="analyze_qoi", fn_log=str(config.fn_log), mode="w")
    trainset = TrainSetInfo.from_dir(config.trainset_dir)
    if len(routines_by_system) != len(trainset.systems):
        raise ValueError(
            "Analysis system count must match the number of staged trainset systems."
        )

    reference_set = _build_reference_set(config)

    logger.info("=== Quantities of Interest (QoI) Analysis ===\n", level=0)

    qoi_ref_sets = analyze_trajectory_sets(
        [reference_set],
        mol_resname=trainset.specs.mol_resname,
        routines_by_system=routines_by_system,
        start=config.reference_start,
        stop=config.reference_stop,
        step=config.reference_step,
        workers=1,
        progress_stride=1,
        progress_label="Reference QoI",
        logger=logger,
        in_memory=config.analysis.in_memory,
        gc_collect=config.analysis.gc_collect,
    )

    qoi_train = analyze_trajectory_sets(
        trainset.samples,
        mol_resname=trainset.specs.mol_resname,
        routines_by_system=routines_by_system,
        start=config.training_start,
        stop=config.training_stop,
        step=config.training_step,
        workers=config.training_workers,
        progress_stride=config.training_progress_stride,
        progress_label="Training QoI",
        logger=logger,
        in_memory=config.analysis.in_memory,
        gc_collect=config.analysis.gc_collect,
        maxtasksperchild=config.analysis.maxtasksperchild,
    )

    # qoi_ref_sets = analyze_trajectory_sets(
    #     [reference_set],
    #     mol_resname=trainset.specs.mol_resname,
    #     routines_by_system=routines_by_system,
    #     start=config.reference_start,
    #     stop=config.reference_stop,
    #     step=config.reference_step,
    #     workers=1,
    #     progress_stride=1,
    #     progress_label="Reference QoI",
    #     logger=logger,
    #     in_memory=config.analysis.in_memory,
    #     gc_collect=config.analysis.gc_collect,
    # )
    qoi_ref = qoi_ref_sets[0]
    logger.info("", level=0)
    _write_qoi_datasets(
        config=config,
        trainset=trainset,
        qoi_train=qoi_train,
        qoi_ref=qoi_ref,
        logger=logger,
    )

    if config.write_raw_qoi:
        logger.info("Saving raw QoI data: in progress...", level=1, overwrite=True)
        fn_out = config.base_name.with_name(config.base_name.name + ".raw.json")
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
