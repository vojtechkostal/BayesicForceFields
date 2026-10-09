"""Build ID-paired QoI datasets from explicit analysis inputs."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from ...domain.samples import SampleSet
from ...io.logs import Logger
from ...qoi.analysis import AnalysisTask, analyze_samples
from ...qoi.dataset import QoI, QoIDataset
from .config import BuildQoIDatasetsConfig, required_input_roles


def _validate_qoi_blocks(blocks: list[QoI], *, context: str) -> None:
    if not blocks:
        raise ValueError(f"{context} contains no QoI blocks.")
    first = blocks[0]
    for block in blocks[1:]:
        if block.values_per_label != first.values_per_label:
            raise ValueError(
                f"{context}: expected values_per_label={first.values_per_label}, "
                f"got {block.values_per_label}."
            )
        if block.labels != first.labels:
            raise ValueError(
                f"{context}: expected labels {first.labels!r}, got {block.labels!r}."
            )
        if block.n_values != first.n_values:
            raise ValueError(
                f"{context}: expected {first.n_values} values, got {block.n_values}."
            )


def _labels(blocks: list[QoI], system_ids: tuple[str, ...]) -> tuple[str, ...] | None:
    labels = blocks[0].labels
    if len(blocks) == 1:
        return labels
    if labels is None:
        if all(block.n_values == block.values_per_label for block in blocks):
            return system_ids
        return None
    return tuple(
        f"{system_id}:{label}"
        for system_id, block in zip(system_ids, blocks)
        for label in block.labels or ()
    )


def _shared_block_metadata(
    blocks: list[QoI],
) -> tuple[dict[str, Any], dict[str, Any]]:
    settings = [dict(block.settings) for block in blocks]
    metadata = [dict(block.metadata) for block in blocks]
    settings_are_shared = all(value == settings[0] for value in settings)
    metadata_are_shared = all(value == metadata[0] for value in metadata)
    shared_settings = dict(settings[0]) if settings_are_shared else {}
    shared_metadata = dict(metadata[0]) if metadata_are_shared else {}
    if not settings_are_shared:
        shared_metadata["settings_by_system"] = settings
    if not metadata_are_shared:
        shared_metadata["metadata_by_system"] = metadata
    return shared_settings, shared_metadata


def _mismatch(reference: QoI, block: QoI) -> str | None:
    """How a sample's QoI differs from the reference's, if it does."""
    if block.settings != reference.settings:
        return f"settings {block.settings!r} differ from {reference.settings!r}"
    if block.labels != reference.labels:
        return f"labels {block.labels!r} differ from {reference.labels!r}"
    if block.n_values != reference.n_values:
        return f"{block.n_values} values instead of {reference.n_values}"
    if block.values_per_label != reference.values_per_label:
        return "values_per_label differs from the reference"
    return None


def main(fn_config: str | Path) -> None:
    started = time.perf_counter()
    config = BuildQoIDatasetsConfig.load(fn_config)
    training = config.training_samples
    system_ids = training.system_ids
    sample_set = SampleSet.from_manifest(training.manifest)
    absent = sorted(set(system_ids) - set(sample_set.system_ids))
    if absent:
        raise ValueError(
            f"training_samples.systems lists {absent}, which the campaign "
            f"{training.manifest} does not contain."
        )
    routines_by_system = {
        system_id: tuple(
            routine for routine in config.routines if system_id in routine.systems
        )
        for system_id in system_ids
    }

    config.output.directory.mkdir(parents=True, exist_ok=True)
    logger = Logger("build-qoi-datasets", str(config.output.log), mode="w")
    logger.section("Build QoI Datasets")
    logger.kv("Config", config.fn_config)
    logger.kv("Campaign", training.manifest)
    logger.kv("Systems", ", ".join(system_ids))
    logger.kv(
        "Samples",
        f"{len(sample_set.samples)} completed, {training.workers} worker(s)"
        if training.workers != -1
        else f"{len(sample_set.samples)} completed, all CPUs",
    )
    logger.kv("Routines", ", ".join(routine.name for routine in config.routines))
    logger.kv("Output", config.output.directory)
    logger.blank()

    # Samples are skipped, not fatal: missing files, missing roles, failed
    # analysis, or QoIs that do not match the reference.
    skipped: dict[str, str] = dict(sample_set.unusable)
    tasks: list[AnalysisTask] = []
    for sample in sample_set.samples:
        inputs = {system_id: sample.inputs[system_id] for system_id in system_ids}
        missing = {
            system_id: sorted(
                required_input_roles(routines_by_system[system_id])
                - set(inputs[system_id])
            )
            for system_id in system_ids
        }
        missing = {key: value for key, value in missing.items() if value}
        if missing:
            skipped[sample.sample_id] = (
                f"no {missing} output; store it in the campaign (trajectory: xtc)"
            )
            continue
        tasks.append((sample.sample_id, inputs))

    reference_inputs = {
        system.system_id: dict(system.inputs.inputs)
        for system in config.reference.systems
    }
    ref_frames = config.reference.frames
    reference_results, reference_failures = analyze_samples(
        [("reference", reference_inputs)],
        routines_by_system=routines_by_system,
        frames=slice(ref_frames.start, ref_frames.stop, ref_frames.step),
        workers=1,
        in_memory=config.in_memory,
        logger=logger,
        label="Reference",
    )
    if reference_failures:
        raise ValueError(
            f"Reference analysis failed: {reference_failures['reference']}"
        )
    reference = reference_results["reference"]
    for routine in config.routines:
        _validate_qoi_blocks(
            [reference[system_id][routine.name] for system_id in routine.systems],
            context=f"Reference routine {routine.name!r}",
        )

    train_frames = training.frames
    results, failures = analyze_samples(
        tasks,
        routines_by_system=routines_by_system,
        frames=slice(train_frames.start, train_frames.stop, train_frames.step),
        workers=training.workers,
        in_memory=config.in_memory,
        logger=logger,
        label="Samples",
    )
    skipped |= failures
    for sample_id, result in list(results.items()):
        for routine in config.routines:
            for system_id in routine.systems:
                problem = _mismatch(
                    reference[system_id][routine.name], result[system_id][routine.name]
                )
                if problem and sample_id in results:
                    skipped[sample_id] = f"{routine.name} of {system_id}: {problem}"
                    del results[sample_id]
    for sample_id in sorted(skipped):
        logger.warn(f"Sample {sample_id} skipped: {skipped[sample_id]}")
    samples = [sample for sample in sample_set.samples if sample.sample_id in results]
    if not samples:
        raise ValueError(
            "No sample could be analyzed; the reasons are listed in "
            f"{config.output.log}."
        )

    logger.blank()
    sample_ids = [sample.sample_id for sample in samples]
    X = np.asarray([sample.params for sample in samples], dtype=float)
    for routine in config.routines:
        ref_blocks = [
            reference[system_id][routine.name]
            for system_id in routine.systems
        ]
        settings, metadata = _shared_block_metadata(ref_blocks)
        metadata["system_ids"] = list(routine.systems)
        dataset = QoIDataset(
            name=routine.name,
            X=X,
            y=np.asarray(
                [
                    np.concatenate(
                        [
                            results[sample_id][system_id][routine.name].values
                            for system_id in routine.systems
                        ]
                    )
                    for sample_id in sample_ids
                ],
                dtype=float,
            ),
            y_ref=np.concatenate([block.values for block in ref_blocks]),
            labels=_labels(ref_blocks, routine.systems),
            values_per_label=ref_blocks[0].values_per_label,
            settings=settings,
            metadata=metadata,
            sample_ids=sample_ids,
            parameter_names=sample_set.parameter_names,
        )
        fn_dataset = config.output.directory / f"{routine.name}.pt"
        dataset.write(fn_dataset)
        logger.done(
            routine.name,
            detail=f"{dataset.n_samples} samples x {dataset.y.shape[1]} values"
            f" | {fn_dataset}",
        )

    logger.blank()
    total = len(sample_set.samples) + len(sample_set.unusable)
    logger.done(
        "Build QoI Datasets",
        detail=f"{len(samples)} of {total} samples, {len(config.routines)} "
        f"dataset(s) | {time.perf_counter() - started:.1f} s",
    )
