"""Fit and evaluate two BFF GPs of CALVADOS2 chi-squared discrepancies."""

import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from bff.bayes.fit import fit_surrogates
from bff.io.logs import Logger
from bff.qoi.dataset import QoIDataset
from examples.calvados.fetch_data import COMMIT, RAW, verified_bytes

SEED = 2026
TEST_FRACTION = 0.2
OUTPUT = Path(__file__).resolve().parent / "generated" / "discrepancy_direct"


def history_datasets(histories):
    parameter_ids = None
    vectors, outputs, sample_ids, cycles = [], [], [], []
    for cycle, table in sorted(histories.items()):
        if table.index.has_duplicates:
            raise ValueError(f"CALVADOS cycle {cycle}: duplicate iteration IDs")
        for iteration, row in table.iterrows():
            parameters = row.lambdas
            if parameters.index.has_duplicates or len(parameters) != 20:
                raise ValueError(f"CALVADOS cycle {cycle}: expected 20 unique residues")
            if parameter_ids is None:
                parameter_ids = tuple(sorted(parameters.index))
            if set(parameters.index) != set(parameter_ids):
                raise ValueError(f"CALVADOS cycle {cycle}: inconsistent residue IDs")
            vectors.append(parameters.loc[list(parameter_ids)].to_numpy(float))
            outputs.append([float(row.chi2_rg), float(row.chi2_pre)])
            sample_ids.append(f"M1_24_cycle_{cycle}_iteration_{iteration}")
            cycles.append(cycle)
    parameters = np.asarray(vectors)
    discrepancies = np.asarray(outputs)
    if parameters.size == 0:
        raise ValueError("CALVADOS histories: no training records")
    if not np.isfinite(parameters).all() or np.any(
        (parameters <= 0) | (parameters > 1)
    ):
        raise ValueError("CALVADOS histories: invalid stickiness parameters")
    if not np.isfinite(discrepancies).all() or np.any(discrepancies < 0):
        raise ValueError(
            "CALVADOS histories: discrepancies must be finite and nonnegative"
        )
    if len(np.unique(parameters, axis=0)) != len(parameters):
        raise ValueError(
            "CALVADOS histories: duplicate inputs would contaminate the split"
        )
    datasets = [
        QoIDataset(
            name=column,
            X=parameters,
            y=discrepancies[:, [output_index]],
            y_ref=np.zeros(1),
            parameter_names=tuple(f"lambda {residue}" for residue in parameter_ids),
            sample_ids=tuple(sample_ids),
            metadata={
                "upstream_commit": COMMIT,
                "run": "M1_24",
                "transform": "identity",
                "source_column": column,
                "cycles": cycles,
                "reference_role": "zero discrepancy; likelihood scaling not configured",
            },
        )
        for output_index, column in enumerate(("chi2_rg", "chi2_pre"))
    ]
    return datasets


def regression_metrics(observed, predicted):
    residual = predicted - observed
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(
            1 - np.sum(residual**2) / np.sum((observed - observed.mean()) ** 2)
        ),
        "max_absolute_error": float(np.max(np.abs(residual))),
    }


def main():
    plt.switch_backend("Agg")
    torch.set_num_threads(2)
    histories = {
        cycle: pd.read_pickle(io.BytesIO(verified_bytes(RAW / f"{cycle}_chi2.pkl")))
        for cycle in range(1, 12)
    }
    datasets = history_datasets(histories)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sample_count = datasets[0].n_samples
    indices = torch.randperm(
        sample_count, generator=torch.Generator().manual_seed(SEED)
    )
    test_size = int(sample_count * TEST_FRACTION)
    test_indices = indices[:test_size].numpy()
    train_indices = indices[test_size:].numpy()
    manifest = pd.DataFrame(
        {
            "sample_id": datasets[0].sample_ids,
            "cycle": datasets[0].metadata["cycles"],
            "split": "train",
        }
    )
    manifest.loc[test_indices, "split"] = "test"
    manifest.to_csv(OUTPUT / "split.csv", index=False)
    report = {
        "seed": SEED,
        "upstream_commit": COMMIT,
        "train_samples": len(train_indices),
        "test_samples": len(test_indices),
        "n_hyper_max": 200,
        "transform": "identity",
        "models": {},
    }
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for dataset, axis in zip(datasets, axes):
        dataset.write(str(OUTPUT / f"{dataset.name}.pt"))
        torch.manual_seed(SEED)
        logger = Logger("fit-lgp", str(OUTPUT / f"{dataset.name}.log"), mode="w")
        model = fit_surrogates(
            [dataset],
            model_paths={dataset.name: OUTPUT / f"{dataset.name}.lgp"},
            reuse_models=False,
            n_hyper_max=200,
            committee_size=1,
            test_fraction=TEST_FRACTION,
            logger=logger,
            max_iter=500,
        )[dataset.name]
        np.testing.assert_array_equal(
            model.members[0].X_train.numpy(), dataset.X[train_indices]
        )
        predicted = model.predict(torch.tensor(dataset.X[test_indices])).numpy()[:, 0]
        observed = dataset.y[test_indices, 0]
        baseline = np.full_like(observed, dataset.y[train_indices].mean())
        metrics = {
            "chi2_scale": regression_metrics(observed, predicted),
            "baseline_chi2_scale": regression_metrics(observed, baseline),
            "smape_percent": model.test_error,
            "negative_predictions": int(np.sum(predicted < 0)),
        }
        report["models"][dataset.name] = metrics
        pd.DataFrame(
            {
                "sample_id": np.asarray(dataset.sample_ids)[test_indices],
                "observed_chi2": observed,
                "predicted_chi2": predicted,
            }
        ).to_csv(OUTPUT / f"{dataset.name}_test.csv", index=False)
        axis.scatter(observed, predicted, s=12, alpha=0.6)
        limits = [
            min(observed.min(), predicted.min()),
            max(observed.max(), predicted.max()),
        ]
        axis.plot(limits, limits, "k--", linewidth=1)
        axis.set(
            xlabel="Recorded mean chi-squared",
            ylabel="Predicted mean chi-squared",
            title=f"{dataset.name}: R² = {metrics['chi2_scale']['r2']:.3f}",
        )
        print(dataset.name, json.dumps(metrics, indent=2), flush=True)
    figure.tight_layout()
    figure.savefig(OUTPUT / "held_out_predictions.png", dpi=180)
    plt.close(figure)
    (OUTPUT / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
