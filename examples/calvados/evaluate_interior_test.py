"""Evaluate the existing 80/20 GPs after filtering boundary test inputs."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from bff.bayes.gaussian_process import LGPCommittee
from bff.qoi.dataset import QoIDataset
from examples.calvados.train_discrepancy import regression_metrics

SOURCE = Path(__file__).resolve().parent / "generated" / "discrepancy_direct"
OUTPUT = SOURCE.parent / "discrepancy_direct_80_20_interior_005"


def interior_mask(parameters):
    return np.all((parameters > 0.005) & (parameters < 0.995), axis=1)


def main():
    plt.switch_backend("Agg")
    torch.set_num_threads(2)
    split = pd.read_csv(SOURCE / "split.csv").set_index("sample_id")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = {
        "source": SOURCE.name,
        "test_parameter_interval": [0.005, 0.995],
        "interval_endpoints": "excluded",
        "models": {},
    }
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for target, axis in zip(("chi2_rg", "chi2_pre"), axes):
        dataset = QoIDataset.load(str(SOURCE / f"{target}.pt"))
        parameters = pd.DataFrame(
            dataset.X, index=dataset.sample_ids, columns=dataset.parameter_names
        )
        train = parameters.loc[split.index[split.split == "train"]]
        test = parameters.loc[split.index[split.split == "test"]]
        keep = interior_mask(test.to_numpy())
        selected = test.loc[keep]
        if len(selected) < 2:
            raise ValueError(
                "CALVADOS interior test: fewer than two test samples remain"
            )
        model = LGPCommittee.load(SOURCE / f"{target}.lgp")
        expected_inputs = set(map(tuple, train.to_numpy()))
        for member in model.members:
            if set(map(tuple, member.X_train.numpy())) != expected_inputs:
                raise ValueError(
                    f"CALVADOS {target}: model does not match 80/20 training split"
                )
        predictions = model.predict(torch.tensor(selected.to_numpy())).numpy()[:, 0]
        values = pd.Series(dataset.y[:, 0], index=dataset.sample_ids)
        observed = values.loc[selected.index].to_numpy()
        baseline = np.full_like(observed, values.loc[train.index].mean())
        metrics = {
            "chi2_scale": regression_metrics(observed, predictions),
            "baseline_chi2_scale": regression_metrics(observed, baseline),
            "smape_percent": model.validate(
                torch.tensor(selected.to_numpy()), torch.tensor(observed[:, None])
            ),
        }
        report["models"][target] = metrics
        results = selected.copy()
        results.insert(0, "absolute_error", np.abs(predictions - observed))
        results.insert(0, "predicted_chi2", predictions)
        results.insert(0, "observed_chi2", observed)
        results.to_csv(OUTPUT / f"{target}_test.csv", index_label="sample_id")
        axis.scatter(observed, predictions)
        limits = [
            min(observed.min(), predictions.min()),
            max(observed.max(), predictions.max()),
        ]
        axis.plot(limits, limits, "k--", linewidth=1)
        axis.set(
            xlabel="Recorded mean chi-squared",
            ylabel="Predicted mean chi-squared",
            title=f"{target}: R² = {metrics['chi2_scale']['r2']:.3f}",
        )
        print(target, json.dumps(metrics, indent=2))
    audit = split.copy()
    audit.loc[test.index[~keep], "split"] = "excluded_test_boundary"
    audit.join(parameters).to_csv(OUTPUT / "split.csv", index_label="sample_id")
    report.update(
        train_samples=len(train),
        original_test_samples=len(test),
        excluded_test_samples=int((~keep).sum()),
        test_samples=len(selected),
    )
    (OUTPUT / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    figure.tight_layout()
    figure.savefig(OUTPUT / "held_out_predictions.png", dpi=180)
    plt.close(figure)
    print(f"Train: {len(train)}; test: {len(selected)}; excluded: {(~keep).sum()}")


if __name__ == "__main__":
    main()
