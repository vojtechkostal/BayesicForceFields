"""Fit the Rg surrogate to the Sobol campaign with an 80/20 split.

Loads ``campaign/rg_hst5.pt`` (written by run_sobol_campaign), fits one
BFF GP (data mean, single committee member, 200 hyperparameter rows) to
the 20-dimensional lambda vectors with mean Rg outputs, and reports
held-out metrics on the 20 % test split - the generalization check that
the M1_24-derived fit could never honestly pass, because its training
points were 11 near-parallel optimization filaments.

Outputs in ``campaign/fit8020/``: the ``.lgp`` model, fit log, test
predictions CSV, and metrics.json (RMSE/MAE/R^2 against the training-mean
baseline, on the Rg scale in nm).

Run from the repository root:

    python -m examples.calvados_gmx.fit_sobol_gp
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from bff.bayes.fit import fit_surrogates  # noqa: E402
from bff.io.logs import Logger  # noqa: E402
from bff.qoi.dataset import QoIDataset  # noqa: E402
from examples.calvados_gmx.run_sobol_campaign import OUT  # noqa: E402

FIT = OUT / "fit8020"
SEED = 2026
TEST_FRACTION = 0.2
N_HYPER_MAX = 80  # 200 (= training count) would fit every training point


def regression_metrics(observed, predicted):
    residual = predicted - observed
    return {
        "rmse_nm": float(np.sqrt(np.mean(residual**2))),
        "mae_nm": float(np.mean(np.abs(residual))),
        "r2": float(
            1 - np.sum(residual**2) / np.sum((observed - observed.mean()) ** 2)
        ),
        "max_abs_error_nm": float(np.max(np.abs(residual))),
    }


def main():
    torch.set_num_threads(4)
    FIT.mkdir(parents=True, exist_ok=True)
    dataset = QoIDataset.load(str(OUT / "rg_hst5.pt"))
    n = dataset.n_samples
    print(f"dataset: {n} samples, {len(dataset.parameter_names)} parameters")
    if n != 100:
        print(f"warning: expected 100 samples, got {n}")

    indices = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))
    n_test = int(n * TEST_FRACTION)
    test_indices = indices[:n_test].numpy()
    train_indices = indices[n_test:].numpy()
    np.savetxt(FIT / "split_indices.txt", np.column_stack(
        [np.arange(n), np.isin(np.arange(n), test_indices).astype(int)]
    ))

    torch.manual_seed(SEED)
    logger = Logger("fit-lgp", str(FIT / "rg_hst5.log"), mode="w")
    model = fit_surrogates(
        [dataset],
        model_paths={"rg_hst5": FIT / "rg_hst5.lgp"},
        reuse_models=False,
        n_hyper_max=N_HYPER_MAX,
        committee_size=1,
        test_fraction=TEST_FRACTION,
        logger=logger,
        max_iter=500,
    )["rg_hst5"]

    np.testing.assert_array_equal(
        model.members[0].X_train.numpy(), dataset.X[train_indices]
    )
    predicted = model.predict(torch.tensor(dataset.X[test_indices])).numpy()[:, 0]
    observed = dataset.y[test_indices, 0]
    baseline = np.full_like(observed, dataset.y[train_indices].mean())
    metrics = {
        "seed": SEED,
        "train_samples": int(len(train_indices)),
        "test_samples": int(len(test_indices)),
        "n_hyper_max": N_HYPER_MAX,
        "test": regression_metrics(observed, predicted),
        "baseline_train_mean": regression_metrics(observed, baseline),
        "units": "nm (mass-weighted Rg, frame mean over 10 ns)",
    }
    for name, block in (("GP", metrics["test"]),
                        ("training-mean baseline", metrics["baseline_train_mean"])):
        print(
            f"{name}: RMSE {block['rmse_nm']:.4f} nm, MAE {block['mae_nm']:.4f} nm, "
            f"R2 {block['r2']:.3f}, max err {block['max_abs_error_nm']:.4f} nm"
        )
    (FIT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savetxt(
        FIT / "test_predictions.csv",
        np.column_stack([test_indices, observed, predicted]),
        delimiter=",",
        header="test_index,observed_rg_nm,predicted_rg_nm",
        comments="",
        fmt=["%d", "%.4f", "%.4f"],
    )
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.scatter(observed, predicted, s=25, alpha=0.7)
    limits = [1.1, 1.5]
    axis.plot(limits, limits, "k--", linewidth=1)
    axis.set_xlabel("observed mean Rg (nm)")
    axis.set_ylabel("predicted mean Rg (nm)")
    axis.set_title(
        f"Held-out (20%): RMSE {metrics['test']['rmse_nm']:.3f} nm, "
        f"R2 = {metrics['test']['r2']:.2f}"
    )
    figure.tight_layout()
    figure.savefig(FIT / "held_out_predictions.png", dpi=160)
    print(f"outputs in {FIT}")


if __name__ == "__main__":
    main()
