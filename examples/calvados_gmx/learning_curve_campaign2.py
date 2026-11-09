"""Learning curve for the bounded Hst5 campaign at every 50 samples.

At the end of the campaign (or any time), trains a single-member GP on
the first ``k`` finished Sobol points for every multiple of 50 available
``(50, 100, 150, 200, 250, ...)`` plus however many are cached now, and
evaluates it on the fixed interim test points (the ``randperm`` seed-2026
split from ``fit_campaign2_interim``), which are never in any training
pool. Reports held-out RMSE and R2 against the fixed test set as a
function of training-set size.

Outputs in ``campaign2/fit_interim/``: ``learning_curve_50.json``,
``learning_curve_rmse_r2.png``.

Run from the repository root (any time, best at the end of the run):

    python -m examples.calvados_gmx.learning_curve_campaign2
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from bff.bayes.fit import fit_surrogates  # noqa: E402
from bff.io.logs import Logger  # noqa: E402
from bff.qoi.dataset import QoIDataset  # noqa: E402
from examples.calvados_gmx.fit_campaign2_interim import (  # noqa: E402
    FIT, SEED, TEST_FRACTION, cached_samples, predictive_std,
)
from examples.calvados_gmx.fit_sobol_gp import regression_metrics  # noqa: E402
from examples.calvados_gmx.run_sobol_campaign import EXPERIMENTAL_RG  # noqa: E402

STEP = 50


def test_indices(ids):
    """Reproduce the deterministic 90/10 holdout used by the interim fit."""
    rng = np.random.default_rng(SEED)
    n_test = max(1, int(round(len(ids) * TEST_FRACTION)))
    test_idx = np.sort(rng.choice(len(ids), size=n_test, replace=False))
    train_idx = np.setdiff1d(np.arange(len(ids)), test_idx)
    return test_idx, train_idx


def milestones(n):
    """1, then every multiple of STEP up to n, plus n itself."""
    steps = [1] + list(range(STEP, n + 1, STEP))
    if steps[-1] != n:
        steps.append(n)
    return steps


def main():
    torch.set_num_threads(4)
    X, y, ids, names = cached_samples()
    test_idx, pool_idx = test_indices(ids)
    observed = y[test_idx]
    test_ids = [ids[i] for i in test_idx]
    print(f"{len(X)} cached samples; test points: {test_ids}")

    rows = []
    for k in milestones(len(X)):
        sub = pool_idx[:k].tolist()
        if k == 1:
            # A 1-point GP posterior mean is that point's y; the posterior
            # std is the prior std, estimated from all cached means.
            predicted = np.full(len(test_idx), y[sub[0]])
            sigma = np.full(len(test_idx), y.std())
        else:
            dataset = QoIDataset(
                name=f"rg_curve_{k}",
                X=X[sub],
                y=y[sub].reshape(-1, 1),
                y_ref=np.array([EXPERIMENTAL_RG]),
                parameter_names=names,
                sample_ids=tuple(ids[i] for i in sub),
                metadata={"note": "learning curve, fixed test points"},
            )
            torch.manual_seed(SEED)
            logger = Logger("fit-lgp", str(FIT / f"curve_{k}.log"), mode="w")
            model = fit_surrogates(
                [dataset],
                model_paths={},
                reuse_models=False,
                n_hyper_max=min(80, k),
                committee_size=1,
                test_fraction=0.05,
                logger=logger,
                max_iter=500,
            )[dataset.name]
            with torch.inference_mode():
                predicted = model.predict(
                    torch.tensor(X[test_idx], dtype=torch.float64)
                ).numpy()[:, 0]
            sigma = predictive_std(model, X[test_idx])
        rmse = float(np.sqrt(np.mean((observed - predicted) ** 2)))
        baseline = np.full(len(observed), y[sub].mean())
        baseline_rmse = float(np.sqrt(np.mean((observed - baseline) ** 2)))
        covered = (np.abs(observed - predicted) <= sigma).sum()
        rows.append(
            {
                "n_train": k,
                "predicted": [float(p) for p in predicted],
                "gp_sigma": [float(s) for s in sigma],
                "rmse_nm": rmse,
                "r2": regression_metrics(observed, predicted)["r2"],
                "baseline_rmse_nm": baseline_rmse,
                "baseline_r2": regression_metrics(observed, baseline)["r2"],
                "within_1sigma": int(covered),
            }
        )
        print(f"k={k}: RMSE {rmse:.4f} nm, R2 "
              f"{rows[-1]['r2']:+.3f}, within 1 sigma {covered}/"
              f"{len(test_idx)}, mean sigma {sigma.mean():.4f} nm")

    (FIT / "learning_curve_50.json").write_text(
        json.dumps({"test_points": test_ids, "observed": observed.tolist(),
                    "runs": rows}, indent=2) + "\n"
    )

    plotted = [row for row in rows if row["n_train"] >= STEP]
    ks = [row["n_train"] for row in plotted]
    figure, (ax_rmse, ax_r2) = plt.subplots(
        2, 1, figsize=(6.5, 7), sharex=True
    )
    ax_rmse.plot(ks, [r["rmse_nm"] for r in plotted], "o-", color="C0",
                 label="GP surrogate")
    ax_rmse.plot(ks, [r["baseline_rmse_nm"] for r in plotted], "s--",
                 color="0.45", label="train-mean baseline")
    ax_rmse.set_ylabel("held-out RMSE (nm)")
    ax_rmse.set_title(
        f"Fixed held-out set ({len(test_idx)} points) vs training size",
        fontsize=10,
    )
    ax_rmse.legend(fontsize=9)
    ax_rmse.grid(alpha=0.25)

    ax_r2.plot(ks, [r["r2"] for r in plotted], "o-", color="C0",
               label="GP surrogate")
    ax_r2.plot(ks, [r["baseline_r2"] for r in plotted], "s--",
               color="0.45", label="train-mean baseline")
    ax_r2.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
    ax_r2.set_xlabel("training points")
    ax_r2.set_ylabel(r"held-out $R^2$")
    ax_r2.set_xticks(ks)
    ax_r2.legend(fontsize=9)
    ax_r2.grid(alpha=0.25)
    for axis in (ax_rmse, ax_r2):
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(FIT / "learning_curve_rmse_r2.png", dpi=160)
    print(f"outputs in {FIT}")


if __name__ == "__main__":
    main()
