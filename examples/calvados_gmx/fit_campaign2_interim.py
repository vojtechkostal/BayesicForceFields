"""Interim 80/20 GP fit on the running bounded Hst5 campaign.

Trains on however many ``campaign2/hst_<i>/rg.txt`` files exist right
now (the campaign keeps running; this script never touches the cache),
with an 80/20 split, and reports held-out RMSE/R2 against the
training-mean baseline plus a true-vs-predicted scatter for the test
points. The first honest check of whether the 9-D bounded redesign made
the Rg surface learnable.

Outputs in ``campaign2/fit_interim/``: metrics.json, test predictions
CSV, and ``held_out_predictions.png``.

Run from the repository root (any time):

    python -m examples.calvados_gmx.fit_campaign2_interim
"""

import json
import sys
from collections import Counter
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
from examples.calvados_gmx.fit_sobol_gp import regression_metrics  # noqa: E402
from examples.calvados_gmx.run_campaign2 import (  # noqa: E402
    HST5,
    N_SAMPLES,
    OUT,
    parameter_names,
    sobol_points,
)
from examples.calvados_gmx.run_sobol_campaign import EXPERIMENTAL_RG  # noqa: E402

FIT = OUT / "fit_interim"
SEED = 2026
TEST_FRACTION = 0.1
TEST_FRACTION = 0.1


def cached_samples():
    """(X, y) for every finished campaign point, in Sobol order."""
    residues = [one for one in sorted(set(HST5)) if one != "E"]
    points = sobol_points(residues, N_SAMPLES)
    X, y, ids = [], [], []
    for index in range(N_SAMPLES):
        rg_file = OUT / f"hst_{index:03d}" / "rg.txt"
        if rg_file.is_file():
            X.append(points[index])
            y.append(float(np.loadtxt(rg_file).mean()))
            ids.append(f"hst_{index:03d}")
    return np.array(X), np.array(y), ids, parameter_names(residues)


GROUPS = {
    "H": {"H"},
    "charged": {"D", "E", "K", "R"},
    "aromatic": {"F", "Y"},
    "polar": {"S", "T", "N", "Q", "C"},
    "aliphatic": {"A", "V", "L", "I", "M", "G", "P"},
}


def stickiness_features(X, residues):
    """Sequence-weighted group stickiness: ``sum_i n_i * lambda_i / N``.

    One column per chemistry group, linear in the sampled lambdas. This is
    the low-dimensional feature behind a Flory-type prior mean for Rg: the
    dominant dependence on sequence-averaged stickiness, which a stationary
    GP otherwise has to extrapolate with its kernel (shrinking the tails).
    """
    counts = Counter(HST5)
    total = len(HST5)
    columns = []
    for members in GROUPS.values():
        weights = np.zeros(len(residues))
        for j, one in enumerate(residues):
            if one in members:
                weights[j] = counts[one] / total
        columns.append(X @ weights)
    return np.column_stack(columns)


def predictive_std(model, X):
    """Exact GP posterior std at ``X`` for a single-member LGPCommittee.

    BFF's ``predict`` returns means only, but every ingredient of the
    posterior variance is stored on the member:
    ``var(x) = amp^2 - k* (K + s2 I)^-1 k*``.
    """
    from bff.bayes.kernels import gaussian_kernel

    member = model.members[0]
    X_train = member.X_train
    K = gaussian_kernel(X_train, X_train, member.lengthscales, member.amplitude)
    K = K + member.noise_variance * torch.eye(
        len(X_train), dtype=K.dtype, device=K.device
    )
    K_star = gaussian_kernel(
        torch.tensor(X, dtype=X_train.dtype), X_train,
        member.lengthscales, member.amplitude,
    )
    solved = torch.linalg.solve(K, K_star.T)
    variance = (
        member.amplitude ** 2
        - (K_star * solved.T).sum(1)
    ).clamp(min=0.0)
    return variance.sqrt().numpy()


def main():
    torch.set_num_threads(4)
    FIT.mkdir(parents=True, exist_ok=True)
    X, y, ids, names = cached_samples()
    n = len(X)
    rng = np.random.default_rng(SEED)
    n_test = max(1, int(round(n * TEST_FRACTION)))
    test_idx = np.sort(rng.choice(n, size=n_test, replace=False))
    train_idx = np.setdiff1d(np.arange(n), test_idx)
    n_test = len(test_idx)
    print(f"{n} cached samples: {len(train_idx)} train / {n_test} test")

    dataset = QoIDataset(
        name="rg_hst5_bounded_interim",
        X=X[train_idx],
        y=y[train_idx].reshape(-1, 1),
        y_ref=np.array([EXPERIMENTAL_RG]),
        parameter_names=names,
        sample_ids=tuple(ids[i] for i in train_idx),
        metadata={"note": "interim fit while campaign2 runs"},
    )
    torch.manual_seed(SEED)
    logger = Logger("fit-lgp", str(FIT / "fit.log"), mode="w")
    model = fit_surrogates(
        [dataset],
        model_paths={dataset.name: FIT / "model.lgp"},
        reuse_models=False,
        n_hyper_max=min(80, len(train_idx)),
        committee_size=1,
        test_fraction=0.05,
        logger=logger,
        max_iter=500,
    )[dataset.name]

    with torch.inference_mode():
        predicted = model.predict(
            torch.tensor(X[test_idx], dtype=torch.float64)
        ).numpy()[:, 0]
    observed = y[test_idx]
    sigma = predictive_std(model, X[test_idx])

    residues = [one for one in sorted(set(HST5)) if one != "E"]
    features = stickiness_features(X, residues)
    feature_names = ["intercept", *GROUPS]
    design_train = np.column_stack(
        [np.ones(len(train_idx)), features[train_idx]]
    )
    mean_coef, *_ = np.linalg.lstsq(
        design_train, y[train_idx], rcond=None
    )
    mean_train = design_train @ mean_coef
    scaler = np.zeros((len(test_idx), len(feature_names)))
    scaler[:, 0] = 1.0
    scaler[:, 1:] = features[test_idx]
    mean_test = scaler @ mean_coef
    residual_dataset = QoIDataset(
        name="rg_hst5_bounded_interim_residual",
        X=X[train_idx],
        y=(y[train_idx] - mean_train).reshape(-1, 1),
        y_ref=np.array([EXPERIMENTAL_RG]),
        parameter_names=names,
        sample_ids=tuple(ids[i] for i in train_idx),
        metadata={"note": "residual after stickiness-mean regression"},
    )
    torch.manual_seed(SEED)
    residual_logger = Logger(
        "fit-lgp", str(FIT / "fit_linear_mean.log"), mode="w"
    )
    residual_model = fit_surrogates(
        [residual_dataset],
        model_paths={
            residual_dataset.name: FIT / "model_linear_mean.lgp"
        },
        reuse_models=False,
        n_hyper_max=min(80, len(train_idx)),
        committee_size=1,
        test_fraction=0.05,
        logger=residual_logger,
        max_iter=500,
    )[residual_dataset.name]
    with torch.inference_mode():
        predicted_linear = residual_model.predict(
            torch.tensor(X[test_idx], dtype=torch.float64)
        ).numpy()[:, 0] + mean_test
    residual_train = y[train_idx] - mean_train
    dof = max(1, len(train_idx) - design_train.shape[1])
    coef_cov = (residual_train @ residual_train / dof) * np.linalg.pinv(
        design_train.T @ design_train
    )
    mean_var = np.einsum("ij,jk,ik->i", scaler, coef_cov, scaler)
    sigma_linear = np.sqrt(
        predictive_std(residual_model, X[test_idx]) ** 2 + mean_var
    )

    covered = np.abs(observed - predicted) <= sigma
    baseline = np.full_like(observed, y[train_idx].mean())
    metrics = {
        "n_samples": n,
        "train": int(len(train_idx)),
        "test": int(n_test),
        "test_metrics": regression_metrics(observed, predicted),
        "baseline_train_mean": regression_metrics(observed, baseline),
        "linear_mean": {
            "test_metrics": regression_metrics(observed, predicted_linear),
            "coefficients": {
                key: float(value)
                for key, value in zip(feature_names, mean_coef)
            },
            "gp_sigma_test_nm": [float(s) for s in sigma_linear],
            "within_1sigma": [
                bool(c)
                for c in np.abs(observed - predicted_linear) <= sigma_linear
            ],
        },
        "signal_std_nm": float(y.std()),
        "gp_sigma_test_nm": [float(s) for s in sigma],
        "within_1sigma": [bool(c) for c in covered],
    }
    for label, block in (("GP", metrics["test_metrics"]),
                         ("linear mean", metrics["linear_mean"]["test_metrics"]),
                         ("baseline", metrics["baseline_train_mean"])):
        print(
            f"{label}: RMSE {block['rmse_nm']:.4f} nm, "
            f"R2 {block['r2']:+.3f}, "
            f"max err {block['max_abs_error_nm']:.4f} nm"
        )
    print(
        "linear-mean coefficients: "
        + ", ".join(
            f"{key} {value:+.3f}"
            for key, value in zip(feature_names, mean_coef)
        )
    )
    print(f"signal std (all means): {metrics['signal_std_nm']:.4f} nm")
    for i, (o, p, s) in enumerate(zip(observed, predicted, sigma)):
        print(
            f"  {ids[test_idx[i]]}: obs {o:.4f}, pred {p:.4f} "
            f"+- {s:.4f} nm ({'|resid|/sigma = '}"
            f"{abs(o - p) / max(s, 1e-9):.1f})"
        )
    print(
        f"  within 1 sigma: {sum(covered)}/{len(covered)} "
        f"(~68% expected if calibrated), "
        f"mean sigma {sigma.mean():.4f} nm"
    )
    (FIT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    import pandas as pd

    pd.DataFrame(
        {
            "test_index": test_idx,
            "sample_id": [ids[i] for i in test_idx],
            "observed_rg_nm": observed,
            "predicted_rg_nm": predicted,
            "gp_sigma_nm": sigma,
            "linear_mean_predicted_rg_nm": predicted_linear,
            "linear_mean_sigma_nm": sigma_linear,
        }
    ).to_csv(FIT / "test_predictions.csv", index=False)

    figure, axis = plt.subplots(figsize=(5.5, 5.5))
    axis.errorbar(
        observed, predicted, yerr=sigma, fmt="o", markersize=7,
        alpha=0.75, color="C0", ecolor="C0", elinewidth=1.0, capsize=3,
        zorder=3, label="constant-mean LGP",
    )
    axis.errorbar(
        observed, predicted_linear, yerr=sigma_linear, fmt="s",
        markersize=7, alpha=0.75, color="C1", ecolor="C1", elinewidth=1.0,
        capsize=3, zorder=4, label="stickiness-mean LGP",
    )
    limits = [
        min(observed.min(), predicted.min(), predicted_linear.min()) - 0.03,
        max(observed.max(), predicted.max(), predicted_linear.max()) + 0.03,
    ]
    axis.plot(limits, limits, "k--", linewidth=1, zorder=1)
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_xlabel("observed mean Rg (nm), held-out")
    axis.set_ylabel("predicted Rg (nm)")
    axis.set_title(
        f"n={n} ({len(train_idx)}/{n_test} split): "
        f"constant R2 {metrics['test_metrics']['r2']:+.2f}, "
        f"stickiness R2 {metrics['linear_mean']['test_metrics']['r2']:+.2f}",
        fontsize=10,
    )
    axis.legend(fontsize=9)
    figure.tight_layout()
    figure.savefig(FIT / "held_out_predictions.png", dpi=160)
    print(f"outputs in {FIT}")


if __name__ == "__main__":
    main()
