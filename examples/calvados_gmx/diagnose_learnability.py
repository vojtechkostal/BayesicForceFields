"""Two zero-MD experiments on the cached Sobol-200 campaign data.

1. Mean-lambda test: scatter each sample's mean Rg against its sequence-
   mean lambda (and a few other one-dimensional aggregates), with rank
   correlation and one-dimensional fits. If Rg tracks an aggregate, the
   20-D surface is effectively low-rank and the campaign can sample a few
   aggregate coordinates densely instead.

2. Synthetic control: a known-smooth 20-D function (squared-exponential,
   lengthscale 0.35), evaluated at the same 200 Sobol points with the same
   0.038 nm label noise, trained with the same 80/20 GP pipeline. If this
   recovers the function (R2 >> 0) at our density, our flat real-data
   curves mean the real surface is rough, not that the pipeline is blind.

Run from the repository root:

    python -m examples.calvados_gmx.diagnose_learnability
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bff.bayes.fit import fit_surrogates  # noqa: E402
from bff.io.logs import Logger  # noqa: E402
from bff.qoi.dataset import QoIDataset  # noqa: E402
from examples.calvados_gmx import run_sobol_campaign as rc  # noqa: E402
from examples.calvados_gmx.fit_sobol_gp import regression_metrics  # noqa: E402
from examples.calvados_gmx.run_learning_curve import sobol_design  # noqa: E402

NEG_LABEL_NOISE = 0.038  # split-half-measured SEM of a 10 ns mean Rg
OUT = rc.OUT / "diagnose"
NAMES = rc.parameter_names()


def cached_labels():
    """Mean Rg per Sobol index, from the campaign cache."""
    design = sobol_design(200)
    import pandas as pd

    manifest = pd.read_csv(rc.OUT / "manifest.csv")
    labels = {}
    for index in range(len(design)):
        rg_file = rc.OUT / f"sobol_{index:03d}" / "rg.txt"
        if rg_file.is_file():
            labels[index] = float(np.loadtxt(rg_file).mean())
    for _, row in manifest.iterrows():
        index = int(row["sample_id"].split("_")[1])
        labels.setdefault(index, float(row["rg_mean_nm"]))
    keep = sorted(labels)
    return design[keep], np.array([labels[i] for i in keep])


def mean_lambda_test(X, y):
    """Is mean Rg explained by low-dimensional aggregates of lambda?"""
    three = [n.split()[1] for n in NAMES]
    # hydrophobicity-weighted mean uses the sequence composition; all 20
    # residues appear in Hst5, so the plain mean is the natural aggregate.
    aggregates = {
        "mean lambda": X.mean(1),
        "max lambda": X.max(1),
        "min lambda": X.min(1),
        "mean of top-5": np.sort(X, 1)[:, -5:].mean(1),
        "variance of lambdas": X.var(1),
    }
    # PCA directions of the design itself (unsupervised slowest directions)
    centered = X - X.mean(0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    for k in range(3):
        aggregates[f"design PC{k + 1}"] = centered @ vh[k]

    from scipy.stats import spearmanr

    results = {}
    for name, values in aggregates.items():
        rho, p = spearmanr(values, y)
        results[name] = rho
        print(f"  {name:22s}: spearman {rho:+.3f} (p = {p:.1e})")
    figure, axes = plt.subplots(2, 4, figsize=(14, 7))
    for cell, (name, values) in enumerate(aggregates.items()):
        axis = axes.flat[cell]
        column = cell % 4
        if cell >= len(axes.flat):
            break
        axis.scatter(values, y, s=10, alpha=0.6)
        axis.set_title(f"{name}: rho={results[name]:+.2f}", fontsize=9)
        if column == 0:
            axis.set_ylabel("mean Rg (nm)")
        axis.set_xlabel(name)
    figure.tight_layout()
    figure.savefig(OUT / "mean_lambda_scatter.png", dpi=150)
    best = max(results, key=lambda k: abs(results[k]))
    print(f"  best aggregate: {best} (|rho| = {abs(results[best]):.3f})")
    print(f"  signal std {y.std():.4f} nm, label noise ~{NEG_LABEL_NOISE} nm")
    return results


def synthetic_control(X, y, smooth_lengthscales=(0.35, 0.6)):
    """Train the same pipeline on a known smooth function + equal noise."""
    rng = np.random.default_rng(2026)
    indices = torch.randperm(len(X), generator=torch.Generator().manual_seed(2026))
    n_test = len(X) // 5
    test_idx = indices[:n_test].numpy()
    train_idx = indices[n_test:].numpy()

    outputs = {}
    for ls in smooth_lengthscales:
        w = rng.normal(0, 1, X.shape[1])
        w = w / np.linalg.norm(w)
        d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1) / ls**2
        K = np.exp(-0.5 * d2)
        L = np.linalg.cholesky(K + 1e-8 * np.eye(len(X)))
        f = L @ rng.normal(0, y.std(), len(X))
        f = f - f.mean() + y.mean()          # same scale and center as Rg
        noisy = f + rng.normal(0, NEG_LABEL_NOISE, len(X))
        # clean test labels for the recoverable-function check
        clean_test = f[test_idx]

        dataset = QoIDataset(
            name="synthetic",
            X=X[train_idx],
            y=noisy[train_idx].reshape(-1, 1),
            y_ref=np.array([rc.EXPERIMENTAL_RG]),
            parameter_names=NAMES,
            sample_ids=tuple(f"syn_{i}" for i in range(len(train_idx))),
            metadata={"note": "synthetic smooth control"},
        )
        fit_dir = OUT / f"synthetic_ls{ls}"
        fit_dir.mkdir(exist_ok=True)
        torch.manual_seed(2026)
        logger = Logger("fit-lgp", str(fit_dir / "fit.log"), mode="w")
        model = fit_surrogates(
            [dataset],
            model_paths={"synthetic": fit_dir / "model.lgp"},
            reuse_models=False,
            n_hyper_max=80,
            committee_size=1,
            test_fraction=0.05,
            logger=logger,
            max_iter=500,
        )["synthetic"]
        with torch.inference_mode():
            predicted = model.predict(
                torch.tensor(X[test_idx], dtype=torch.float64)
            ).numpy()[:, 0]
        against_clean = regression_metrics(clean_test, predicted)
        against_noisy = regression_metrics(noisy[test_idx], predicted)
        learned_ls = float(
            np.mean(model.members[0].lengthscales.numpy())
        )
        outputs[ls] = (against_clean, against_noisy, learned_ls)
        print(
            f"  synthetic ls={ls}: R2 vs clean {against_clean['r2']:+.3f}, "
            f"RMSE {against_clean['rmse_nm']:.4f} | "
            f"vs noisy labels {against_noisy['r2']:+.3f} | "
            f"learned ls {learned_ls:.3f} (true {ls})"
        )
        print(
            f"    (function std {f.std():.4f} nm = our real signal scale; "
            f"residual R2 ceiling from noise ~ "
            f"{f.std() ** 2 / (f.std() ** 2 + NEG_LABEL_NOISE ** 2):.2f})"
        )
    return outputs


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    X, y = cached_labels()
    print(f"cached samples: {len(X)}")
    print(f"label stats: mean {y.mean():.4f}, std {y.std():.4f} nm")
    print()
    print("== 1. mean-lambda test")
    mean_lambda_test(X, y)
    print()
    print("== 2. synthetic smooth control")
    synthetic_control(X, y)


if __name__ == "__main__":
    main()
