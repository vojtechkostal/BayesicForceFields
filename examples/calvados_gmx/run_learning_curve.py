"""Learning-curve experiment: how many Sobol samples trains a good GP?

Extends the 100-point Sobol campaign with a larger design, then fits GPs
at increasing training-set sizes and reports held-out error at each size.
The 20/80 split is nested: every larger fit reuses the same test points,
so the metrics are comparable across sizes.

Protocol per point: identical to run_sobol_campaign (10 ns production,
2 ns equilibration, Rg frame mean). Existing campaign points are reused
via the cache; only new Sobol indices run MD.

Numbers to keep honest: a squared-exponential GP in 20 dimensions needs
the nearest-neighbor distance to training data to shrink, which scales
poorly (n^(-1/20)); the learning curve measures the real cost even if it
is steep. A plateau above the noise floor means the SE kernel cannot
exploit this data at any affordable size - see the fit8020 notes.

Run from the repository root:

    python -m examples.calvados_gmx.run_learning_curve
"""

import sys
import time
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
from examples.calvados_gmx import run_sobol_campaign as rc  # noqa: E402
from examples.calvados_gmx.fit_sobol_gp import regression_metrics  # noqa: E402

TARGET_N = 200          # total design size this round (100 already exist)
SIZES = (20, 50, 80, 120, 160, TARGET_N)
SEED = rc.SEED


def sobol_design(n_total):
    """The first ``n_total`` points of the scrambled Sobol sequence."""
    from scipy.stats import qmc

    names = rc.parameter_names()
    sampler = qmc.Sobol(d=len(names), scramble=True, seed=SEED)
    # power-of-two efficiency: draw the next power of two, keep the head
    n_pow2 = 1 << (n_total - 1).bit_length()
    points = sampler.random(n_pow2)[:n_total]
    return rc.LOW + (rc.HIGH - rc.LOW) * points


def run_missing(points):
    """Run every Sobol index not yet cached; returns nothing."""
    names = rc.parameter_names()
    for index in range(len(points)):
        lam = rc.lambda_vector(points[index], names)
        rc.OUT.mkdir(parents=True, exist_ok=True)
        rg, seconds = rc.run_sample(index, lam)
        tag = "ran" if seconds else "cached"
        print(f"sobol_{index:03d}: {tag}, Rg {rg.mean():.4f}", flush=True)


def fit_at_size(dataset, n_train_fit):
    """Train on the first ``n_train_fit`` samples of the design, test on
    the nested fixed tail; returns (metrics, observed, predicted)."""
    X = dataset.X
    y = dataset.y
    n_test = max(10, TARGET_N // 5)
    test_slice = slice(len(X) - n_test, len(X))
    train_indices = np.arange(0, n_train_fit)
    test_indices = np.arange(test_slice.start, test_slice.stop)

    class _View(QoIDataset):
        pass

    sub = QoIDataset(
        name="rg_hst5_curve",
        X=X[train_indices],
        y=y[train_indices].reshape(-1, 1) if y[train_indices].ndim == 1 else y[train_indices],
        y_ref=dataset.y_ref,
        parameter_names=dataset.parameter_names,
        sample_ids=tuple(
            f"sobol_{i:03d}" for i in range(len(train_indices))
        ),
        metadata=dict(dataset.metadata or {}),
    )
    out_dir = rc.OUT / f"curve_fit_n{n_train_fit}"
    out_dir.mkdir(exist_ok=True)
    torch.manual_seed(SEED)
    logger = Logger("fit-lgp", str(out_dir / "fit.log"), mode="w")
    model = fit_surrogates(
        [sub],
        model_paths={"rg_hst5_curve": out_dir / "model.lgp"},
        reuse_models=False,
        n_hyper_max=min(80, n_train_fit),
        committee_size=1,
        test_fraction=0.05,
        logger=logger,
        max_iter=500,
    )["rg_hst5_curve"]
    with torch.inference_mode():
        predicted = model.predict(
            torch.tensor(X[test_indices])
        ).numpy()[:, 0]
    observed = y[test_indices, 0]
    return regression_metrics(observed, predicted), observed, predicted


def main():
    torch.set_num_threads(4)
    points = sobol_design(TARGET_N)
    print(f"design: {TARGET_N} Sobol points (scrambled, seed {SEED})",
          flush=True)
    run_missing(points)

    dataset = QoIDataset.load(str(rc.OUT / "rg_hst5.pt"))
    print(f"dataset on disk has {dataset.n_samples} samples", flush=True)
    if dataset.n_samples < TARGET_N:
        # rebuild the dataset over the full design
        names = rc.parameter_names()
        means, stds = [], []
        for index in range(TARGET_N):
            rg, _ = rc.run_sample(index, rc.lambda_vector(points[index], names))
            means.append(rg.mean())
            stds.append(rg.std())
        dataset = QoIDataset(
            name="rg_hst5",
            X=points,
            y=np.asarray(means)[:, None],
            y_ref=np.array([rc.EXPERIMENTAL_RG]),
            parameter_names=names,
            sample_ids=tuple(f"sobol_{i:03d}" for i in range(TARGET_N)),
            metadata={
                "protein": "Hst5",
                "production_ps": rc.PROD_PS,
                "sobol_seed": SEED,
                "qoi": "learning-curve rebuild",
            },
        )
        dataset.write(str(rc.OUT / "rg_hst5.pt"))
        dataset = QoIDataset.load(str(rc.OUT / "rg_hst5.pt"))

    results = {}
    for n_train in SIZES:
        if n_train >= dataset.n_samples:
            continue
        minimum = dataset.n_samples - max(10, TARGET_N // 5)
        n_train = min(n_train, minimum)
        metrics, observed, predicted = fit_at_size(dataset, n_train)
        results[n_train] = metrics
        noise_floor = 0.05  # ~ per-sample SEM of the 10 ns mean
        print(
            f"n_train {n_train:4d}: RMSE {metrics['rmse_nm']:.4f} nm, "
            f"R2 {metrics['r2']:+.3f} (noise floor ~{noise_floor} nm, "
            f"signal spread {np.std(observed):.3f} nm)",
            flush=True,
        )
    import json

    (rc.OUT / "learning_curve.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    figure, axis = plt.subplots(figsize=(6, 4.5))
    ns = sorted(results)
    axis.plot(
        ns,
        [results[n]["rmse_nm"] for n in ns],
        "o-",
        label="held-out RMSE",
    )
    axis.axhline(0.05, color="gray", linestyle=":", label="~noise floor")
    axis.axhline(
        np.std(dataset.y), color="C3", linestyle="--",
        label="signal (std of Rg means)",
    )
    axis.set_xscale("log")
    axis.set_xlabel("training samples")
    axis.set_ylabel("RMSE (nm)")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(rc.OUT / "learning_curve.png", dpi=160)
    print(f"outputs: {rc.OUT / 'learning_curve.png'}")


if __name__ == "__main__":
    main()
