"""Active-learning pilot: expected improvement over the Sobol baseline.

Compares, at a fixed budget of 200 evaluated lambda vectors, passive Sobol
sampling against sequential acquisition by expected improvement (EI),
with the experimental target Rg = 1.38 nm as the objective. Both arms use
the same seed set (the first 150 Sobol points) and are scored on the same
never-queried Sobol tail, so the comparison is apples-to-apples.

BFF's GP predicts means only (torch.no_grad, no predictive variance
exposed), so uncertainty comes from a committee: sigma is the standard
deviation across ``COMMITTEE`` independently fitted members, and EI uses
that sigma in the standard improvement algebra:

    gap      = |Rg_target - best predicted Rg distance so far|   (>= 0)
    EI(a)    = gap * Phi(gap / sigma(a)) + sigma(a) * phi(gap / sigma(a))

where for each candidate a with committee mean m(a) and std sigma(a),
its own gap to the target is used on improvement over the *current best*
incumbent: I(a) = max(0, |Rg_target - incumbent| - |Rg_target - m(a)|).
When sigma is tiny everywhere EI reduces to pure exploitation; when gaps
dominate, to pure variance - the requested behaviour.

Outputs in ``campaign/active/``: per-iteration log (acquisition values,
chosen point), the AL-trained model, and a comparison table against the
passive run on the shared test tail.

Run from the repository root (the Sobol-200 baseline should exist first):

    python -m examples.calvados_gmx.run_active_learning
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
from scipy.stats import norm  # noqa: E402

from bff.bayes.fit import fit_surrogates  # noqa: E402
from bff.io.logs import Logger  # noqa: E402
from bff.qoi.dataset import QoIDataset  # noqa: E402
from examples.calvados_gmx import run_sobol_campaign as rc  # noqa: E402
from examples.calvados_gmx.fit_sobol_gp import regression_metrics  # noqa: E402
from examples.calvados_gmx.run_learning_curve import sobol_design  # noqa: E402

SEED_SET = 150          # shared seed budget for both arms
TOTAL_BUDGET = 200      # evaluated points per arm
CANDIDATE_POOL = 512    # Sobol points scored at each acquisition step
COMMITTEE = 5           # members per committee fit
TARGET_RG = 1.38        # experimental Rg; the acquisition objective
OUT = rc.OUT / "active"


def committee_sigma(models, X):
    """Model-disagreement sigma and mean over committee members."""
    with torch.inference_mode():
        predictions = np.stack(
            [m.predict(torch.tensor(X, dtype=torch.float64)).numpy()[:, 0]
             for m in models]
        )
    return predictions.mean(0), predictions.std(0)


def fit_committee(X, y, names, tag):
    """Fit ``COMMITTEE`` independent members on the same data."""
    members = []
    out_dir = OUT / f"fits_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for member in range(COMMITTEE):
        dataset = QoIDataset(
            name=f"rg_hst5_al_{member}",
            X=X,
            y=np.asarray(y).reshape(-1, 1),
            y_ref=np.array([TARGET_RG]),
            parameter_names=names,
            sample_ids=tuple(f"al_{i:03d}" for i in range(len(X))),
            metadata={"committee_member": member},
        )
        torch.manual_seed(rc.SEED + 31 * member)
        logger = Logger("fit-lgp", str(out_dir / f"m{member}.log"), mode="w")
        members.append(
            fit_surrogates(
                [dataset],
                model_paths={dataset.name: out_dir / f"m{member}.lgp"},
                reuse_models=False,
                n_hyper_max=80,
                committee_size=1,
                test_fraction=0.05,
                logger=logger,
                max_iter=500,
            )[dataset.name]
        )
    return members


def evaluate_point(index, points, names):
    """Run (or reuse the cache for) one lambda vector; returns mean Rg."""
    rg, seconds = rc.run_sample(index, rc.lambda_vector(points[index], names))
    return float(rg.mean()), seconds


def expected_improvement(mean, sigma, incumbent_distance):
    """EI of closing the gap to the target beyond the incumbent."""
    candidate_distance = np.abs(TARGET_RG - mean)
    improvement = np.maximum(0.0, incumbent_distance - candidate_distance)
    safe_sigma = np.maximum(sigma, 1e-12)
    z = improvement / safe_sigma
    return improvement * norm.cdf(z) + safe_sigma * norm.pdf(z)


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    names = rc.parameter_names()
    pool_points = sobol_design(TOTAL_BUDGET + CANDIDATE_POOL)
    # Test tail: a fixed Sobol block that neither arm queries. The passive
    # arm evaluates 0..199 of the same scrambled sequence, so its 200th
    # point on is untouched by both designs and shared for scoring.
    test_indices = list(range(TOTAL_BUDGET, TOTAL_BUDGET + 24))

    X_seed = pool_points[:SEED_SET]
    X_eval = np.zeros((TOTAL_BUDGET, len(names)))
    y_eval = np.zeros(TOTAL_BUDGET)
    queried = set(range(SEED_SET))
    for i in range(SEED_SET):
        y, _ = evaluate_point(i, pool_points, names)
        y_eval[i] = y
    print(f"seed set: {SEED_SET} Sobol points evaluated", flush=True)

    log = []
    for step in range(SEED_SET, TOTAL_BUDGET):
        members = fit_committee(
            X_eval[:step][:, :], y_eval[:step], names, tag=f"step{step:03d}"
        )
        incumbent = float(np.min(np.abs(TARGET_RG - y_eval[:step])))
        candidates = [
            i for i in range(len(pool_points)) if i not in queried
        ]
        X_cand = pool_points[candidates]
        mean, sigma = committee_sigma(members, X_cand)
        ei = expected_improvement(mean, sigma, incumbent)
        best = int(np.argmax(ei))
        chosen = candidates[best]
        y, seconds = evaluate_point(chosen, pool_points, names)
        X_eval[step] = pool_points[chosen]
        y_eval[step] = y
        queried.add(chosen)
        log.append(
            {
                "step": step,
                "sobol_index": chosen,
                "committee_mean_rg": float(mean[best]),
                "committee_sigma_rg": float(sigma[best]),
                "expected_improvement": float(ei[best]),
                "incumbent_distance_nm": incumbent,
                "actual_rg_nm": y,
                "seconds": seconds,
            }
        )
        (OUT / "acquisition_log.json").write_text(
            json.dumps(log, indent=2) + "\n"
        )
        print(
            f"step {step:3d}: chose sobol_{chosen:03d} "
            f"(EI {ei[best]:.4f}, sigma {sigma[best]:.4f}, "
            f"mean {mean[best]:.3f}) -> Rg {y:.4f} nm, "
            f"incumbent distance {incumbent:.4f} nm",
            flush=True,
        )

    # final committee and shared-tail comparison
    members = fit_committee(X_eval, y_eval, names, tag="final")
    criterion = np.array(test_indices)
    X_test = pool_points[criterion]
    mean, _ = committee_sigma(members, X_test)
    observed = []
    for c in test_indices:
        y, _ = evaluate_point(c, pool_points, names)
        observed.append(y)
    observed = np.array(observed)
    al_metrics = regression_metrics(observed, mean)
    (OUT / "metrics.json").write_text(
        json.dumps(
            {
                "arm": "active (EI, committee of 5)",
                "seed_set": SEED_SET,
                "total_budget": TOTAL_BUDGET,
                "test_points": len(test_indices),
                "metrics": al_metrics,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"AL arm: RMSE {al_metrics['rmse_nm']:.4f} nm on "
        f"{len(test_indices)} shared-tail test points",
        flush=True,
    )


if __name__ == "__main__":
    main()
