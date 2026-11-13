"""Shotgun study: local Sobol boxes around the closest-to-experiment point.

Successively shrinks the sampling box around the lambda vector whose Hst5
Rg came closest to experiment (campaign point sobol_026, Rg 1.387 nm) and
re-runs the surrogate feasibility test inside each box:

- side 0.40 (plus/minus 0.20 per lambda)  - "open choke"
- side 0.20 (plus/minus 0.10)             - tightened
- side 0.10 (plus/minus 0.05)             - "full choke"

Each box gets 100 fresh scrambled-Sobol points (never the same lambda
vectors as the global campaign), the same 10 ns protocol per point, an
80/20 GP fit, and a nested local learning curve (n_train 20/40/60/80) so
the local log-log slope can be compared with the global one (-0.026).

Outputs per box in ``campaign/shotgun/side_<s>/``: points, manifest,
fit metrics, learning-curve json, and ``SHOTGUN.md`` at the study root.

Run from the repository root:

    python -m examples.calvados_gmx.run_shotgun_study
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import qmc  # noqa: E402

from bff.bayes.fit import fit_surrogates  # noqa: E402
from bff.io.logs import Logger  # noqa: E402
from bff.qoi.dataset import QoIDataset  # noqa: E402
from examples.calvados_gmx import run_sobol_campaign as rc  # noqa: E402
from examples.calvados_gmx.fit_sobol_gp import regression_metrics  # noqa: E402

SEED = 2026
SIDES = (0.40, 0.20, 0.10)
POINTS_PER_BOX = 100
EQUIL_SCALE = 1.0  # same protocol as the global campaign
OUT = rc.OUT / "shotgun"
GLOBAL_BEST_INDEX = 26   # sobol_026, the closest-to-experiment point
LOCAL_SIZES = (20, 40, 60, 80)


def center_vector():
    """The sobol_026 lambda vector, from the campaign's saved design."""
    from examples.calvados_gmx.run_learning_curve import sobol_design

    return sobol_design(200)[GLOBAL_BEST_INDEX]


def box_design(center, side, n):
    """Scrambled Sobol inside the per-dimension box centered on ``center``."""
    sampler = qmc.Sobol(d=len(center), scramble=True, seed=SEED + int(1000 * side))
    unit = sampler.random(n)
    lower = np.clip(center - side / 2, rc.LOW, rc.HIGH)
    upper = np.clip(center + side / 2, rc.LOW, rc.HIGH)
    return lower + (upper - lower) * unit


def run_box_points(points, names, box_dir):
    """Run every point of one box (cache lives inside the box directory)."""
    means, stds = [], []
    manifest = []
    for index, lam_unit in enumerate(points):
        lam_by_three = rc.lambda_vector(lam_unit, names)
        out_dir = box_dir / f"pt_{index:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        rg_file = out_dir / "rg.txt"
        if rg_file.is_file():
            rg = np.loadtxt(rg_file)
            seconds = 0.0
        else:
            rg, seconds = run_local_sample(out_dir, lam_by_three, index)
        means.append(float(rg.mean()))
        stds.append(float(rg.std()))
        manifest.append(
            {
                "point": index,
                "rg_mean_nm": float(rg.mean()),
                "rg_std_nm": float(rg.std()),
                "seconds": seconds,
            }
        )
        print(
            f"  pt_{index:03d}: Rg {rg.mean():.4f} nm"
            + ("" if seconds else " (cached)"),
            flush=True,
        )
    return np.array(means), np.array(stds), manifest


def run_local_sample(out_dir, lam_by_three, index):
    """One 10 ns trajectory in a specific directory (separate cache)."""
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
    import time

    from examples.calvados_gmx import run_pilot_openmm as rp
    from examples.calvados_gmx.build_pilot import debye

    rp.LAMBDA_COLUMN = "lambdas"
    records, sequence = rp.particle_records()
    three_of = {
        "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
        "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
        "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
        "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
    }
    for record in records:
        base = record["type"] if record["type"] not in ("X", "Z") else (
            sequence[0] if record["type"] == "X" else sequence[-1]
        )
        record["lambda"] = lam_by_three[three_of[base]]

    kappa, lB, RT = debye(rp.TEMP, rp.IONIC)
    system, box = rp.hst5_system(records)
    rp.add_pair_force(system, records, kappa, lB, RT)
    positions = rp.input_positions(records)
    topology = rp.build_topology(records, sequence)

    started = time.time()
    integrator = mm.LangevinMiddleIntegrator(
        rp.TEMP * unit.kelvin, rp.FRICTION_CACHED, rp.DT * 0.5
    )
    integrator.setRandomNumberSeed(SEED + index)
    platforms = [
        mm.Platform.getPlatform(i).getName()
        for i in range(mm.Platform.getNumPlatforms())
    ]
    simulation = app.Simulation(
        topology, system, integrator,
        mm.Platform.getPlatformByName("OpenCL" if "OpenCL" in platforms else "CPU"),
    )
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(rp.TEMP * unit.kelvin, SEED + index)
    simulation.step(int(rc.EQUIL_PS / (rp.DT * 0.5)))
    integrator.setStepSize(rp.DT)
    dcd_path = out_dir / "trajectory.dcd"
    simulation.reporters.append(
        app.DCDReporter(str(dcd_path), int(rc.REPORT_PS / rp.DT))
    )
    simulation.step(int(rc.PROD_PS / rp.DT))
    rp.write_pdb(out_dir / "hst5.pdb", records, positions, box)
    masses = np.array([r["mass"] for r in records])
    rg = rp.radius_of_gyration(out_dir / "hst5.pdb", dcd_path, masses)
    np.savetxt(out_dir / "rg.txt", rg, fmt="%.4f")
    return rg, time.time() - started


def fit_and_curve(X, y, names, box_dir, tag):
    """80/20 fit plus a nested local learning curve in the box."""
    n = len(X)
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))
    n_test = n // 5
    test_idx = indices[:n_test].numpy()
    train_idx = indices[n_test:].numpy()

    results = {}
    observed = y[test_idx]
    for n_train in LOCAL_SIZES + (n - n_test,):
        if n_train > len(train_idx):
            continue
        sub_train = train_idx[:n_train]
        dataset = QoIDataset(
            name=f"shotgun_{tag}",
            X=X[sub_train],
            y=y[sub_train].reshape(-1, 1),
            y_ref=np.array([rc.EXPERIMENTAL_RG]),
            parameter_names=names,
            sample_ids=tuple(f"pt_{i:03d}" for i in sub_train),
            metadata={"box_side": tag},
        )
        fit_dir = box_dir / f"fit_n{n_train}"
        fit_dir.mkdir(exist_ok=True)
        torch.manual_seed(SEED)
        logger = Logger("fit-lgp", str(fit_dir / "fit.log"), mode="w")
        model = fit_surrogates(
            [dataset],
            model_paths={dataset.name: fit_dir / "model.lgp"},
            reuse_models=False,
            n_hyper_max=min(40, n_train),
            committee_size=1,
            test_fraction=0.05,
            logger=logger,
            max_iter=500,
        )[dataset.name]
        with torch.inference_mode():
            predicted = model.predict(torch.tensor(X[test_idx], dtype=torch.float64)).numpy()[:, 0]
        baseline = np.full_like(observed, y[sub_train].mean())
        metrics = regression_metrics(observed, predicted)
        metrics["baseline_r2"] = regression_metrics(observed, baseline)["r2"]
        lengthscales = model.members[0].lengthscales.numpy() if hasattr(
            model.members[0], "lengthscales") else None
        results[n_train] = {
            **metrics,
            "mean_lengthscale": None if lengthscales is None else float(np.mean(lengthscales)),
        }
        print(
            f"  [fit n={n_train :3d}] RMSE {metrics['rmse_nm']:.4f} nm, "
            f"R2 {metrics['r2']:+.3f} (baseline R2 {metrics['baseline_r2']:+.3f}"
            + (
                f", mean lengthscale {results[n_train]['mean_lengthscale']:.3f}"
                if lengthscales is not None else ""
            ) + ")",
            flush=True,
        )
    return results


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    names = rc.parameter_names()
    center = center_vector()
    print(f"center: sobol_{GLOBAL_BEST_INDEX} "
          f"(Rg 1.387 nm, closest to experiment 1.38)", flush=True)

    summary = {}
    for side in SIDES:
        tag = f"side_{side:.2f}".replace(".", "")
        box_dir = OUT / tag
        box_dir.mkdir(exist_ok=True)
        print(f"\n=== box side {side} ({tag}) ===", flush=True)
        points = box_design(center, side, POINTS_PER_BOX)
        np.savetxt(box_dir / "points.txt", points, fmt="%.6f")
        means, stds, manifest = run_box_points(points, names, box_dir)
        np.savetxt(box_dir / "rg_means.txt", means, fmt="%.4f")
        import pandas as pd

        pd.DataFrame(manifest).to_csv(box_dir / "manifest.csv", index=False)
        print(
            f"box Rg means: min {means.min():.3f}, max {means.max():.3f}, "
            f"std {means.std():.3f} nm",
            flush=True,
        )
        results = fit_and_curve(points, means, names, box_dir, tag)
        summary[side] = results
        (box_dir / "learning_curve.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    figure, axis = plt.subplots(figsize=(7, 5))
    for side, results in summary.items():
        ns = sorted(results)
        axis.plot(
            ns,
            [results[n]["rmse_nm"] for n in ns],
            "o-",
            label=f"box side {side}: slope "
            f"{np.polyfit(np.log(ns), np.log([results[n]['rmse_nm'] for n in ns]), 1)[0]:+.3f}",
        )
    axis.axhline(0.05, color="gray", linestyle=":", label="~noise floor")
    axis.set_xscale("log")
    axis.set_xlabel("training samples in box")
    axis.set_ylabel("held-out RMSE (nm)")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(OUT / "shotgun_curves.png", dpi=160)
    print(f"\noutputs: {OUT}")


if __name__ == "__main__":
    main()
