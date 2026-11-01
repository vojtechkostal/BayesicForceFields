"""Long equilibrated Hst5 runs at both parameterizations; Rg comparison.

Runs 20 ns (Langevin, dt 0.01 ps, OpenCL) at each of:

- ``lambdas_0``    the M1 starting parameters, whose upstream cycle-1
                   snapshot has the simulated reference (Hst5 Rg 1.3236 nm,
                   5000 frames from 18 us over 10 replicas)
- ``lambdas``      the CALVADOS2-optimized parameters, compared with the
                   experimental value (1.38 +- 0.05 nm)

The first 5 ns of each run is discarded as equilibration, the rest saved
every 10 ps. Three seeds are run per parameterization and pooled with
pooled standard errors. Outputs go to ``longrun/<column>/seed<i>/`` and
the comparison to ``longrun/Rg_comparison.txt`` and ``Rg_comparison.png``.

Run from the repository root:

    python -m examples.calvados_gmx.run_long_hst5
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import openmm as mm  # noqa: E402
import openmm.app as app  # noqa: E402
import openmm.unit as unit  # noqa: E402
import pandas as pd  # noqa: E402
import io  # noqa: E402

from examples.calvados.fetch_data import RAW, verified_bytes  # noqa: E402
from examples.calvados_gmx import run_pilot_openmm as rp  # noqa: E402
from examples.calvados_gmx.build_pilot import debye  # noqa: E402

HERE = rp.HERE
OUT = HERE / "longrun"
SEEDS = (2026, 2027, 2028)
EQUIL_PS = 5000       # discarded
PROD_PS = 20_000      # kept per seed
REPORT_PS = 10        # between saved frames
EXPERIMENTAL_RG = 1.38
EXPERIMENTAL_ERR = 0.05


def upstream_rg0():
    snapshot = pd.read_pickle(
        io.BytesIO(verified_bytes(RAW / "1_init_proteinsRgs.pkl"))
    )
    row = snapshot.loc["Hst5"]
    return np.asarray(row.rgarray, dtype=float), float(row.Rg)


def run_one(column, seed):
    """20 ns at one lambdas column and seed; returns (rg array, meta)."""
    out_dir = OUT / column / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rg_file = out_dir / "rg.txt"
    if rg_file.is_file():
        return np.loadtxt(rg_file), "cached"

    rp.LAMBDA_COLUMN = column
    records, sequence = rp.particle_records()
    kappa, lB, RT = debye(rp.TEMP, rp.IONIC)
    system, box = rp.hst5_system(records)
    rp.add_pair_force(system, records, kappa, lB, RT)
    positions = rp.input_positions(records)
    topology = rp.build_topology(records, sequence)

    integrator = mm.LangevinMiddleIntegrator(
        rp.TEMP * unit.kelvin, 1.16, rp.DT * 0.5
    )
    integrator.setRandomNumberSeed(seed)
    platform_name = "OpenCL" if "OpenCL" in [
        mm.Platform.getPlatform(i).getName()
        for i in range(mm.Platform.getNumPlatforms())
    ] else "CPU"
    simulation = app.Simulation(
        topology, system, integrator,
        mm.Platform.getPlatformByName(platform_name),
    )
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(rp.TEMP * unit.kelvin, seed)

    print(f"{column}/seed{seed}: equilibration {EQUIL_PS} ps", flush=True)
    simulation.step(int(EQUIL_PS / (rp.DT * 0.5)))
    integrator.setStepSize(rp.DT)
    dcd_path = out_dir / "trajectory.dcd"
    dcd = app.DCDReporter(
        str(dcd_path), int(REPORT_PS / rp.DT)
    )
    simulation.reporters.append(dcd)
    n_steps = int(PROD_PS / rp.DT)
    print(f"{column}/seed{seed}: production {PROD_PS} ps on {platform_name}",
          flush=True)
    for block in range(0, n_steps, 50_000):
        simulation.step(min(50_000, n_steps - block))
    simulation.saveState(str(out_dir / "final_state.xml"))

    rp.write_pdb(out_dir / "hst5.pdb", records, positions, box)
    masses = np.array([r["mass"] for r in records])
    rg = rp.radius_of_gyration(out_dir / "hst5.pdb", dcd_path, masses)
    np.savetxt(rg_file, rg, fmt="%.4f")
    return rg, "ran"


def block_mean_sem(values, block_ps=1000, frame_ps=REPORT_PS):
    """Block average over ``block_ps`` windows; returns (mean, sem, n_blocks)."""
    block = int(block_ps / frame_ps)
    n_blocks = len(values) // block
    means = values[: n_blocks * block].reshape(n_blocks, block).mean(axis=1)
    return float(means.mean()), float(means.std(ddof=1) / np.sqrt(n_blocks)), n_blocks


def main():
    results = {}
    for column in ("lambdas_0", "lambdas"):
        pooled = []
        for seed in SEEDS:
            rg, status = run_one(column, seed)
            print(f"{column}/seed{seed}: {status}, {len(rg)} frames, "
                  f"mean {rg.mean():.4f} nm", flush=True)
            pooled.append(rg)
        results[column] = np.concatenate(pooled)

    theirs_frame, theirs_mean = upstream_rg0()

    lines = ["Hst5 Rg comparison (mass-weighted, nm)", "=" * 50]
    for column, label in (
        ("lambdas_0", "M1 starting parameters (engine validation)"),
        ("lambdas", "CALVADOS2 optimized parameters"),
    ):
        ours = results[column]
        mean, sem, n = block_mean_sem(ours)
        lines += [
            "",
            f"ours, {label}:",
            f"  n_frames {len(ours)}, pooled over {len(SEEDS)} seeds, "
            f"{PROD_PS // 1000} ns each (first {EQUIL_PS // 1000} ns discarded)",
            f"  Rg = {mean:.4f} +- {sem:.4f} nm "
            f"({n} x 1-ns block average)",
        ]
        if column == "lambdas_0":
            lines += [
                "  upstream simulated: "
                f"{theirs_mean:.4f} nm (5000 frames, 18 us, 10 replicas)",
                f"  difference: {mean - theirs_mean:+.4f} nm",
                f"  z-score vs upstream per-frame std: "
                f"{(mean - theirs_mean) / theirs_frame.std():+.2f}",
                "  experimental (for reference): "
                f"{EXPERIMENTAL_RG} +- {EXPERIMENTAL_ERR} nm",
            ]
        else:
            lines += [
                "  experimental: "
                f"{EXPERIMENTAL_RG} +- {EXPERIMENTAL_ERR} nm",
                f"  difference: {mean - EXPERIMENTAL_RG:+.4f} nm "
                f"({abs(mean - EXPERIMENTAL_RG) / EXPERIMENTAL_ERR:.1f} sigma)",
                f"  upstream simulated at lambdas_0 (not the same "
                f"parameters): {theirs_mean:.4f} nm",
            ]
    report = "\n".join(lines) + "\n"
    (OUT / "Rg_comparison.txt").write_text(report)
    print(report, end="")

    figure, axis = plt.subplots(figsize=(7, 4.5))
    for column, label, color in (
        ("lambdas_0", "ours @ M1 start (lambdas_0)", "C0"),
        ("lambdas", "ours @ optimized (lambdas)", "C2"),
    ):
        ours = results[column]
        mean, sem, _ = block_mean_sem(ours)
        axis.hist(ours, bins=60, range=(0.7, 2.1), density=True, alpha=0.4,
                  color=color, label=f"{label}: {mean:.3f}+-{sem:.3f}")
        axis.axvline(mean, color=color, linewidth=1.5)
    axis.hist(theirs_frame, bins=60, range=(0.7, 2.1), density=True,
              histtype="step", linewidth=1.5, color="k",
              label=f"upstream @ M1 start: {theirs_mean:.3f}")
    axis.axvline(EXPERIMENTAL_RG, color="r", linestyle="--", linewidth=1.2,
                 label=f"experiment: {EXPERIMENTAL_RG}+-{EXPERIMENTAL_ERR}")
    axis.set_xlabel("mass-weighted Rg (nm)")
    axis.set_ylabel("frame density")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(OUT / "Rg_comparison.png", dpi=160)
    print(f"outputs in {OUT}")


if __name__ == "__main__":
    main()
