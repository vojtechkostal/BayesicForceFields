"""Hst5 Sobol campaign: 500 bounded 9-D lambda vectors x 50 ns each.

The second training set for the stickiness surrogate, redesigned after
the diagnosis of the first (100-point full-cube) campaign:

- per-protein dimension: only the 9 lambda types Hst5 actually contains
  (A D F G H K R S Y; E is pinned at 0 by lambda_bounds.FIXED) - the
  20-D campaign burned GP capacity on residues that never appear in the
  sequence
- hard per-residue bounds from ``lambda_bounds.BOUNDS`` (user-estimated
  intervals around the CALVADOS2 optima) instead of the full cube
- 50 ns production per point (5x the first campaign): the split-half
  audit put SEM(10 ns) = 0.038 nm at the level of the in-cube signal;
  SEM scales as 1/sqrt(T), so 50 ns targets ~0.017 nm
- 2 ns equilibration as before: the longrun validation showed Hst5
  equilibrates in a few ns (tau(Rg) ~ 60 ps)

Resumable: every point caches to ``campaign2/hst_<i>/rg.txt`` and is
skipped when present, so the script survives interruption and can be
split across machines. The final QoIDataset X is the (n, 9) lambda
matrix in the sorted ``lambda <THREE>`` order of Hst5's residues, y the
per-point mean Rg, y_ref the experimental value.

Run from the repository root:

    python -m examples.calvados_gmx.run_campaign2

The cached points in campaign2/ ran at rp.FRICTION_CACHED (1.16 /ps).
To run the same design at the upstream collision rate into a new folder:

    python -m examples.calvados_gmx.run_campaign2 --friction 0.01 \
        --out examples/calvados_gmx/campaign2_rate0.01
"""

import csv
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
import pandas as pd  # noqa: E402

from examples.calvados.fetch_data import RAW, verified_bytes  # noqa: E402
from examples.calvados_gmx import run_pilot_openmm as rp  # noqa: E402
from examples.calvados_gmx import run_sobol_campaign as rc  # noqa: E402
from examples.calvados_gmx.build_pilot import debye  # noqa: E402
from examples.calvados_gmx.lambda_bounds import (  # noqa: E402
    BOUNDS,
    FIXED,
    free_ones,
)

HERE = rp.HERE
OUT = HERE / "campaign2"
HST5 = "DSHAKRHHGYKRKFHEKHHSHRGY"
N_SAMPLES = 500
EQUIL_PS = 2_000
PROD_PS = 50_000
REPORT_PS = 10
SEED = 2026
THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


def parameter_names(residues):
    """``lambda <THREE>`` for each free residue, sorted by one-letter."""
    return tuple(f"lambda {THREE[one]}" for one in sorted(residues))


def sobol_points(residues, n):
    """Bounded scrambled-Sobol design over the free residues only."""
    from scipy.stats import qmc

    lows = np.array([BOUNDS[one][0] for one in sorted(residues)])
    highs = np.array([BOUNDS[one][1] for one in sorted(residues)])
    sampler = qmc.Sobol(d=len(residues), scramble=True, seed=SEED)
    n_pow2 = 1 << (n - 1).bit_length()
    return lows + (highs - lows) * sampler.random(n_pow2)[:n]


def lambda_by_three(sample, residues):
    """Full 20-residue lambda dict: sampled for Hst5's free residues,
    FIXED for pinned ones, CALVADOS2 optima for absent residues (they
    never enter the potential for this sequence)."""
    fixed_values = {"E": FIXED["E"][0]}
    optima = {
        row["one"]: float(row["lambdas"])
        for row in csv.DictReader(
            io.StringIO(verified_bytes(RAW / "residues.csv").decode())
        )
    }
    values = dict(fixed_values)
    for one, value in zip(sorted(residues), sample):
        values[one] = float(value)
    for one in THREE:
        if one not in values:
            values[one] = optima[one]
    return {THREE[one]: value for one, value in values.items()}


def run_sample(index, lam_by_three, residues, friction=rp.FRICTION_CACHED,
               out=OUT):
    """One 50 ns trajectory; caches to <out>/hst_<index>/ (campaign2/ by default)."""
    out_dir = out / f"hst_{index:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rg_file = out_dir / "rg.txt"
    if rg_file.is_file():
        return np.loadtxt(rg_file), 0.0

    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit

    rp.LAMBDA_COLUMN = "lambdas"
    records, sequence = rp.particle_records()
    for record in records:
        base = record["type"] if record["type"] not in ("X", "Z") else (
            sequence[0] if record["type"] == "X" else sequence[-1]
        )
        record["lambda"] = lam_by_three[THREE[base]]

    kappa, lB, RT = debye(rp.TEMP, rp.IONIC)
    system, box = rp.hst5_system(records)
    rp.add_pair_force(system, records, kappa, lB, RT)
    positions = rp.input_positions(records)
    topology = rp.build_topology(records, sequence)

    started = time.time()
    integrator = mm.LangevinMiddleIntegrator(
        rp.TEMP * unit.kelvin, friction, rp.DT * 0.5
    )
    integrator.setRandomNumberSeed(SEED + index)
    platforms = [
        mm.Platform.getPlatform(i).getName()
        for i in range(mm.Platform.getNumPlatforms())
    ]
    # Prefer the GPU: OpenCL on macOS (Apple Silicon), CUDA elsewhere.
    # Fall back to CPU if the OpenCL context cannot be created.
    platform_name = "CUDA" if "CUDA" in platforms else (
        "OpenCL" if "OpenCL" in platforms else "CPU"
    )
    try:
        simulation = app.Simulation(
            topology, system, integrator,
            mm.Platform.getPlatformByName(platform_name),
        )
    except mm.OpenMMException:
        platform_name = "CPU"
        simulation = app.Simulation(
            topology, system, integrator,
            mm.Platform.getPlatformByName(platform_name),
        )
    print(f"hst_{index:03d}: running on {platform_name}", flush=True)
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(
        rp.TEMP * unit.kelvin, SEED + index
    )
    simulation.step(int(EQUIL_PS / (rp.DT * 0.5)))
    integrator.setStepSize(rp.DT)
    dcd_path = out_dir / "trajectory.dcd"
    simulation.reporters.append(
        app.DCDReporter(str(dcd_path), int(REPORT_PS / rp.DT))
    )
    simulation.step(int(PROD_PS / rp.DT))

    rp.write_pdb(out_dir / "hst5.pdb", records, positions, box)
    masses = np.array([r["mass"] for r in records])
    rg = rp.radius_of_gyration(out_dir / "hst5.pdb", dcd_path, masses)
    np.savetxt(rg_file, rg, fmt="%.4f")
    (out_dir / "lambda.json").write_text(
        json.dumps(lam_by_three, indent=2, sort_keys=True) + "\n"
    )
    return rg, time.time() - started


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--friction", type=float, default=rp.FRICTION_CACHED,
                        help="Langevin collision rate, 1/ps (upstream: 0.01)")
    parser.add_argument("--out", type=Path, default=OUT,
                        help="output folder (default: campaign2/)")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    residues = [one for one in sorted(set(HST5)) if one not in FIXED]
    names = parameter_names(residues)
    points = sobol_points(residues, N_SAMPLES)
    np.savetxt(out / "sobol_points.txt", points, fmt="%.6f")
    print(
        f"design: {N_SAMPLES} Sobol points in {len(residues)}-D box "
        f"(free residues {''.join(residues)})",
        flush=True,
    )

    means = np.full(N_SAMPLES, np.nan)
    stds = np.full(N_SAMPLES, np.nan)
    manifest = []
    for index in range(N_SAMPLES):
        lam = lambda_by_three(points[index], residues)
        rg, seconds = run_sample(index, lam, residues, args.friction, out)
        means[index] = rg.mean()
        stds[index] = rg.std()
        manifest.append(
            {
                "sample_id": f"hst_{index:03d}",
                "rg_mean_nm": float(rg.mean()),
                "rg_std_nm": float(rg.std()),
                "seconds": seconds,
            }
        )
        print(
            f"hst_{index:03d}: Rg {rg.mean():.4f} +- {rg.std():.4f} nm "
            + (f"({seconds:.0f} s)" if seconds else "(cached)"),
            flush=True,
        )
    pd.DataFrame(manifest).to_csv(out / "manifest.csv", index=False)

    from bff.qoi.dataset import QoIDataset

    dataset = QoIDataset(
        name="rg_hst5_bounded",
        X=points,
        y=means[:, None],
        y_ref=np.array([rc.EXPERIMENTAL_RG]),
        parameter_names=names,
        sample_ids=tuple(f"hst_{i:03d}" for i in range(N_SAMPLES)),
        metadata={
            "protein": "Hst5",
            "sequence": HST5,
            "free_residues": "".join(residues),
            "bounds": {one: BOUNDS[one] for one in residues},
            "fixed_lambdas": {k: v[0] for k, v in FIXED.items()},
            "temperature_k": rp.TEMP,
            "ionic_m": rp.IONIC,
            "ph": 7.5,
            "equilibration_ps": EQUIL_PS,
            "production_ps": PROD_PS,
            "sobol_seed": SEED,
            "engine": "openmm",
            "collision_rate_per_ps": args.friction,
            "experimental_rg_nm": rc.EXPERIMENTAL_RG,
            "experimental_rg_err_nm": rc.EXPERIMENTAL_ERR,
            "frame_std_nm": stds.tolist(),
            "qoi": "mass-weighted Rg, frame mean over 50 ns",
        },
    )
    dataset.write(str(out / "rg_hst5_bounded.pt"))
    print(f"QoI dataset: {out / 'rg_hst5_bounded.pt'}")
    print(
        f"Rg over the box: min {np.nanmin(means):.3f}, "
        f"max {np.nanmax(means):.3f}, "
        f"spread {np.nanmax(means) - np.nanmin(means):.3f} nm "
        f"(experimental {rc.EXPERIMENTAL_RG} +- {rc.EXPERIMENTAL_ERR})"
    )


if __name__ == "__main__":
    main()
