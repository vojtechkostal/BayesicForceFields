"""Sobol campaign: 100 lambda vectors x 10 ns Hst5, one QoI dataset.

A first real training set for the stickiness surrogate: a Sobol sequence
covers the 20-dimensional lambda cube uniformly (scipy.stats.qmc), each
vector gets a fresh 10 ns Hst5 trajectory (2 ns discarded), and the
mass-weighted Rg of every frame is stored. The campaign writes a BFF
``QoIDataset`` whose X is the (n_samples, 20) lambda matrix in the GP
parameter order, y the per-sample mean Rg, and y_ref the experimental
value - ready for ``bff fit-lgp`` and ``bff learn``.

Sampling budget: 100 x (10 ns production + 2 ns equilibration) at dt 0.01,
single OpenMM context per sample, OpenCL platform if available. Every
sample caches to ``campaign/sobol_<i>/rg.txt`` and is skipped when it
already exists, so the campaign resumes after interruption.

Numbers shared with the pilot: same potential, conditions (293 K, pH 7.5,
I = 0.15 M), Hst5 sequence with X/Z terminal beads, global Langevin
friction rp.FRICTION_CACHED (1.16 1/ps; the upstream rate is
rp.FRICTION = 0.01 1/ps, see run_pilot_openmm).

Run from the repository root:

    python -m examples.calvados_gmx.run_sobol_campaign
"""

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

sys.path.insert(0, ".")
import openmm as mm  # noqa: E402
import openmm.app as app  # noqa: E402
import openmm.unit as unit  # noqa: E402

from examples.calvados.fetch_data import RAW, verified_bytes  # noqa: E402
from examples.calvados_gmx import run_pilot_openmm as rp  # noqa: E402
from examples.calvados_gmx.build_pilot import debye  # noqa: E402

HERE = rp.HERE
OUT = HERE / "campaign"
N_SAMPLES = 100
EQUIL_PS = 2000
PROD_PS = 10_000
REPORT_PS = 10
SEED = 2026
LOW = 0.001
HIGH = 0.999
EXPERIMENTAL_RG = 1.38
EXPERIMENTAL_ERR = 0.05

# The GP parameter order: lambda <THREE> sorted by one-letter code, the
# same convention as examples/calvados/train_discrepancy.py.
def parameter_names():
    residues = pd.read_csv(io.BytesIO(verified_bytes(RAW / "residues.csv")))
    one_to_three = dict(zip(residues["one"], residues["three"]))
    return tuple(
        f"lambda {one_to_three[one]}" for one in sorted(one_to_three)
    )


def lambda_vector(sample, names):
    """Map a Sobol point onto {'three': lambda}, matching parameter_names."""
    residues = pd.read_csv(io.BytesIO(verified_bytes(RAW / "residues.csv")))
    residues = residues.set_index("three")
    return {
        name.split()[1]: float(value)
        for name, value in zip(names, sample)
    }


def run_sample(index, lam_by_three):
    """One 10 ns trajectory; caches to campaign/sobol_<index>/."""
    out_dir = OUT / f"sobol_{index:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rg_file = out_dir / "rg.txt"
    if rg_file.is_file():
        return np.loadtxt(rg_file), 0.0

    rp.LAMBDA_COLUMN = "lambdas"  # base table; overridden per sample below
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
    platform_name = "OpenCL" if "OpenCL" in platforms else "CPU"
    simulation = app.Simulation(
        topology, system, integrator,
        mm.Platform.getPlatformByName(platform_name),
    )
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
        json.dumps(lam_by_three, indent=2) + "\n"
    )
    return rg, time.time() - started


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names = parameter_names()
    sampler = qmc.Sobol(d=len(names), scramble=True, seed=SEED)
    lambdas = LOW + (HIGH - LOW) * sampler.random(N_SAMPLES)
    np.savetxt(OUT / "sobol_points.txt", lambdas, fmt="%.6f")

    means = np.full(N_SAMPLES, np.nan)
    stds = np.full(N_SAMPLES, np.nan)
    manifest = []
    for index in range(N_SAMPLES):
        lam_by_three = lambda_vector(lambdas[index], names)
        rg, seconds = run_sample(index, lam_by_three)
        means[index] = rg.mean()
        stds[index] = rg.std()
        manifest.append(
            {
                "sample_id": f"sobol_{index:03d}",
                "rg_mean_nm": float(rg.mean()),
                "rg_std_nm": float(rg.std()),
                "seconds": seconds,
            }
        )
        print(
            f"sobol_{index:03d}: Rg {rg.mean():.4f} +- {rg.std():.4f} nm "
            f"({seconds:.0f} s)" if seconds else
            f"sobol_{index:03d}: Rg {rg.mean():.4f} nm (cached)",
            flush=True,
        )
    pd.DataFrame(manifest).to_csv(OUT / "manifest.csv", index=False)

    from bff.qoi.dataset import QoIDataset

    dataset = QoIDataset(
        name="rg_hst5",
        X=lambdas,
        y=means[:, None],
        y_ref=np.array([EXPERIMENTAL_RG]),
        parameter_names=names,
        sample_ids=tuple(f"sobol_{i:03d}" for i in range(N_SAMPLES)),
        metadata={
            "protein": "Hst5",
            "temperature_k": rp.TEMP,
            "ionic_m": rp.IONIC,
            "ph": 7.5,
            "equilibration_ps": EQUIL_PS,
            "production_ps": PROD_PS,
            "sobol_seed": SEED,
            "engine": "openmm",
            "experimental_rg_nm": EXPERIMENTAL_RG,
            "experimental_rg_err_nm": EXPERIMENTAL_ERR,
            "frame_std_nm": stds.tolist(),
            "qoi": "mass-weighted Rg, frame mean",
        },
    )
    dataset.write(str(OUT / "rg_hst5.pt"))
    print(f"QoI dataset: {OUT / 'rg_hst5.pt'}")
    print(
        f"Rg over the cube: min {np.nanmin(means):.3f}, "
        f"max {np.nanmax(means):.3f}, "
        f"spread {np.nanmax(means) - np.nanmin(means):.3f} nm "
        f"(experimental {EXPERIMENTAL_RG} +- {EXPERIMENTAL_ERR})"
    )


if __name__ == "__main__":
    main()
