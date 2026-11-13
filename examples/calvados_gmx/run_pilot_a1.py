"""CALVADOS2 pilot in OpenMM: protein A1 at the M1_24 optimized lambdas.

Same physics as run_pilot_openmm (Ashbaugh-Hatch + Yukawa, one bead per
residue, X/Z terminal pseudo-types), generalized from the hardcoded Hst5
sequence to the A1 FASTA from the upstream cycle-1 snapshot. A1 is the
long-chain label target for the bounded lambda campaign: 137 residues
with SAXS Rg = 2.76 +- 0.02 nm, so per-residue lambda perturbations
leverage a much larger Rg response than the 24-bead Hst5 pilot.

Conditions from the upstream snapshot: 298 K, ionic strength 0.15 M,
pH 7.0 (charges are the residues.csv defaults; ADAPT for protonation).
Residue charges do not vary with pH here.

Single long trajectory per the agreed protocol: 50 ns equilibration
(discarded) + 500 ns production at dt 0.01 ps, frames every 10 ps. The
50 ns precursor run (``pilot_a1_50ns/``) showed unresolved slow Rg mode
swings (2.1-2.9 nm on multi-ns timescales), hence the 10x extension.
The mean Rg over 1-ns blocks is compared with the CALVADOS2 reported
simulation value (2.637 nm, cycle-11 snapshot at the near-optimized
parameters) and experiment.

Outputs in ``examples/calvados_gmx/pilot_a1/``: input PDB, trajectory,
per-frame Rg, and ``Rg_comparison.txt``.

Run from the repository root:

    python -m examples.calvados_gmx.run_pilot_a1
"""

import io
from pathlib import Path

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import pandas as pd

from examples.calvados.fetch_data import RAW, verified_bytes
from examples.calvados_gmx import run_pilot_openmm as rp
from examples.calvados_gmx.build_pilot import debye, residues_table

HERE = rp.HERE
OUT = HERE / "pilot_a1"
PROTEIN = "A1"
LAMBDA_COLUMN = "lambdas"
SEED = 2026
TEMP = 298.0
IONIC = 0.15
EQUIL_PS = 50_000
PROD_PS = 500_000
REPORT_PS = 10


def upstream_record():
    """(fasta, exp_rg, exp_err, calvados2_rg) for A1 from the snapshots."""
    table = pd.read_csv(HERE.parent / "calvados" / "c2_rg.csv")
    row = table[table.iloc[:, 0] == PROTEIN].iloc[0]
    cycle11 = pd.read_pickle(
        io.BytesIO(verified_bytes(RAW / "11_init_proteinsRgs.pkl"))
    )
    return (
        str(row["fasta"]),
        float(row["exp_rg"]),
        float(row["exp_rg_err"]),
        float(cycle11.loc[PROTEIN, "Rg"]),
    )


def particle_records(fasta, lambda_column=LAMBDA_COLUMN):
    """One record per bead: X (N-term +1 e, +2 Da) ... Z (C-term -1 e,
    +16 Da), each carrying the upstream sigma/lambda/charge/mass."""
    residues = residues_table(lambda_column)
    sequence = list(fasta)
    types = list(sequence)
    types[0], types[-1] = "X", "Z"
    records = []
    for i, one in enumerate(types):
        base = sequence[i]
        row = residues.loc[base]
        q, mass = float(row.q), float(row.MW)
        if one == "X":
            q, mass = q + 1.0, mass + 2.0
        elif one == "Z":
            q, mass = q - 1.0, mass + 16.0
        records.append(
            {
                "type": one,
                "base": base,
                "sigma": float(row.sigmas),
                "lambda": float(row.lambdas),
                "q": q,
                "mass": mass,
            }
        )
    return records, sequence


def linear_system(records):
    """Bonds + periodic box, exactly the Hst5 pilot's system shape."""
    n = len(records)
    box = n * 0.38 + 1
    system = mm.System()
    for record in records:
        system.addParticle(record["mass"])
    from examples.calvados_gmx.build_pilot import BOND_K, BOND_R0

    bonds = mm.HarmonicBondForce()
    for i in range(n - 1):
        bonds.addBond(i, i + 1, BOND_R0, BOND_K)
    system.addForce(bonds)
    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(box, 0, 0), mm.Vec3(0, box, 0), mm.Vec3(0, 0, box)
    )
    return system, box


def write_pdb(path, records, box):
    """A minimal inspectable PDB of the input line of beads."""
    residues = residues_table(LAMBDA_COLUMN)
    three_of = residues["three"].to_dict()
    lines = ["REMARK   CALVADOS2 A1 pilot input: beads on a line"]
    for i, record in enumerate(records, start=1):
        z = (i - len(records) / 2) * 0.38
        lines.append(
            f"ATOM  {i:5d} {record['type']:>4s} {three_of[record['base']]:>3s}"
            f" A{i:4d}    {0.0:8.3f}{0.0:8.3f}{z:8.3f}"
            f"  1.00  0.00          {record['base']:>2s}"
        )
    lines.append(f"CRYST1{box:9.3f}{box:9.3f}{box:9.3f} 90.00 90.00 90.00")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def radius_of_gyration(pdb, dcd, masses, box):
    """Mass-weighted Rg per frame, wrap-safe.

    DCD coordinates are wrapped into the periodic box, so a chain whose
    center of mass crosses a box face is stored straddling the boundary
    and would give a wildly wrong Rg. Consecutive beads are bonded at
    0.38 nm - far below the box - so unwrapping by the minimum image of
    each backbone step is unambiguous.
    """
    import MDAnalysis as mda

    universe = mda.Universe(str(pdb), str(dcd))
    values = []
    for _ in universe.trajectory:
        # MDAnalysis positions are in Angstrom; convert to nm.
        positions = universe.atoms.positions / 10.0
        steps = np.diff(positions, axis=0)
        steps -= box * np.round(steps / box)
        positions[1:] = positions[0] + np.cumsum(steps, axis=0)
        com = np.average(positions, axis=0, weights=masses)
        values.append(
            float(np.sqrt(
                np.average(((positions - com) ** 2).sum(1), weights=masses)
            ))
        )
    return np.array(values)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rg_file = OUT / "rg.txt"
    fasta, exp_rg, exp_err, calvados_rg = upstream_record()
    records, sequence = particle_records(fasta)

    if rg_file.is_file():
        rg = np.loadtxt(rg_file)
        print(f"cached: {len(rg)} frames in {rg_file}", flush=True)
    else:
        kappa, lB, RT = debye(TEMP, IONIC)
        system, box = linear_system(records)
        rp.add_pair_force(system, records, kappa, lB, RT)
        positions = [
            mm.Vec3(0, 0, (i - len(records) / 2) * 0.38)
            for i in range(len(records))
        ]
        write_pdb(OUT / "a1.pdb", records, box)
        topology = rp.build_topology(records, sequence)

        integrator = mm.LangevinMiddleIntegrator(
            TEMP * unit.kelvin, rp.FRICTION_CACHED, rp.DT * 0.5
        )
        integrator.setRandomNumberSeed(SEED)
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
            TEMP * unit.kelvin, SEED
        )
        print(
            f"{PROTEIN}: equilibration {EQUIL_PS} ps at dt {rp.DT * 0.5} ps "
            f"on {platform_name}",
            flush=True,
        )
        simulation.step(int(EQUIL_PS / (rp.DT * 0.5)))
        integrator.setStepSize(rp.DT)
        simulation.reporters.append(
            app.DCDReporter(
                str(OUT / "trajectory.dcd"), int(REPORT_PS / rp.DT)
            )
        )
        print(f"{PROTEIN}: production {PROD_PS} ps at dt {rp.DT} ps", flush=True)
        n_steps = int(PROD_PS / rp.DT)
        start = pd.Timestamp.now()
        for block in range(0, n_steps, 100_000):
            simulation.step(min(100_000, n_steps - block))
            print(
                f"  {min(block + 100_000, n_steps)}/{n_steps} steps "
                f"({pd.Timestamp.now() - start})",
                flush=True,
            )
        simulation.saveState(str(OUT / "final_state.xml"))
        masses = np.array([record["mass"] for record in records])
        rg = radius_of_gyration(OUT / "a1.pdb", OUT / "trajectory.dcd", masses, box)
        np.savetxt(rg_file, rg, fmt="%.4f")

    block = int(1000 / REPORT_PS)
    n_blocks = len(rg) // block
    means = rg[: n_blocks * block].reshape(n_blocks, block).mean(axis=1)
    mean, sem = means.mean(), means.std(ddof=1) / np.sqrt(n_blocks)

    report = [
        f"A1 Rg (mass-weighted, nm), CALVADOS2 optimized lambdas",
        f"{'=' * 60}",
        f"ours: {mean:.4f} +- {sem:.4f} nm ({n_blocks} x 1-ns block average;",
        f"  single {PROD_PS // 1000} ns trajectory, {EQUIL_PS // 1000} ns equilibration discarded)",
        f"CALVADOS2 reported (cycle-11 snapshot): {calvados_rg:.4f} nm",
        f"  difference: {mean - calvados_rg:+.4f} nm",
        f"experiment (SAXS): {exp_rg:.2f} +- {exp_err:.2f} nm",
        f"  difference: {mean - exp_rg:+.4f} nm "
        f"({abs(mean - exp_rg) / max(sem, exp_err):.1f} sigma)",
    ]
    text = "\n".join(report) + "\n"
    (OUT / "Rg_comparison.txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
