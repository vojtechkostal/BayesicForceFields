"""CALVADOS2 pilot in OpenMM: Hst5 at the M1_24 optimized lambdas.

Reproduces upstream simulate.py (HOOMD + azplugins ashbaugh) exactly, in
one readable place:

- one bead per residue, terminal beads X (+1 e, +2 Da) and Z (-1 e, +16 Da)
- Ashbaugh-Hatch pair energy
    U = 4*eps*[(sig_ij/r)^12-(sig_ij/r)^6] + eps*(1-lam_ij)
        for r <= 2^(1/6)*sig_ij
    U = 4*eps*lam_ij*[(sig_ij/r)^12-(sig_ij/r)^6]
        for r >  2^(1/6)*sig_ij
  energy-shifted to zero at 2.4 nm (upstream mode='shift')
- sig_ij and lam_ij are arithmetic means of the per-residue values
- Yukawa (Debye-Huckel) electrostatics q_i q_j lB RT exp(-kappa r)/r,
  energy-shifted to zero at 4.0 nm
- harmonic bonds k = 8033 kJ/mol/nm^2, r0 = 0.38 nm; only bonded pairs
  are excluded from nonbonded; no angles or dihedrals
- Langevin dynamics, dt = 0.01 ps, collision rate FRICTION = 0.01 ps^-1.
  Upstream sets the drag coefficient gamma = MW/100 (HOOMD set_gamma),
  i.e. a collision rate gamma/MW = 0.01 ps^-1 for every residue; the
  CALVADOS package's OpenMM default (friction_coeff) is 0.01 as well.
  Studies cached so far ran at FRICTION_CACHED = 1.16 ps^-1 (MW/100 read
  as a rate); see compare_friction_campaigns.

Residue numbers come from the checksummed upstream residues.csv via
build_pilot.py, which also keeps the GROMACS-format artifacts
(pilot/PARAMETERS.md, pilot/tables/) on disk for eyeballing - modern
GROMACS cannot run per-pair user tables (no group cutoff scheme), so
OpenMM evaluates the same pair functions directly as algebra.

Outputs, in examples/calvados_gmx/pilot_openmm/:

- hst5.pdb           input coordinates (inspectable)
- trajectory.dcd     1100 frames, every 1 ps, after 10k equilibration steps
- rg.txt             mass-weighted Rg per frame (nm)
- energy_check.txt   the two-bead H-Y single-point verification against
                     the same pair math as the GROMACS tables

Run from the repository root:

    python -m examples.calvados_gmx.run_pilot_openmm
"""

from pathlib import Path

import numpy as np

import openmm as mm
import openmm.app as app
import openmm.unit as unit

from examples.calvados_gmx.build_pilot import (
    BOND_K,
    BOND_R0,
    DT,
    EPS_HI,
    IONIC,
    R_AH,
    R_YK,
    TEMP,
    debye,
    extended_residues,
    pair_table,
    residues_table,
    type_sequence,
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "pilot_openmm"
LAMBDA_COLUMN = "lambdas"     # 'lambdas' = optimized; 'lambdas_0' = M1 start
SEED = 2026
EQUIL_STEPS = 10_000          # upstream: 10000 steps at dt 0.005
PROD_STEPS = 110_000          # 1.1 ns pilot at dt 0.01
REPORT_INTERVAL = 100         # steps between trajectory frames
FRICTION = 0.01               # ps^-1, upstream collision rate (gamma/MW)
FRICTION_CACHED = 1.16        # ps^-1, rate of the outputs cached so far
                              # (campaign, campaign2, shotgun, longrun,
                              # pilot_a1): kept there so each cache stays
                              # one consistent dataset

# The whole pair physics, evaluated by OpenMM per pair. OpenMM takes one
# flat expression, so the algebra is assembled from named pieces: sigma_ij
# and lambda_ij are arithmetic means; the Ashbaugh-Hatch and Yukawa parts
# are each energy-shifted at their own cutoffs (upstream mode='shift').
_SIGMA = "0.5*(sigma1 + sigma2)"
_LAMBDA = "0.5*(lambda1 + lambda2)"
_QQ = "q1*q2*lb*rt"
_SR6 = f"({_SIGMA}/r)^6"
_LJ = f"(4*epsilon*({_SR6}*{_SR6} - {_SR6}))"
_WCA = f"({_LJ} + epsilon*(1 - {_LAMBDA}))"
_ATT = f"({_LAMBDA}*{_LJ})"
_AH = f"select(step(1.1224620483093721*{_SIGMA} - r), {_WCA}, {_ATT})"
_AH_SHIFT = f"(4*epsilon*{_LAMBDA}*(({_SIGMA}/rc_ah)^12 - ({_SIGMA}/rc_ah)^6))"
_YK = f"({_QQ}*exp(-kappa*r)/r)"
_YK_SHIFT = f"({_QQ}*exp(-kappa*rc_yk)/rc_yk)"
PAIR_ENERGY = (
    f"step(rc_ah - r)*({_AH} - {_AH_SHIFT})"
    f" + step(rc_yk - r)*({_YK} - {_YK_SHIFT})"
)


def particle_records():
    """One record per bead: (type, sigma, lambda, charge, mass)."""
    residues = residues_table(LAMBDA_COLUMN)
    sequence, types, terminal = type_sequence(residues)
    records_ext = extended_residues(residues, terminal)
    records = []
    for one in types:
        row = records_ext.loc[one]
        records.append(
            {
                "type": one,
                "sigma": float(row.sigmas),
                "lambda": float(row.lambdas),
                "q": float(row.q),
                "mass": float(row.MW),
            }
        )
    return records, sequence


def add_pair_force(system, records, kappa, lB, RT):
    """The Ashbaugh-Hatch + Yukawa pair force with per-particle parameters."""
    force = mm.CustomNonbondedForce(PAIR_ENERGY)
    for name in ("sigma", "lambda", "q"):
        force.addPerParticleParameter(name)
    force.addGlobalParameter("epsilon", EPS_HI)
    force.addGlobalParameter("lb", lB)
    force.addGlobalParameter("rt", RT)
    force.addGlobalParameter("kappa", kappa)
    force.addGlobalParameter("rc_ah", R_AH)
    force.addGlobalParameter("rc_yk", R_YK)
    force.setCutoffDistance(R_YK * unit.nanometer)
    force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    for record in records:
        force.addParticle(
            [record["sigma"], record["lambda"], record["q"]]
        )
    for i in range(len(records) - 1):
        force.addExclusion(i, i + 1)
    system.addForce(force)
    return force


def hst5_system(records):
    """24-bead system in the upstream box, bonds only, no PBC pressure."""
    n = len(records)
    box = n * 0.38 + 1
    system = mm.System()
    for record in records:
        system.addParticle(record["mass"])
    bonds = mm.HarmonicBondForce()
    for i in range(n - 1):
        bonds.addBond(i, i + 1, BOND_R0, BOND_K)
    system.addForce(bonds)
    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(box, 0, 0), mm.Vec3(0, box, 0), mm.Vec3(0, 0, box)
    )
    return system, box


def input_positions(records):
    """Beads on a line along z, exactly as upstream make_snapshot."""
    n = len(records)
    positions = [
        mm.Vec3(0, 0, (i - n / 2) * BOND_R0) for i in range(n)
    ]
    return positions


def single_point_check(energy_text):
    """Two H/Y beads at 0.6 nm: OpenMM energy vs the GROMACS-table math."""
    residues = residues_table(LAMBDA_COLUMN)
    kappa, lB, RT = debye(TEMP, IONIC)
    system = mm.System()
    for one in ("H", "Y"):
        system.addParticle(float(residues.loc[one, "MW"]))
    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(10, 0, 0), mm.Vec3(0, 10, 0), mm.Vec3(0, 0, 10)
    )
    records = [
        {
            "sigma": float(residues.loc["H", "sigmas"]),
            "lambda": float(residues.loc["H", "lambdas"]),
            "q": float(residues.loc["H", "q"]),
        },
        {
            "sigma": float(residues.loc["Y", "sigmas"]),
            "lambda": float(residues.loc["Y", "lambdas"]),
            "q": float(residues.loc["Y", "q"]),
        },
    ]
    force = mm.CustomNonbondedForce(PAIR_ENERGY)
    for name in ("sigma", "lambda", "q"):
        force.addPerParticleParameter(name)
    force.addGlobalParameter("epsilon", EPS_HI)
    force.addGlobalParameter("lb", lB)
    force.addGlobalParameter("rt", RT)
    force.addGlobalParameter("kappa", kappa)
    force.addGlobalParameter("rc_ah", R_AH)
    force.addGlobalParameter("rc_yk", R_YK)
    force.setCutoffDistance(R_YK * unit.nanometer)
    force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffNonPeriodic)
    for record in records:
        force.addParticle([record[k] for k in ("sigma", "lambda", "q")])
    system.addForce(force)
    integrator = mm.LangevinMiddleIntegrator(TEMP * unit.kelvin, 1.0, DT)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("CPU"))
    context.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(0.6, 0, 0)])
    openmm_energy = context.getState(getEnergy=True).getPotentialEnergy()
    _, values, _, grid = pair_table(residues, "H", "Y", kappa, lB, RT)
    expected = values[int(round(0.6 / 0.002))]
    text = (
        f"two-bead check (H-Y at 0.6 nm, no bonds)\n"
        f"  openmm : {openmm_energy.value_in_unit(unit.kilojoule_per_mole):.8f}"
        " kJ/mol\n"
        f"  tables : {expected:.8f} kJ/mol (build_pilot.py pair math)\n"
    )
    assert abs(
        openmm_energy.value_in_unit(unit.kilojoule_per_mole) - expected
    ) < 1e-4, "pair energy implementations disagree"
    return text + "  agreement within 1e-4 kJ/mol\n"


def build_topology(records, sequence):
    """An OpenMM topology matching the beads, for trajectory reporting."""
    residues = residues_table(LAMBDA_COLUMN)
    three_of = residues["three"].to_dict()
    topology = app.Topology()
    chain = topology.addChain("A")
    for i, record in enumerate(records):
        one = record["type"]
        base = one if one not in ("X", "Z") else ("D" if one == "X" else "Y")
        residue = topology.addResidue(
            f"{three_of[base]}{i + 1}", chain
        )
        element = app.element.carbon
        topology.addAtom(one, element, residue)
    return topology


def write_pdb(path, records, positions, box):
    """A minimal inspectable PDB of the input coordinates."""
    residues = residues_table(LAMBDA_COLUMN)
    three_of = residues["three"].to_dict()
    lines = ["REMARK   CALVADOS2 pilot input: Hst5 beads on a line"]
    for i, (record, position) in enumerate(zip(records, positions), start=1):
        one = record["type"]
        base = one if one not in ("X", "Z") else (
            "D" if one == "X" else "Y"
        )
        lines.append(
            f"ATOM  {i:5d} {one:>4s} {three_of[base]:>3s} A{i:4d}    "
            f"{position[0]:8.3f}{position[1]:8.3f}{position[2]:8.3f}"
            f"  1.00  0.00          {base:>2s}"
        )
    lines.append(f"CRYST1{box:9.3f}{box:9.3f}{box:9.3f} 90.00 90.00 90.00")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def radius_of_gyration(pdb, dcd, masses):
    """Mass-weighted Rg per frame, as upstream analyse.py calcRg."""
    import MDAnalysis as mda

    universe = mda.Universe(str(pdb), str(dcd))
    values = []
    for _ in universe.trajectory:
        # MDAnalysis positions are in Angstrom; convert to nm.
        positions = universe.atoms.positions / 10.0
        com = np.average(positions, axis=0, weights=masses)
        values.append(
            float(np.sqrt(np.average(((positions - com) ** 2).sum(1), weights=masses)))
        )
    return np.array(values)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records, sequence = particle_records()
    kappa, lB, RT = debye(TEMP, IONIC)

    check_text = single_point_check(None)
    (OUT / "energy_check.txt").write_text(check_text)
    print(check_text, end="")

    system, box = hst5_system(records)
    add_pair_force(system, records, kappa, lB, RT)
    positions = input_positions(records)
    write_pdb(OUT / "hst5.pdb", records, positions, box)

    integrator = mm.LangevinMiddleIntegrator(
        TEMP * unit.kelvin, FRICTION, DT * 0.5
    )
    integrator.setRandomNumberSeed(SEED)
    topology = build_topology(records, sequence)
    simulation = app.Simulation(
        topology,
        system,
        integrator,
        mm.Platform.getPlatformByName("CPU"),
    )
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(
        TEMP * unit.kelvin, SEED
    )

    print(f"equilibration: {EQUIL_STEPS} steps at dt {DT * 0.5} ps", flush=True)
    simulation.step(EQUIL_STEPS)
    integrator.setStepSize(DT)
    simulation.reporters.append(
        app.DCDReporter(str(OUT / "trajectory.dcd"), REPORT_INTERVAL)
    )
    print(f"production: {PROD_STEPS} steps at dt {DT} ps", flush=True)
    for block in range(0, PROD_STEPS, 5_000):
        chunk = min(5_000, PROD_STEPS - block)
        simulation.step(chunk)
        energy = simulation.context.getState(
            getEnergy=True
        ).getPotentialEnergy()
        print(
            f"  step {block + chunk}/{PROD_STEPS}, "
            f"potential {energy.value_in_unit(unit.kilojoule_per_mole):8.2f}"
            " kJ/mol",
            flush=True,
        )
    masses = np.array([record["mass"] for record in records])
    rg = radius_of_gyration(OUT / "hst5.pdb", OUT / "trajectory.dcd", masses)
    np.savetxt(OUT / "rg.txt", rg, fmt="%.4f")
    tail = rg[len(rg) // 2 :]
    print(
        f"Rg (mass-weighted): mean over second half {tail.mean():.3f} nm; "
        f"experimental 1.38 +- 0.05 nm (Tesei et al., 24-residue pilot - "
        "1.1 ns is far short of the upstream 18 us)"
    )
    print(f"outputs in {OUT}")


if __name__ == "__main__":
    main()
