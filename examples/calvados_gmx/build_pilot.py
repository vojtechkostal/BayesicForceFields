"""Build a GROMACS pilot system: CALVADOS2 Hst5 at the optimized lambdas.

Reproduces the upstream M1_24 simulation (simulate.py, pinned commit) with
GROMACS instead of HOOMD: one bead per residue, Ashbaugh-Hatch attraction
with arithmetic sigma/lambda mixing, exact per-pair potentials in user
tables (including the energy shift at the cutoffs), Yukawa/Debye-Huckel
electrostatics baked into the same tables, and harmonic bonds.

All residue numbers (sigma, charge, mass, lambda) come from the
checksum-verified upstream residues.csv. Numbers a reviewing eye should
check are printed to pilot/PARAMETERS.md.

Writes:

- pilot/        the pilot system: hst5.top, hst5.gro, index.ndx, run.mdp,
                tables/table_<T1>-<T2>.xvg for every residue-type pair,
                and PARAMETERS.md
- validate/     a two-bead single-point system (H and Y at 0.6 nm) whose
                expected energy is written to expected_energy.txt

Then run, from the repository root:

    gmx grompp -f validate/run.mdp -p validate/two.top -c validate/two.gro \
        -n validate/index.ndx -o validate/tpr.tpr
    gmx mdrun -deffnm validate/md -s validate/tpr.tpr
    printf 'Potential\\n0\\n' | gmx energy -f md.edr -xvg no -o validate/epot.xvg

and compare validate/epot.xvg against validate/expected_energy.txt; they
must agree exactly, which verifies the table convention (column meaning
and sign). Then grompp/mdrun the pilot the same way with -n pilot/index.ndx.
"""

import io
import math
from pathlib import Path

import numpy as np
import pandas as pd

from examples.calvados.fetch_data import RAW, verified_bytes

HERE = Path(__file__).resolve().parent
PILOT = HERE / "pilot"
VALIDATE = HERE / "validate"

# --- upstream simulate.py constants (M1_24) -----------------------------
EPS_HI = 4.184 * 0.2          # kJ/mol, lj_eps
R_AH = 2.4                    # nm, Ashbaugh-Hatch cutoff (mode='shift')
R_YK = 4.0                    # nm, Yukawa cutoff (mode='shift')
BOND_K = 8033.0               # kJ/mol/nm^2
BOND_R0 = 0.38                # nm
DT = 0.01                     # ps
TABLE_STEP = 0.002            # nm, user-table grid spacing
TABLE_MAX = 5.0               # nm, table range: must cover rvdw + 1
# Hst5 conditions from upstream analyse.py (initProteinsRgs)
TEMP = 293.0                  # K
PH = 7.5
IONIC = 0.15                  # mol/L
# Bonded beads are excluded from nonbonded pairs (HOOMD reset_exclusions
# ['bond'] == GROMACS nrexcl 2); no angles or dihedrals in CALVADOS.
NREXCL = 2

# ADAPT: the GROMACS convention under test in validate/. If the single
# point energy differs from expected_energy.txt by a factor, change this.
TABLE_COLUMN_SCALE = 1.0


def residues_table(lambda_column="lambdas"):
    """The 20 CALVADOS residues; ``lambda_column`` selects the
    parameterization ('lambdas' = optimized, 'lambdas_0' = M1 start)."""
    table = pd.read_csv(io.BytesIO(verified_bytes(RAW / "residues.csv")))
    table = table.set_index("one", drop=False)
    table["q"] = table["q"].astype(float)
    if lambda_column not in table.columns:
        raise ValueError(f"CALVADOS residues: no column {lambda_column!r}")
    table["lambdas"] = table[lambda_column].astype(float)
    # pH-dependent HIS charge, exactly as upstream genParams
    table.loc["H", "q"] = 1.0 / (1.0 + 10 ** (PH - 6))
    return table


def debye(temperature, ionic):
    """(kappa in 1/nm, lB in nm, RT in kJ/mol) exactly as upstream."""
    RT = 8.3145 * temperature * 1e-3
    fepsw = lambda T: (5321 / T + 233.76 - 0.9297 * T
                       + 0.1417e-2 * T * T - 0.8292e-6 * T ** 3)  # noqa: E731
    epsw = fepsw(temperature)
    lB = 1.6021766 ** 2 / (4 * math.pi * 8.854188 * epsw) * 6.022 * 1000 / RT
    kappa = math.sqrt(8 * math.pi * lB * ionic * 6.022 / 10)
    return kappa, lB, RT


def ashbaugh_hatch(r, lam, sig, eps):
    """Shifted AH pair energy in kJ/mol and its derivative.

    Upstream azplugins.pair.ashbaugh, mode='shift': the whole pair energy
    is shifted by its value at r_cut so the potential reaches zero at R_AH.
    """
    rmin = 2 ** (1 / 6) * sig
    q = (sig / r) ** 6
    lj = 4 * eps * (q * q - q)
    r_energy = np.where(r <= rmin, lj + eps * (1 - lam), lam * lj)
    d_r = (sig / r) ** 7
    d_lj = 4 * eps * (-12 * d_r * (sig / r) ** 6 + 6 * d_r) / sig
    d_energy = np.where(r <= rmin, d_lj, lam * d_lj)
    r_energy = np.where(r < R_AH, r_energy, 0.0)
    d_energy = np.where(r < R_AH, d_energy, 0.0)
    shift = 4 * eps * lam * ((sig / R_AH) ** 12 - (sig / R_AH) ** 6)
    return r_energy - shift, -d_energy


def yukawa(r, qi, qj, kappa, lB, RT):
    """Shifted Yukawa pair energy in kJ/mol and its derivative."""
    prefactor = qi * qj * lB * RT
    energy = np.where(r < R_YK, prefactor * np.exp(-kappa * r) / r, 0.0)
    slope = -prefactor * np.exp(-kappa * r) * (kappa / r + 1 / r ** 2)
    d_energy = np.where(r < R_YK, slope, 0.0)
    at_cut = prefactor * np.exp(-kappa * R_YK) / R_YK
    return energy - at_cut, -d_energy


def pair_table(residues, name_i, name_j, kappa, lB, RT):
    """A GROMACS user table for one residue-type pair.

    The full pair potential (shifted AH + shifted Yukawa) is in the
    dispersion column; the Coulomb and repulsion columns are zero, because
    single global tables cannot express per-pair electrostatics and all
    physics here is per-pair by construction (analysis may rely on this
    file directly: r, V(r), -V'(r) in columns 1, 2, 3).
    """
    ri, rj = residues.loc[name_i], residues.loc[name_j]
    sig = (ri.sigmas + rj.sigmas) / 2
    lam = (ri.lambdas + rj.lambdas) / 2
    r = np.arange(0.0, TABLE_MAX + TABLE_STEP / 2, TABLE_STEP)
    r_safe = np.maximum(r, 1e-6)
    ah, ah_deriv = ashbaugh_hatch(r_safe, lam, sig, EPS_HI)
    yk, yk_deriv = yukawa(r_safe, ri.q, rj.q, kappa, lB, RT)
    value = TABLE_COLUMN_SCALE * (ah + yk)
    deriv = TABLE_COLUMN_SCALE * (ah_deriv + yk_deriv)
    columns = np.column_stack(
        [r, np.zeros_like(r), np.zeros_like(r), value, deriv,
         np.zeros_like(r), np.zeros_like(r)]
    )
    text = "".join(
        " ".join(f"{x:.10e}" for x in row) + "\n" for row in columns
    )
    return text, value, deriv, r


def type_sequence(residues):
    """Hst5 with upstream terminal handling: X = N-term (+1 charge, +2 MW),
    Z = C-term (-1 charge, +16 MW)."""
    sequence = list("DSHAKRHHGYKRKFHEKHHSHRGY")
    types = list(sequence)
    first, last = types[0], types[-1]
    types[0] = "X"
    types[-1] = "Z"
    terminal = {
        "X": dict(base=first, dq=+1.0, dm=+2.0),
        "Z": dict(base=last, dq=-1.0, dm=+16.0),
    }
    return sequence, types, terminal


def extended_residues(residues, terminal):
    """Residue table extended with the terminal pseudo-types X and Z."""
    table = residues.copy()
    for name, spec in terminal.items():
        row = residues.loc[spec["base"]].copy()
        row["one"] = name
        row["q"] = row["q"] + spec["dq"]
        row["MW"] = row["MW"] + spec["dm"]
        table.loc[name] = row
    return table


def gro_lines(types, residues, sequence, box):
    """Fixed-format .gro lines: beads on a line along z, as upstream."""
    n = len(types)
    lines = [f"{n:5d}"]
    for i, (one, seq) in enumerate(zip(types, sequence), start=1):
        three = residues.loc[one, "three"]
        mass_center = (i - n / 2) * BOND_R0
        lines.append(
            f"{i % 100000:5d}{three[:5]:<5s}{one:>5s}{i % 100000:5d}"
            f"{0.0:8.3f}{0.0:8.3f}{mass_center:8.3f}"
        )
    lines.append(f"{box:10.5f}{box:10.5f}{box:10.5f}")
    return "\n".join(lines) + "\n"


def top_text(types, residues_ext, sequence):
    """Topology: dummy LJ parameters (tables carry the physics), zero
    charges (electrostatics is inside the tables, else it would be double
    counted), upstream masses, CALVADOS harmonic bonds, nrexcl 2."""
    lines = [
        "; CALVADOS2 Hst5 pilot topology (M1_24 numbers)",
        "; LJ numbers below are UNUSED: vdwtype=user tables carry the",
        "; Ashbaugh-Hatch + Yukawa physics. Charges are ZERO because",
        "; electrostatics is inside the tables. See PARAMETERS.md.",
        "",
        "[ defaults ]",
        "; nbfunc  comb-rule  gen-pairs  fudgeLJ",
        "1         1          no         1",
        "",
        "[ atomtypes ]",
        "; name  at.num  mass    charge  ptype  sigma(nm)  epsilon(kJ/mol)",
    ]
    for one in sorted(set(types)):
        row = residues_ext.loc[one]
        lines.append(
            f"{one:<6s}  0  {row.MW:10.4f}  0.0000  A  "
            f"{row.sigmas:9.5f}  {EPS_HI:9.5f}"
        )
    lines += ["", "[ moleculetype ]", "; name  nrexcl", f"Hst5  {NREXCL}", ""]
    lines += [
        "[ atoms ]",
        "; nr  type  resnr  residue  atom  cgnr  charge  mass",
    ]
    three_of = residues_ext["three"].to_dict()
    for i, (one, seq) in enumerate(zip(types, sequence), start=1):
        row = residues_ext.loc[one]
        lines.append(
            f"{i:4d}  {one:<2s}  {i:4d}  {three_of[seq]:<6s}  {one}{i:<4d}  "
            f"{i:4d}  0.0000  {row.MW:9.4f}"
        )
    lines += ["", "[ bonds ]", "; ai  aj  func  b0(nm)  kb(kJ/mol/nm^2)"]
    for i in range(1, len(types)):
        lines.append(f"{i:4d}  {i + 1:4d}  1  {BOND_R0:.3f}  {BOND_K:.1f}")
    lines += [
        "",
        "[ system ]",
        "CALVADOS2 Hst5 pilot",
        "",
        "[ molecules ]",
        "Hst5  1",
    ]
    return "\n".join(lines) + "\n"


def ndx_text(types):
    """Index file: one energy group per residue type."""
    blocks = []
    for one in sorted(set(types)):
        members = [str(i) for i, t in enumerate(types, start=1) if t == one]
        chunk = [f"[ {one} ]"]
        for start in range(0, len(members), 15):
            chunk.append(" ".join(members[start : start + 15]))
        blocks.append("\n".join(chunk))
    return "\n".join(blocks) + "\n"


def mdp_text(groups, nsteps):
    """Run parameters mirroring upstream: Langevin-like sd integrator,
    dt 0.01, user tables with rvdw 4.0 (Yukawa tail reaches 4.0)."""
    entries = []
    for i, gi in enumerate(groups):
        for j in range(i, len(groups)):
            gj = groups[j]
            entries.append(f"{gi} {gj} tables/table_{gi}-{gj}.xvg")
    lines = [
        "integrator               = sd",
        f"dt                       = {DT}",
        f"nsteps                   = {nsteps}",
        "cutoff-scheme            = Verlet",
        "nstlist                  = 10",
        "rlist                    = 4.0",
        "coulombtype              = Cut-off",
        "rcoulomb                 = 4.0",
        "vdwtype                  = User",
        "rvdw                     = 4.0",
        "; plain table cutoffs: shifts are baked into the tables",
        "vdw-modifier             = None",
        "constraints              = none",
        "continuation             = no",
        "gen-vel                  = yes",
        f"gen-temp                 = {TEMP}",
        f"ref-t                    = {TEMP}",
        "tau-t                    = 1.0",
        "nstcalcenergy            = 100",
        "nstenergy                = 500",
        "nstlog                   = 5000",
        "nstxout-compressed       = 100",
        f"energygrps               = {' '.join(groups)}",
        "",
        "energygrp-table          = " + " ".join(entries),
        "",
    ]
    return "\n".join(lines) + "\n"


def parameters_md(residues_ext, types, kappa, lB, RT):
    """Human-readable summary of every number, with provenance."""
    k_dh, lB_dh, RT_dh = debye(TEMP, IONIC)
    epsw = (5321 / TEMP + 233.76 - 0.9297 * TEMP
            + 0.1417e-2 * TEMP ** 2 - 0.8292e-6 * TEMP ** 3)
    lines = [
        "# CALVADOS2 pilot parameters (Hst5 @ M1_24 optimized lambdas)",
        "",
        "Sources: upstream `simulate.py`, `analyse.py`, `residues.csv` ",
        "(pinned commit, checksum-verified by fetch_data.py).",
        "Review every ADAPT-marked number.",
        "",
        f"- temperature: {TEMP} K (Hst5 in analyse.py)",
        f"- pH: {PH} (HIS charge = 1/(1+10^(pH-6)) = "
        f"{residues_ext.loc['H', 'q']:.4f} e)",
        f"- ionic strength: {IONIC} M",
        f"- water dielectric series: epsw = {epsw:.2f}",
        f"- Bjerrum length lB = {lB:.4f} nm, RT = {RT:.4f} kJ/mol",
        f"- Debye kappa = {kappa:.4f} 1/nm (Debye length {1/kappa:.3f} nm)",
        f"- eps_HI = {EPS_HI} kJ/mol, AH cutoff = {R_AH} nm (energy-shifted)",
        f"- Yukawa cutoff = {R_YK} nm (energy-shifted)",
        f"- bonds: k = {BOND_K} kJ/mol/nm^2, r0 = {BOND_R0} nm",
        f"- dt = {DT} ps, Langevin gamma = MW/100 (sd tau-t 1.0 ADAPT)",
        f"- exclusions: bonded pairs only (nrexcl {NREXCL}); no angles",
        "",
        "| type | three | sigma (nm) | lambda (M1 opt) | charge (e) | MW (Da) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for one in sorted(set(types)):
        row = residues_ext.loc[one]
        note = ""
        if one in ("X", "Z"):
            note = " (terminal bead, charge and mass corrected)"
        lines.append(
            f"| {one} | {row.three} | {row.sigmas:.4f} | {row.lambdas:.4f} | "
            f"{row.q:+.4f} | {row.MW:.2f}{note} |"
        )
    return "\n".join(lines) + "\n"


def build_pilot():
    residues = residues_table()
    sequence, types, terminal = type_sequence(residues)
    residues_ext = extended_residues(residues, terminal)
    kappa, lB, RT = debye(TEMP, IONIC)
    groups = sorted(set(types))
    (PILOT / "tables").mkdir(parents=True, exist_ok=True)
    for i, gi in enumerate(groups):
        for gj in groups[i:]:
            text, _, _, _ = pair_table(residues_ext, gi, gj, kappa, lB, RT)
            (PILOT / "tables" / f"table_{gi}-{gj}.xvg").write_text(text)
    n = len(types)
    box = n * BOND_R0 + 1
    (PILOT / "hst5.gro").write_text(gro_lines(types, residues_ext, sequence, box))
    (PILOT / "hst5.top").write_text(top_text(types, residues_ext, sequence))
    (PILOT / "index.ndx").write_text(ndx_text(types))
    # 10k equilibration steps (upstream: 10000 at dt 0.005) + 1 ns pilot
    (PILOT / "run.mdp").write_text(mdp_text(groups, nsteps=10000 + 100000))
    (PILOT / "PARAMETERS.md").write_text(
        parameters_md(residues_ext, types, kappa, lB, RT)
    )
    print(f"pilot written to {PILOT}")
    print(f"  {len(groups)} residue types -> {len(groups) * (len(groups) + 1) // 2} pair tables")
    print(f"  box {box:.2f} nm, {n} beads, {TEMP} K, I = {IONIC} M")


def build_validate():
    """Two beads (H, Y) at 0.6 nm: hand energy vs GROMACS single point."""
    residues = residues_table()
    kappa, lB, RT = debye(TEMP, IONIC)
    VALIDATE.mkdir(parents=True, exist_ok=True)
    text, value, _, _ = pair_table(residues, "H", "Y", kappa, lB, RT)
    (VALIDATE / "table_H-Y.xvg").write_text(text)
    distance = 0.6
    index = int(round(distance / TABLE_STEP))
    expected = value[index] / TABLE_COLUMN_SCALE
    lines = [
        "; two-bead single-point check of the table convention",
        "",
        "[ defaults ]",
        "1  1  no  1",
        "",
        "[ atomtypes ]",
        "; name  at.num  mass  charge  ptype  sigma  epsilon (unused)",
        f"H   0  {residues.loc['H', 'MW']:.4f}  0.0  A  0.47  0.8368",
        f"Y   0  {residues.loc['Y', 'MW']:.4f}  0.0  A  0.55  0.8368",
        "",
        "[ moleculetype ]",
        "; name  nrexcl",
        "Two  1",
        "",
        "[ atoms ]",
        "; nr  type  resnr  residue  atom  cgnr  charge  mass",
        f"1  H  1  HIS  H1  1  0.0  {residues.loc['H', 'MW']:.4f}",
        f"2  Y  2  TYR  Y2  2  0.0  {residues.loc['Y', 'MW']:.4f}",
        "",
        "[ moleculetype ]",
        "Water  1",
        "",
        "[ atoms ]",
        "; a lone dummy third bead (far away, its table entries are unused:",
        "; at 9 nm there is no interaction within any cutoff)",
        f"1  H  1  DUM  D1  1  0.0  {residues.loc['H', 'MW']:.4f}",
        "",
        "[ system ]",
        "Two-bead table convention check",
        "",
        "[ molecules ]",
        "Two  1",
        "Water  1",
    ]
    (VALIDATE / "two.top").write_text("\n".join(lines) + "\n")
    (VALIDATE / "two.gro").write_text(
        f"3\n"
        f"    1HIS      H    0.000   0.000   0.000\n"
        f"    2TYR      Y    0.600   0.000   0.000\n"
        f"    3DUM      D    9.000   9.000   9.000\n"
        f"  10.00000  10.00000  10.00000\n"
    )
    (VALIDATE / "index.ndx").write_text("[ H ]\n1\n\n[ Y ]\n2\n\n[ DUM ]\n3\n")
    (VALIDATE / "run.mdp").write_text(
        "\n".join(
            [
                "integrator      = sd",
                "dt              = 0.01",
                "nsteps          = 0",
                "cutoff-scheme   = Verlet",
                "nstlist         = 1",
                "rlist           = 4.0",
                "coulombtype     = Cut-off",
                "rcoulomb        = 4.0",
                "vdwtype         = User",
                "rvdw            = 4.0",
                "vdw-modifier    = None",
                "constraints     = none",
                "gen-vel         = no",
                f"ref-t           = {TEMP}",
                "tau-t           = 1.0",
                "nstenergy       = 1",
                "energygrps      = H Y DUM",
                "",
                "energygrp-table = H H table_H-Y.xvg"
                " H Y table_H-Y.xvg"
                " H DUM table_H-Y.xvg"
                " Y Y table_H-Y.xvg"
                " Y DUM table_H-Y.xvg"
                " DUM DUM table_H-Y.xvg",
                "",
            ]
        )
        + "\n"
    )
    (VALIDATE / "expected_energy.txt").write_text(
        f"expected pair energy H-Y at {distance} nm: {expected:.8f} kJ/mol\n"
        f"(shifted AH + shifted Yukawa; table convention check)\n"
    )
    print(f"validate written to {VALIDATE}")
    print(f"expected H-Y pair energy at 0.6 nm: {expected:.8f} kJ/mol")


if __name__ == "__main__":
    build_validate()
    build_pilot()
