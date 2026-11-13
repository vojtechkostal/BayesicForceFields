"""Friction audit: does the Langevin collision rate change Hst5's Rg, or
only how fast Rg is sampled?

The cached studies (campaign2 and the others) ran with a collision rate
of 1.16 ps^-1 (rp.FRICTION_CACHED, marked ADAPT before), read from upstream simulate.py's
``integrator.set_gamma(a, residues.loc[a].MW/100)``. In HOOMD v2 that
gamma is a drag coefficient: the Langevin force is -gamma*v, so gamma
has units of mass/time and the collision rate is gamma/m =
(MW/100)/MW = 0.01 ps^-1 for every residue. The CALVADOS package's own
OpenMM code uses the same number (``friction_coeff = 0.01``, in
``calvados/inputmodels.py``). OpenMM's ``LangevinMiddleIntegrator``
takes the collision rate directly, so the faithful translation is 0.01,
not 1.16.

Friction does not change the equilibrium ensemble, so the campaign's
Rg values stand. It changes how fast Rg decorrelates, and with it the
noise of every 50 ns mean the surrogate is trained on. This script
measures both, at the CALVADOS2 optimized lambdas (the pilot point):

- three collision rates: 1.16 (campaign2), 0.1 and 0.01 ps^-1 (upstream)
- four seeds per rate, the campaign2 protocol: 2 ns equilibration at
  dt 0.005 ps from the extended chain, then 50 ns at dt 0.01 ps; Rg every
  1 ps (campaign2 keeps 10 ps) so short correlation times are resolved
- per rate: pooled mean Rg (the equilibrium check), the integrated
  autocorrelation time of Rg, the scatter of 10 ns block means (the
  campaign2 split-half audit put SEM(10 ns) = 0.038 nm), the SEM of one
  50 ns mean, and the production length that reaches the campaign2
  target SEM of 0.017 nm

Every run caches its Rg series and is skipped when present, so the audit
resumes after interruption. Runs go to a process pool, one CPU thread
each (Hst5 is 24 beads: about 13,000 ns/day on one core).

Outputs, in examples/calvados_gmx/audit_friction/:

- gamma_<rate>/seed_<seed>/rg.txt   mass-weighted Rg per ps (nm)
- summary.json                      the numbers above, per rate
- friction_rg.png                   Rg distributions and autocorrelation
- FRICTION.md                       the table and what it means

Run from the repository root:

    python -m examples.calvados_gmx.audit_friction
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from examples.calvados_gmx import run_pilot_openmm as rp  # noqa: E402
from examples.calvados_gmx.build_pilot import debye  # noqa: E402

HERE = rp.HERE
OUT = HERE / "audit_friction"
RATES = (1.16, 0.1, 0.01)     # ps^-1; 1.16 = campaign2, 0.01 = upstream
CAMPAIGN_RATE = 1.16
SEEDS = (2026, 2027, 2028, 2029)
EQUIL_PS = 2_000              # as run_campaign2, at dt 0.005
PROD_PS = 50_000              # as run_campaign2, at dt 0.01
FRAME_PS = 1                  # campaign2 reports every 10 ps
BLOCK_PS = 10_000             # the campaign2 split-half audit used 10 ns
TARGET_SEM_NM = 0.017         # campaign2's design target for one 50 ns mean
MAX_LAG_PS = 2_000            # autocorrelation is evaluated up to this lag
N_WORKERS = 8


def run_dir(rate, seed):
    return OUT / f"gamma_{rate:g}" / f"seed_{seed}"


def radius_of_gyration(positions, masses):
    """Mass-weighted Rg of one frame, the formula of rp.radius_of_gyration."""
    com = np.average(positions, axis=0, weights=masses)
    return float(
        np.sqrt(np.average(((positions - com) ** 2).sum(1), weights=masses))
    )


def run_one(job):
    """One 2 + 50 ns trajectory at one collision rate; caches rg.txt."""
    rate, seed = job
    out_dir = run_dir(rate, seed)
    rg_file = out_dir / "rg.txt"
    if rg_file.is_file():
        return str(rg_file), True

    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit

    rp.LAMBDA_COLUMN = "lambdas"
    records, sequence = rp.particle_records()
    kappa, lB, RT = debye(rp.TEMP, rp.IONIC)
    system, _ = rp.hst5_system(records)
    rp.add_pair_force(system, records, kappa, lB, RT)
    positions = rp.input_positions(records)
    topology = rp.build_topology(records, sequence)
    masses = np.array([record["mass"] for record in records])

    integrator = mm.LangevinMiddleIntegrator(
        rp.TEMP * unit.kelvin, rate, rp.DT * 0.5
    )
    integrator.setRandomNumberSeed(seed)
    simulation = app.Simulation(
        topology, system, integrator,
        mm.Platform.getPlatformByName("CPU"), {"Threads": "1"},
    )
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(rp.TEMP * unit.kelvin, seed)
    simulation.step(int(EQUIL_PS / (rp.DT * 0.5)))
    integrator.setStepSize(rp.DT)

    frame_steps = int(round(FRAME_PS / rp.DT))
    n_frames = int(PROD_PS / FRAME_PS)
    rg = np.empty(n_frames)
    for frame in range(n_frames):
        simulation.step(frame_steps)
        state = simulation.context.getState(getPositions=True)
        xyz = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        rg[frame] = radius_of_gyration(np.asarray(xyz), masses)

    out_dir.mkdir(parents=True, exist_ok=True)
    temporary = out_dir / "rg.txt.part"
    np.savetxt(temporary, rg, fmt="%.4f")
    temporary.replace(rg_file)
    return str(rg_file), False


def autocorrelation(x, max_lag):
    """Normalized autocorrelation of one series up to max_lag frames."""
    centered = x - x.mean()
    n = len(centered)
    size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(centered, size)
    acf = np.fft.irfft(spectrum * np.conj(spectrum), size)[:n]
    acf /= acf[0]
    return acf[: max_lag + 1]


def integrated_time(acf, window_c=5.0):
    """tau_int = 1/2 + sum rho(t), Sokal's window M >= c*tau_int (frames).

    With this convention a mean of N frames has variance
    sigma^2 * 2 * tau_int / N.
    """
    tau = 0.5
    for lag in range(1, len(acf)):
        tau += acf[lag]
        if lag >= window_c * tau:
            return tau
    return tau


def rate_summary(rate):
    """Pooled statistics over the seeds of one collision rate."""
    series = [np.loadtxt(run_dir(rate, seed) / "rg.txt") for seed in SEEDS]
    max_lag = int(MAX_LAG_PS / FRAME_PS)
    acfs = [autocorrelation(x, max_lag) for x in series]
    taus_ps = [integrated_time(acf) * FRAME_PS for acf in acfs]
    frame_sd = float(np.std(np.concatenate(series)))
    run_means = np.array([x.mean() for x in series])
    block = int(BLOCK_PS / FRAME_PS)
    block_means = np.concatenate(
        [x[: len(x) // block * block].reshape(-1, block).mean(1) for x in series]
    )
    tau_ps = float(np.mean(taus_ps))
    sem_50 = frame_sd * np.sqrt(2 * tau_ps / PROD_PS)
    return {
        "collision_rate_per_ps": rate,
        "seeds": list(SEEDS),
        "mean_rg_nm": float(run_means.mean()),
        "mean_rg_sem_nm": float(sem_50 / np.sqrt(len(SEEDS))),
        "run_means_nm": run_means.round(4).tolist(),
        "frame_sd_nm": frame_sd,
        "tau_int_ps": tau_ps,
        "tau_int_ps_per_seed": [round(t, 1) for t in taus_ps],
        "sd_of_10ns_block_means_nm": float(np.std(block_means, ddof=1)),
        "n_10ns_blocks": int(len(block_means)),
        "sem_50ns_mean_nm": float(sem_50),
        "sd_of_seed_means_nm": float(np.std(run_means, ddof=1)),
        "production_ns_for_target_sem": float(
            PROD_PS / 1000 * (sem_50 / TARGET_SEM_NM) ** 2
        ),
        "mean_acf": np.mean(acfs, axis=0).tolist(),
        "histogram": np.histogram(
            np.concatenate(series), bins=60, range=(0.8, 2.6), density=True
        )[0].tolist(),
    }


def write_report(summaries):
    """FRICTION.md: the per-rate table and the reading of it."""
    campaign = next(s for s in summaries if s["collision_rate_per_ps"] == CAMPAIGN_RATE)
    lines = [
        "# Friction audit: Hst5 at the CALVADOS2 optimized lambdas",
        "",
        f"Four seeds per collision rate; each {EQUIL_PS // 1000} ns equilibration "
        f"+ {PROD_PS // 1000} ns production (the campaign2 protocol); Rg every "
        f"{FRAME_PS} ps. Generated by `audit_friction.py`.",
        "",
        "| collision rate (1/ps) | mean Rg (nm) | tau_int (ps) | sd of 10 ns means (nm) "
        "| SEM of a 50 ns mean (nm) | ns for SEM 0.017 nm | speed-up vs 1.16 |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        speedup = campaign["production_ns_for_target_sem"] / s["production_ns_for_target_sem"]
        lines.append(
            f"| {s['collision_rate_per_ps']:g} | {s['mean_rg_nm']:.4f} +- "
            f"{s['mean_rg_sem_nm']:.4f} | {s['tau_int_ps']:.0f} "
            f"({min(s['tau_int_ps_per_seed']):.0f}-{max(s['tau_int_ps_per_seed']):.0f}) | "
            f"{s['sd_of_10ns_block_means_nm']:.4f} | {s['sem_50ns_mean_nm']:.4f} | "
            f"{s['production_ns_for_target_sem']:.1f} | {speedup:.1f}x |"
        )
    lines += [
        "",
        "Reading: the mean Rg is the equilibrium check (friction must not move it);",
        "tau_int and the SEMs are the sampling cost. The SEM of a 50 ns mean is",
        "frame_sd * sqrt(2 tau_int / 50 ns); 'ns for SEM 0.017 nm' scales it by 1/sqrt(T).",
        "",
    ]
    (OUT / "FRICTION.md").write_text("\n".join(lines))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(rate, seed) for rate in RATES for seed in SEEDS]
    pending = [job for job in jobs if not (run_dir(*job) / "rg.txt").is_file()]
    print(f"{len(jobs)} runs, {len(pending)} to simulate on {N_WORKERS} workers",
          flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            for path, cached in pool.map(run_one, pending):
                print(f"  {'cached' if cached else 'done'}: {path}", flush=True)

    summaries = [rate_summary(rate) for rate in RATES]
    slim = [
        {k: v for k, v in s.items() if k not in ("mean_acf", "histogram")}
        for s in summaries
    ]
    (OUT / "summary.json").write_text(json.dumps(slim, indent=2) + "\n")
    write_report(summaries)

    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5))
    edges = np.linspace(0.8, 2.6, 61)
    centers = 0.5 * (edges[1:] + edges[:-1])
    for s in summaries:
        label = f"{s['collision_rate_per_ps']:g} / ps"
        left.plot(centers, s["histogram"], label=label)
        lags = np.arange(len(s["mean_acf"])) * FRAME_PS
        right.plot(lags, s["mean_acf"], label=f"{label} (tau {s['tau_int_ps']:.0f} ps)")
    left.set_xlabel("Rg (nm)")
    left.set_ylabel("density")
    left.set_title("Rg distribution: friction must not move it")
    left.legend()
    right.set_xlim(0, 500)
    right.axhline(0, color="0.6", linewidth=0.8)
    right.set_xlabel("lag (ps)")
    right.set_ylabel("autocorrelation of Rg")
    right.set_title("How fast Rg decorrelates")
    right.legend()
    figure.tight_layout()
    figure.savefig(OUT / "friction_rg.png", dpi=160)

    for s in slim:
        print(
            f"rate {s['collision_rate_per_ps']:g}/ps: Rg {s['mean_rg_nm']:.4f} +- "
            f"{s['mean_rg_sem_nm']:.4f} nm, tau_int {s['tau_int_ps']:.0f} ps, "
            f"sd(10 ns means) {s['sd_of_10ns_block_means_nm']:.4f} nm, "
            f"SEM(50 ns) {s['sem_50ns_mean_nm']:.4f} nm, "
            f"{s['production_ns_for_target_sem']:.1f} ns for 0.017 nm"
        )
    print(f"outputs in {OUT}")


if __name__ == "__main__":
    main()
