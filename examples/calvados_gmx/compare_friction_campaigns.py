"""Campaign2 at two collision rates: the same 500 points at 1.16 and 0.01 /ps.

``audit_friction`` measured the collision rate's effect at one point (the
CALVADOS2 optimum). This compares the whole campaign2 design (500
bounded Sobol points, seed 2026, 2 + 50 ns, Rg every 10 ps) with the same
design run at the upstream rate 0.01 /ps (``campaign2_rate0.01``, from
``run_campaign2 --friction 0.01 --out ...campaign2_rate0.01``), point by point:

- equilibrium: the paired difference of the 500 means against the noise of
  both; friction must not move Rg anywhere in the box, and a systematic
  dependence of the difference on the lambdas would mean it does (the SEMs
  are themselves estimated from 10 blocks, so the chi2 is also given with
  each z treated as Welch-Satterthwaite t);
- noise: the block-averaged SEM of every 50 ns mean, the integrated
  autocorrelation time of Rg, and the share of the variance among the 500
  means that is noise (one minus the best R2 any surrogate can reach);
- cost at equal noise: the production length at 0.01 /ps that matches
  campaign2's SEM, checked directly on the first nanoseconds of each run.

Outputs, in examples/calvados_gmx/campaign2_rate0.01/friction_comparison/:

- summary.json             the numbers above
- per_point.csv            means, SEMs and tau per point at both rates
- FRICTION_CAMPAIGN.md     the table and what it means
- friction_campaign.png    paired means, z of the differences, SEMs

Run from the repository root:

    python -m examples.calvados_gmx.compare_friction_campaigns
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402

from examples.calvados_gmx.audit_friction import (  # noqa: E402
    autocorrelation,
    integrated_time,
)
from examples.calvados_gmx.fit_campaign2_interim import (  # noqa: E402
    stickiness_features,
)
from examples.calvados_gmx.run_campaign2 import (  # noqa: E402
    HST5,
    N_SAMPLES,
    sobol_points,
)

HERE = Path(__file__).resolve().parent
CAMPAIGNS = {1.16: HERE / "campaign2", 0.01: HERE / "campaign2_rate0.01"}
OUT = HERE / "campaign2_rate0.01" / "friction_comparison"
FRAME_PS = 10                 # both campaigns keep Rg every 10 ps
PROD_PS = 50_000
BLOCK_PS = 5_000              # past the plateau of the block SEM at both rates
MAX_LAG_PS = 2_000
SEED = 2026


def block_sem(x, block_frames):
    """SEM of the mean of ``x`` from the scatter of block means."""
    n_blocks = len(x) // block_frames
    means = x[: n_blocks * block_frames].reshape(n_blocks, block_frames).mean(1)
    return float(means.std(ddof=1) / np.sqrt(n_blocks))


def point_statistics(series):
    """Mean, block SEM, tau_int (ps) and frame SD of one Rg series."""
    block = int(BLOCK_PS / FRAME_PS)
    acf = autocorrelation(series, int(MAX_LAG_PS / FRAME_PS))
    return {
        "mean": float(series.mean()),
        "sem": block_sem(series, block),
        "tau_ps": float(integrated_time(acf) * FRAME_PS),
        "frame_sd": float(series.std()),
    }


def load(rate):
    """Rg series of every design point at one rate, in Sobol order."""
    folder = CAMPAIGNS[rate]
    return [np.loadtxt(folder / f"hst_{i:03d}" / "rg.txt") for i in range(N_SAMPLES)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    series = {rate: load(rate) for rate in CAMPAIGNS}
    table = {rate: [point_statistics(x) for x in series[rate]] for rate in CAMPAIGNS}
    means = {r: np.array([p["mean"] for p in table[r]]) for r in CAMPAIGNS}
    sems = {r: np.array([p["sem"] for p in table[r]]) for r in CAMPAIGNS}
    taus = {r: np.array([p["tau_ps"] for p in table[r]]) for r in CAMPAIGNS}

    # Equilibrium: paired differences against their own noise.
    diff = means[0.01] - means[1.16]
    sem_diff = np.hypot(sems[0.01], sems[1.16])
    z = diff / sem_diff
    chi2 = float(np.sum(z**2))
    # Each SEM is itself estimated from 10 blocks (9 dof), so z is t-like and
    # its chi2 runs high even without any shift: Welch-Satterthwaite dof per
    # point, and each z mapped to the normal score of the same tail.
    block_dof = PROD_PS // BLOCK_PS - 1
    nu = sem_diff**4 / ((sems[0.01] ** 4 + sems[1.16] ** 4) / block_dof)
    z_normal = np.sign(z) * stats.norm.isf(stats.t.sf(np.abs(z), nu))
    chi2_t = float(np.sum(z_normal**2))
    residues = [one for one in sorted(set(HST5)) if one != "E"]
    X = sobol_points(residues, N_SAMPLES)
    features = np.column_stack([np.ones(N_SAMPLES), stickiness_features(X, residues)])
    coef, *_ = np.linalg.lstsq(features, diff, rcond=None)
    fitted = features @ coef
    ss_model = float(np.sum((fitted - diff.mean()) ** 2))
    ss_resid = float(np.sum((diff - fitted) ** 2))
    dof_model, dof_resid = features.shape[1] - 1, N_SAMPLES - features.shape[1]
    f_stat = (ss_model / dof_model) / (ss_resid / dof_resid)

    # Noise share of the variance among the 500 means, and the R2 ceiling.
    noise = {}
    for rate in CAMPAIGNS:
        total = float(means[rate].var(ddof=1))
        noise_var = float(np.mean(sems[rate] ** 2))
        noise[rate] = {
            "median_sem_nm": float(np.median(sems[rate])),
            "rms_sem_nm": float(np.sqrt(noise_var)),
            "median_tau_ps": float(np.median(taus[rate])),
            "sd_of_means_nm": float(np.sqrt(total)),
            "noise_share": noise_var / total,
            "r2_ceiling": 1.0 - noise_var / total,
            "signal_sd_nm": float(np.sqrt(max(total - noise_var, 0.0))),
        }

    # Cost at equal noise: production at 0.01 /ps that matches campaign2's SEM.
    needed_ps = PROD_PS * (sems[0.01] / sems[1.16]) ** 2
    t_check = int(np.ceil(np.median(needed_ps) / 1000.0)) * 1000
    frames = int(t_check / FRAME_PS)
    short_sem = np.array([
        block_sem(x[:frames], max(2, frames // 10)) for x in series[0.01]
    ])
    short_means = np.array([x[:frames].mean() for x in series[0.01]])
    summary = {
        "n_points": N_SAMPLES,
        "equilibrium": {
            "mean_difference_nm": float(diff.mean()),
            "mean_difference_se_nm": float(diff.std(ddof=1) / np.sqrt(N_SAMPLES)),
            "chi2": chi2,
            "dof": N_SAMPLES,
            "p_value": float(stats.chi2.sf(chi2, N_SAMPLES)),
            "fraction_abs_z_over_2": float(np.mean(np.abs(z) > 2)),
            "expected_fraction_abs_z_over_2": float(2 * stats.norm.sf(2)),
            "median_welch_dof": float(np.median(nu)),
            "chi2_t_corrected": chi2_t,
            "p_value_t_corrected": float(stats.chi2.sf(chi2_t, N_SAMPLES)),
            "expected_fraction_abs_z_over_2_t": float(np.mean(2 * stats.t.sf(2, nu))),
            "difference_vs_stickiness_F": float(f_stat),
            "difference_vs_stickiness_p": float(stats.f.sf(f_stat, dof_model, dof_resid)),
            "correlation_of_means": float(np.corrcoef(means[0.01], means[1.16])[0, 1]),
        },
        "noise": {f"{rate:g}": noise[rate] for rate in CAMPAIGNS},
        "cost_at_equal_noise": {
            "median_production_ns_at_0.01": float(np.median(needed_ps) / 1000.0),
            "speedup": float(PROD_PS / np.median(needed_ps)),
            "checked_length_ns": t_check / 1000.0,
            "rms_sem_first_part_nm": float(np.sqrt(np.mean(short_sem**2))),
            "rms_sem_campaign2_nm": noise[1.16]["rms_sem_nm"],
            "sd_of_short_means_nm": float(short_means.std(ddof=1)),
        },
        "block_ps": BLOCK_PS,
        "seed": SEED,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    import pandas as pd

    pd.DataFrame({
        "sample_id": [f"hst_{i:03d}" for i in range(N_SAMPLES)],
        **{f"mean_{r:g}": means[r] for r in CAMPAIGNS},
        **{f"sem_{r:g}": sems[r] for r in CAMPAIGNS},
        **{f"tau_ps_{r:g}": taus[r] for r in CAMPAIGNS},
        "z_difference": z,
    }).to_csv(OUT / "per_point.csv", index=False)

    eq, a, b = summary["equilibrium"], noise[1.16], noise[0.01]
    cost = summary["cost_at_equal_noise"]
    lines = [
        "# Campaign2 at two collision rates (same 500 points)",
        "",
        "| | 1.16 /ps (campaign2) | 0.01 /ps (upstream) |",
        "|---|---|---|",
        f"| tau_int of Rg, median (ps) | {a['median_tau_ps']:.0f} | {b['median_tau_ps']:.0f} |",
        f"| SEM of one 50 ns mean, median (nm) | {a['median_sem_nm']:.4f} | {b['median_sem_nm']:.4f} |",
        f"| SD of the 500 means (nm) | {a['sd_of_means_nm']:.4f} | {b['sd_of_means_nm']:.4f} |",
        f"| noise share of that variance | {a['noise_share']:.0%} | {b['noise_share']:.0%} |",
        f"| best R2 any surrogate can reach | {a['r2_ceiling']:.2f} | {b['r2_ceiling']:.2f} |",
        "",
        f"Equilibrium: mean paired difference {eq['mean_difference_nm']:+.4f} +- "
        f"{eq['mean_difference_se_nm']:.4f} nm; chi2 {eq['chi2']:.0f} on {eq['dof']} "
        f"(p = {eq['p_value']:.2f}); |z| > 2 for {eq['fraction_abs_z_over_2']:.1%} of points "
        f"(expected {eq['expected_fraction_abs_z_over_2']:.1%}); dependence on the "
        f"stickiness groups F = {eq['difference_vs_stickiness_F']:.2f} "
        f"(p = {eq['difference_vs_stickiness_p']:.2f}). Counting the SEMs' own "
        f"uncertainty (10 blocks each, median Welch dof "
        f"{eq['median_welch_dof']:.0f}): chi2 {eq['chi2_t_corrected']:.0f} "
        f"(p = {eq['p_value_t_corrected']:.2f}); |z| > 2 expected for "
        f"{eq['expected_fraction_abs_z_over_2_t']:.1%} of points.",
        "",
        f"Cost at equal noise: {cost['median_production_ns_at_0.01']:.1f} ns at 0.01 /ps "
        f"match campaign2's 50 ns ({cost['speedup']:.1f}x). Check: the first "
        f"{cost['checked_length_ns']:.0f} ns of each new run have RMS SEM "
        f"{cost['rms_sem_first_part_nm']:.4f} nm against {cost['rms_sem_campaign2_nm']:.4f}.",
        "",
    ]
    (OUT / "FRICTION_CAMPAIGN.md").write_text("\n".join(lines))

    slow, fast = "#eb6834", "#2a78d6"   # 1.16 /ps (campaign2), 0.01 /ps (upstream)
    trapped = 25                        # hst_025, the 5.5 sigma point
    figure, (left, middle, right) = plt.subplots(1, 3, figsize=(16, 5.2))
    figure.patch.set_facecolor("white")
    figure.suptitle("Campaign2 rerun at the upstream collision rate: "
                    "same averages, about 3x less noise per training label",
                    fontsize=13.5, fontweight="bold")

    left.errorbar(means[1.16], means[0.01], xerr=sems[1.16], yerr=sems[0.01],
                  fmt="o", markersize=2.5, alpha=0.45, elinewidth=0.5, color="#555555",
                  label="one design point (mean +- 1 SEM)")
    lims = [min(means[1.16].min(), means[0.01].min()) - 0.02,
            max(means[1.16].max(), means[0.01].max()) + 0.02]
    left.plot(lims, lims, "k--", linewidth=1.2, label="same average at both rates")
    left.annotate("hst_025: stuck compact at 1.16 /ps",
                  (means[1.16][trapped], means[0.01][trapped]),
                  xytext=(means[1.16][trapped] + 0.045, means[0.01][trapped] - 0.085),
                  arrowprops={"arrowstyle": "->", "color": "#b42318"}, color="#b42318", fontsize=9)
    left.set_xlabel("mean Rg at 1.16 /ps, campaign2 (nm)")
    left.set_ylabel("mean Rg at 0.01 /ps, rerun (nm)")
    left.set_title("Do the averages change?  No", fontsize=11.5)
    left.legend(loc="upper left", fontsize=8.5, framealpha=0.95)

    dof = float(np.median(nu))
    grid = np.linspace(-6, 6, 300)
    middle.hist(z, bins=np.linspace(-6, 6, 49), density=True, color="#8796ab",
                label="observed, 500 points")
    middle.plot(grid, stats.t.pdf(grid, dof), color="#11703d", linewidth=2,
                label=f"expected if friction changes nothing\n(t, {dof:.0f} dof)")
    middle.annotate("hst_025", (z[trapped], 0.01), xytext=(z[trapped] - 1.6, 0.12),
                    arrowprops={"arrowstyle": "->", "color": "#b42318"}, color="#b42318", fontsize=9)
    middle.set_xlabel("difference of the two averages / its SEM")
    middle.set_ylabel("density")
    middle.set_title("Any difference beyond the noise?  Only hst_025\n"
                     f"(chi2 {chi2_t:.0f} on {N_SAMPLES}, p = {stats.chi2.sf(chi2_t, N_SAMPLES):.2f})",
                     fontsize=11.5)
    middle.legend(loc="upper left", fontsize=8.5, framealpha=0.95)

    bins = np.linspace(0, max(sems[1.16].max(), sems[0.01].max()), 40)
    for rate, colour, name in ((1.16, slow, "1.16 /ps (campaign2)"), (0.01, fast, "0.01 /ps (upstream)")):
        median = float(np.median(sems[rate]))
        right.hist(sems[rate], bins=bins, alpha=0.65, color=colour,
                   label=f"{name}: median {median:.4f} nm")
        right.axvline(median, color=colour, linewidth=1.5, linestyle="--")
    right.annotate("less noise is better", xy=(0.004, 0.5), xycoords=("data", "axes fraction"),
                   xytext=(0.017, 0.5), textcoords=("data", "axes fraction"),
                   arrowprops={"arrowstyle": "->", "color": "#11703d", "lw": 1.5},
                   color="#11703d", fontsize=9.5, va="center")
    right.set_xlabel("SEM of one 50 ns training label (nm)")
    right.set_ylabel("number of design points")
    right.set_title("Noise per training label: about 3x lower", fontsize=11.5)
    right.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(OUT / "friction_campaign.png", dpi=160)
    print("\n".join(lines))
    print(f"outputs in {OUT}")


if __name__ == "__main__":
    main()
