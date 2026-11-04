"""Prepare bounded joint stickiness priors for later posterior sampling."""

import io
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp, ndtr
from scipy.stats import beta, norm

from bff.bayes.gaussian_process import LGPCommittee
from examples.calvados.fetch_data import CHECKSUMS, COMMIT, RAW, verified_bytes

OUTPUT = Path(__file__).resolve().parent / "generated" / "priors"
MODELS = OUTPUT.parent / "discrepancy_direct"


def beta_parameters(values):
    """Moment estimates using population variance, retaining endpoint values."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError(
            "CALVADOS beta prior: expected a finite scale-by-residue matrix"
        )
    if np.any((values < 0) | (values > 1)):
        raise ValueError("CALVADOS beta prior: scales must be in [0, 1]")
    means = values.mean(axis=0)
    variances = values.var(axis=0)
    if np.any((variances <= 0) | (variances >= means * (1 - means))):
        raise ValueError(
            "CALVADOS beta prior: moments do not define positive beta shapes"
        )
    concentration = means * (1 - means) / variances - 1
    return means * concentration, (1 - means) * concentration


@dataclass
class StickinessPrior:
    """Joint log density, with columns in the saved GP parameter order."""

    kind: str
    parameter_names: tuple[str, ...]
    centers: np.ndarray | None = None
    bandwidth: float = 0.05
    alpha: np.ndarray | None = None
    beta: np.ndarray | None = None

    def __post_init__(self):
        self.parameter_names = tuple(self.parameter_names)
        dimensions = len(self.parameter_names)
        if not dimensions or len(set(self.parameter_names)) != dimensions:
            raise ValueError(
                "CALVADOS prior: parameter names must be nonempty and unique"
            )
        if self.kind not in ("uniform", "kde", "beta"):
            raise ValueError(f"CALVADOS prior: unknown kind {self.kind!r}")
        if self.kind == "kde":
            self.centers = np.asarray(self.centers, dtype=float)
            if (
                self.centers.ndim != 2
                or self.centers.shape[1] != dimensions
                or len(self.centers) == 0
                or not np.isfinite(self.centers).all()
                or np.any((self.centers < 0) | (self.centers > 1))
                or not np.isfinite(self.bandwidth)
                or self.bandwidth <= 0
            ):
                raise ValueError("CALVADOS KDE prior: invalid centers or bandwidth")
            self.component_masses = ndtr((1 - self.centers) / self.bandwidth) - ndtr(
                -self.centers / self.bandwidth
            )
            log_masses = np.log(self.component_masses).sum(axis=1)
            self.log_normalizer = float(
                logsumexp(log_masses) - math.log(len(log_masses))
            )
            self.component_weights = np.exp(log_masses - logsumexp(log_masses))
        if self.kind == "beta":
            self.alpha = np.asarray(self.alpha, dtype=float)
            self.beta = np.asarray(self.beta, dtype=float)
            if any(
                values.shape != (dimensions,)
                or not np.isfinite(values).all()
                or np.any(values <= 0)
                for values in (self.alpha, self.beta)
            ):
                raise ValueError("CALVADOS beta prior: expected positive finite shapes")

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        """One log density per row; no prior tempering is applied."""
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)
        if theta.ndim != 2 or theta.shape[1] != len(self.parameter_names):
            raise ValueError("CALVADOS prior: input columns must match parameter_names")
        valid = torch.isfinite(theta).all(dim=1)
        valid &= ((theta >= 0) & (theta <= 1)).all(dim=1)
        if self.kind == "beta":
            valid &= ((theta > 0) & (theta < 1)).all(dim=1)
        safe = torch.where(valid[:, None], theta, 0.5)
        if self.kind == "uniform":
            result = torch.zeros(theta.shape[0], dtype=theta.dtype, device=theta.device)
        elif self.kind == "kde":
            centers = torch.as_tensor(
                self.centers, dtype=theta.dtype, device=theta.device
            )
            distance = ((safe[:, None, :] - centers) / self.bandwidth).square().sum(-1)
            result = torch.logsumexp(-0.5 * distance, dim=1)
            normalization = math.log(len(self.centers)) + len(
                self.parameter_names
            ) * math.log(self.bandwidth * math.sqrt(2 * math.pi))
            result = result - normalization - self.log_normalizer
        else:
            alpha = torch.as_tensor(self.alpha, dtype=theta.dtype, device=theta.device)
            beta_shape = torch.as_tensor(
                self.beta, dtype=theta.dtype, device=theta.device
            )
            result = (
                (alpha - 1) * safe.log()
                + (beta_shape - 1) * torch.log1p(-safe)
                + torch.lgamma(alpha + beta_shape)
                - torch.lgamma(alpha)
                - torch.lgamma(beta_shape)
            ).sum(dim=1)
        return torch.where(valid, result, -torch.inf)

    def to_dict(self):
        record = {"kind": self.kind, "parameter_names": list(self.parameter_names)}
        if self.kind == "kde":
            record.update(centers=self.centers.tolist(), bandwidth=self.bandwidth)
        if self.kind == "beta":
            record.update(alpha=self.alpha.tolist(), beta=self.beta.tolist())
        return record


def load_priors(path=OUTPUT / "priors.json"):
    records = json.loads(Path(path).read_text())["priors"]
    return {name: StickinessPrior(**record) for name, record in records.items()}


def ordered_scales(scales, residues, parameter_names):
    if scales.index.has_duplicates or residues["three"].duplicated().any():
        raise ValueError("CALVADOS priors: duplicate residue IDs")
    mapping = residues.set_index("three")["one"]
    identifiers = []
    for name in parameter_names:
        kind, residue = name.split()
        if kind != "lambda":
            raise ValueError(f"CALVADOS prior: expected a lambda parameter, got {name}")
        identifiers.append(mapping.loc[residue])
    return scales.loc[identifiers].to_numpy(float).T


def main():
    plt.switch_backend("Agg")
    models = [
        LGPCommittee.load(MODELS / f"{name}.lgp") for name in ("chi2_rg", "chi2_pre")
    ]
    names = models[0].parameter_names
    if names != models[1].parameter_names or len(names) != 20:
        raise ValueError(
            "CALVADOS priors: original 80/20 models must share 20 parameters"
        )
    scales = pd.read_csv(io.BytesIO(verified_bytes(RAW / "selHPS.csv")), index_col=0)
    residues = pd.read_csv(io.BytesIO(verified_bytes(RAW / "residues.csv")))
    if scales.shape != (20, 70) or scales.columns.has_duplicates:
        raise ValueError(
            "CALVADOS priors: expected 70 hydrophobicity scales for 20 residues"
        )
    values = ordered_scales(scales, residues, names)
    alpha, beta_shape = beta_parameters(values)
    priors = {
        "tophat": StickinessPrior("uniform", names),
        "kde": StickinessPrior("kde", names, centers=values, bandwidth=0.05),
        "independent_beta": StickinessPrior(
            "beta", names, alpha=alpha, beta=beta_shape
        ),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    document = {
        "upstream_commit": COMMIT,
        "source_git_blob": CHECKSUMS["selHPS.csv"],
        "scale_names": list(scales.columns),
        "support": "[0,1]^20; beta evaluated only on the open cube",
        "beta_fit": "method of moments; population variance; no endpoint clipping",
        "kde_fit": (
            "equal-weight isotropic Gaussian kernels, bandwidth 0.05; "
            "globally conditioned on [0,1]^20"
        ),
        "prior_exponent": 1.0,
        "priors": {name: prior.to_dict() for name, prior in priors.items()},
    }
    (OUTPUT / "priors.json").write_text(json.dumps(document, indent=2) + "\n")
    pd.DataFrame(
        {
            "parameter": names,
            "alpha": alpha,
            "beta": beta_shape,
            "scale_mean": values.mean(axis=0),
            "scale_variance": values.var(axis=0),
        }
    ).to_csv(OUTPUT / "beta_parameters.csv", index=False)
    figure, axes = plt.subplots(4, 5, figsize=(15, 10), sharex=True)
    grid = np.linspace(0.001, 0.999, 500)
    kde = priors["kde"]
    # The priors themselves use exponent 1 (recorded in priors.json); the
    # paper's theta = 0.05 tempering is a posterior-sampling choice, the
    # same PRIOR_EXPONENT as examples/calvados/sample_posterior.py.
    exponent = 0.05
    bandwidth_tempered = kde.bandwidth / math.sqrt(exponent)
    grid_mass = np.trapezoid(np.ones_like(grid), grid)
    tempered_curves = []
    integrations = []
    for column, axis in enumerate(axes.flat):
        axis.hist(
            values[:, column],
            bins=10,
            range=(0, 1),
            density=True,
            alpha=0.2,
            color="gray",
            label="Scales",
        )
        axis.plot(grid, np.ones_like(grid), label="Tophat")
        marginal = (
            norm.pdf(grid[:, None], loc=values[:, column], scale=kde.bandwidth)
            / kde.component_masses[:, column]
            * kde.component_weights
        ).sum(axis=1)
        beta_pdf = beta.pdf(grid, alpha[column], beta_shape[column])
        # every plotted density is normalized to unit integral on the
        # plotting grid, uniformly; the integrals are audited below
        marginal = marginal * grid_mass / np.trapezoid(marginal, grid)
        beta_pdf = beta_pdf * grid_mass / np.trapezoid(beta_pdf, grid)
        axis.plot(grid, marginal, label="Joint KDE marginal")
        axis.plot(grid, beta_pdf, label="Beta")

        # --- tempered (pi ** exponent) marginals, probability space, on
        # the same axis and scale as everything else. The tempered beta
        # is exactly Beta(e*(a-1)+1, e*(b-1)+1); the tempered KDE
        # marginal is approximated per column by inflating the kernel
        # bandwidth to bandwidth/sqrt(e) and renormalizing the
        # per-column masses.
        tempered_alpha = exponent * (alpha[column] - 1) + 1
        tempered_beta = exponent * (beta_shape[column] - 1) + 1
        tempered_beta_pdf = beta.pdf(grid, tempered_alpha, tempered_beta)
        centers = values[:, column]
        masses = norm.cdf((1 - centers) / bandwidth_tempered) - norm.cdf(
            -centers / bandwidth_tempered
        )
        weights = masses / masses.sum()
        tempered_kde = (
            weights
            * norm.pdf(grid[:, None], loc=centers, scale=bandwidth_tempered)
        ).sum(axis=1)
        tempered_kde = tempered_kde * grid_mass / np.trapezoid(tempered_kde, grid)
        tempered_beta_pdf = (
            tempered_beta_pdf * grid_mass / np.trapezoid(tempered_beta_pdf, grid)
        )
        tempered_curves.append((tempered_kde, tempered_beta_pdf))
        integrals = {
            "joint KDE marginal": float(
                np.trapezoid(marginal, grid)
            ),
            "beta": float(np.trapezoid(beta_pdf, grid)),
            "tempered KDE (approx marginal)": float(
                np.trapezoid(tempered_kde, grid)
            ),
            "tempered beta": float(np.trapezoid(tempered_beta_pdf, grid)),
            "reference (flat 1)": float(grid_mass),
        }
        integrations.append((names[column], integrals))
        axis.plot(
            grid, tempered_kde, color="C3", linestyle=":",
            label=f"KDE^{exponent:g} marginal",
        )
        axis.plot(
            grid, tempered_beta_pdf, color="C4", linestyle="--",
            label=f"Beta^{exponent:g} marginal",
        )
        axis.set_title(names[column])
        axis.set_xlim(0, 1)
    axes[0, 0].legend(fontsize=7, loc="upper left")
    figure.supxlabel("Stickiness λ")
    figure.supylabel("Marginal prior density")
    figure.tight_layout()
    figure.savefig(OUTPUT / "prior_marginals.png", dpi=180)
    plt.close(figure)

    print(f"Prepared three 20-dimensional priors from {len(values)} scales: {OUTPUT}")
    print(
        "KDE probability inside [0,1]^20 before conditioning: "
        f"{np.exp(kde.log_normalizer):.6f}"
    )
    print("per-residue integrals of every plotted curve (trapezoid, [0,1]):")
    for name, integrals in integrations:
        rendered = ", ".join(
            f"{kind}={value:.4f}" for kind, value in integrals.items()
        )
        print(f"  {name}: {rendered}")


if __name__ == "__main__":
    main()
