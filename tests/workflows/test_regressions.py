from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from bff.io.cp2k import (
    HARTREE_TO_EV,
    collect_single_atom_energies,
)
from bff.workflows.label_snapshots.config import LabelSnapshotsConfig
from bff.workflows.learn.config import LearnConfig
from bff.workflows.learn.main import (
    _qoi_log_likelihood_in_batches,
    _write_default_plots,
)
from bff.workflows.md.main import check_success


def test_collect_single_atom_energies_uses_atomic_numbers(tmp_path: Path) -> None:
    hydrogen = tmp_path / "hydrogen"
    calcium = tmp_path / "calcium"
    hydrogen.mkdir()
    calcium.mkdir()

    hydrogen.joinpath("pos.xyz").write_text("1\ncomment\nH 0.0 0.0 0.0\n")
    calcium.joinpath("pos.xyz").write_text("1\ncomment\nCa 0.0 0.0 0.0\n")
    hydrogen.joinpath("atom.out").write_text(
        " ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]:      -0.500000\n"
    )
    calcium.joinpath("atom.out").write_text(
        " ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]:      -1.250000\n"
    )

    energies = collect_single_atom_energies([hydrogen, calcium])

    assert set(energies) == {1, 20}
    assert energies[1] == pytest.approx(-0.5 * HARTREE_TO_EV)
    assert energies[20] == pytest.approx(-1.25 * HARTREE_TO_EV)


def test_label_snapshots_rejects_import_mode(
    tmp_path: Path,
) -> None:
    fn_config = tmp_path / "label-snapshots.yaml"
    fn_config.write_text(
        yaml.safe_dump(
            {
                "mode": "import",
                "output_dir": "./trajectories",
                "systems": [],
            }
        )
    )

    with pytest.raises(ValueError, match="unsupported key.*mode"):
        LabelSnapshotsConfig.load(fn_config)


def test_check_success_uses_expected_saved_frame_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fn_trj = tmp_path / "traj.xtc"
    fn_trj.write_text("dummy xtc placeholder\n")

    monkeypatch.setattr(
        "bff.workflows.md.main.get_n_frames_target",
        lambda _: (1000, 500),
    )

    class DummyReader:
        def __init__(self, _: str) -> None:
            self.n_frames = 101

        def __enter__(self) -> "DummyReader":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    import MDAnalysis.coordinates.XTC as xtc_module

    monkeypatch.setattr(xtc_module, "XTCReader", DummyReader)

    assert check_success(fn_trj, tmp_path / "prod.mdp", 50000) is True


def test_check_success_returns_false_for_too_few_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fn_trj = tmp_path / "traj.xtc"
    fn_trj.write_text("dummy xtc placeholder\n")

    monkeypatch.setattr(
        "bff.workflows.md.main.get_n_frames_target",
        lambda _: (1000, 500),
    )

    class DummyReader:
        def __init__(self, _: str) -> None:
            self.n_frames = 100

        def __enter__(self) -> "DummyReader":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    import MDAnalysis.coordinates.XTC as xtc_module

    monkeypatch.setattr(xtc_module, "XTCReader", DummyReader)

    assert check_success(fn_trj, tmp_path / "prod.mdp", 50000) is False


def test_write_default_plots_writes_expected_pngs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyResults:
        def __init__(self) -> None:
            self.prepared = False
            self.prepared_samples = np.linspace(-0.5, 0.5, 20)[:, None]
            self.include_implicit_charge = False

        def prepare_samples(self, **kwargs) -> None:
            assert kwargs == {"discard": 0, "thin": 1}
            self.prepared = True

    calls: dict[str, Path] = {}

    def fake_plot_marginals(results, specs, *, fn_out=None, **kwargs):
        assert results.prepared is True
        assert kwargs["max_samples"] == 11
        assert kwargs["plot_metadata"] == {
            "define VSA": {"xlabel": "O-VS", "ylabel": "angle [degree]"}
        }
        calls["marginals"] = Path(fn_out)
        Path(fn_out).write_text("marginals\n")

    def fake_plot_qoi_marginals(
        results,
        specs,
        contributions,
        *,
        fn_out=None,
        **kwargs,
    ):
        assert results.prepared is True
        assert set(contributions) == {"qoi"}
        assert len(contributions["qoi"]) == 7
        assert kwargs["sample_indices"].tolist() == [0, 3, 6, 9, 12, 15, 19]
        assert kwargs["plot_metadata"] == {
            "define VSA": {"xlabel": "O-VS", "ylabel": "angle [degree]"}
        }
        calls["qoi_marginals"] = Path(fn_out)
        Path(fn_out).write_text("qoi marginals\n")

    def fake_plot_corner(results, *, fn_out=None, **kwargs):
        assert results.prepared is True
        assert kwargs["max_samples"] == 5
        calls["corner"] = Path(fn_out)
        Path(fn_out).write_text("corner\n")

    monkeypatch.setattr("bff.plotting.plot_marginals", fake_plot_marginals)
    monkeypatch.setattr(
        "bff.plotting.plot_qoi_marginals",
        fake_plot_qoi_marginals,
    )
    monkeypatch.setattr("bff.plotting.plot_corner", fake_plot_corner)
    likelihood_batch_sizes = []

    def fake_log_likelihood(theta, problem):
        likelihood_batch_sizes.append(len(theta))
        return {"qoi": torch.zeros(len(theta), device=theta.device)}

    monkeypatch.setattr(
        "bff.bayes.likelihoods.gaussian_log_likelihood_by_qoi",
        fake_log_likelihood,
    )

    specs = tmp_path / "specs.yaml"
    specs.write_text(
        "bounds: {}\n"
        "charge_constraints: []\n"
    )
    model = tmp_path / "model.pt"
    model.write_text("model\n")

    fn_config = tmp_path / "learn.yaml"
    fn_config.write_text(
        yaml.safe_dump(
                {
                    "specs": str(specs),
                    "models": {
                        "qoi": {
                            "model_path": str(model),
                            "independent_observations": True,
                        }
                    },
                    "mcmc": {},
                    "plots": {
                        "max_corner_samples": 5,
                        "max_marginal_samples": 11,
                        "max_qoi_samples": 7,
                        "qoi_batch_size": 3,
                        "plot_metadata": {
                            "define VSA": {
                                "xlabel": "O-VS",
                                "ylabel": "angle [degree]",
                            }
                        },
                    },
                    "output": {"directory": str(tmp_path / "learn")},
                }
        )
    )

    config = LearnConfig.load(fn_config)
    problem = SimpleNamespace(
        n_params=1,
        models={
            "qoi": SimpleNamespace(
                lgps=[SimpleNamespace(X_train=torch.zeros((1, 1)))]
            )
        },
        to_torch=lambda device: problem,
    )

    config.output.plots_dir.mkdir(parents=True)
    _write_default_plots(DummyResults(), config, problem)

    assert calls["marginals"].read_text() == "marginals\n"
    assert calls["qoi_marginals"].read_text() == "qoi marginals\n"
    assert calls["corner"].read_text() == "corner\n"
    assert calls["marginals"].name == "marginals.pdf"
    assert calls["qoi_marginals"].name == "qoi-marginals.pdf"
    assert calls["corner"].name == "corner.pdf"
    assert likelihood_batch_sizes == [3, 3, 1]


def test_qoi_likelihood_batches_shrink_after_cuda_oom(monkeypatch) -> None:
    real_as_tensor = torch.as_tensor
    attempted_batch_sizes = []
    cache_clears = []

    def cpu_as_tensor(values, **kwargs):
        kwargs.pop("device", None)
        return real_as_tensor(values, **kwargs)

    def fake_log_likelihood(theta, problem):
        attempted_batch_sizes.append(len(theta))
        if len(theta) > 2:
            raise torch.OutOfMemoryError("synthetic CUDA OOM")
        return {"qoi": theta[:, 0]}

    monkeypatch.setattr(torch, "as_tensor", cpu_as_tensor)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: cache_clears.append(True))
    monkeypatch.setattr(
        "bff.bayes.likelihoods.gaussian_log_likelihood_by_qoi",
        fake_log_likelihood,
    )

    problem = SimpleNamespace(to_torch=lambda device: problem)
    samples = np.arange(10, dtype=float).reshape(5, 2)
    contributions = _qoi_log_likelihood_in_batches(
        samples,
        problem,
        torch.device("cuda"),
        batch_size=5,
    )

    assert attempted_batch_sizes == [5, 2, 2, 1]
    assert cache_clears == [True]
    assert contributions["qoi"].tolist() == [0.0, 2.0, 4.0, 6.0, 8.0]
