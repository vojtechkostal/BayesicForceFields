from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


import pytest
import yaml

from bff.io.cp2k import (
    HARTREE_TO_EV,
    collect_single_atom_energies,
    strip_cp2k_gfn_type,
)
from bff.workflows.learn.config import LearnConfig
from bff.workflows.learn.main import _write_default_plots
from bff.workflows.md.main import check_success
from bff.workflows.reference.config import ReferenceConfig


def test_strip_cp2k_gfn_type_removes_keyword(tmp_path: Path) -> None:
    fn_input = tmp_path / "md.inp"
    fn_input.write_text(
        "&XTB\n"
        "  GFN_TYPE 1\n"
        "  SOME_OTHER_KEY 2\n"
        "&END XTB\n"
    )

    changed = strip_cp2k_gfn_type(fn_input)

    assert changed is True
    assert "GFN_TYPE" not in fn_input.read_text()
    assert "SOME_OTHER_KEY 2" in fn_input.read_text()


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


def test_reference_import_config_requires_top_gro_and_trajectory(
    tmp_path: Path,
) -> None:
    fn_top = tmp_path / "system.top"
    fn_gro = tmp_path / "system.gro"
    fn_trj = tmp_path / "traj.xtc"
    fn_top.write_text("; topology\n")
    fn_gro.write_text("dummy gro\n")
    fn_trj.write_text("trajectory\n")

    fn_config = tmp_path / "reference.yaml"
    fn_config.write_text(
        yaml.safe_dump(
            {
                "mode": "import",
                "reference_dir": "./reference-assets",
                "systems": [
                    {
                        "topology": str(fn_top),
                        "coordinates": str(fn_gro),
                        "trajectory": str(fn_trj),
                    }
                ],
            }
        )
    )

    config = ReferenceConfig.load(fn_config)

    assert config.mode == "import"
    assert len(config.systems) == 1
    system = config.systems[0]
    assert system.fn_topol == fn_top.resolve()
    assert system.fn_gro == fn_gro.resolve()
    assert system.fn_trj == fn_trj.resolve()


def test_reference_import_config_rejects_wrong_topology_suffix(
    tmp_path: Path,
) -> None:
    fn_top = tmp_path / "system.itp"
    fn_gro = tmp_path / "system.gro"
    fn_trj = tmp_path / "traj.xtc"
    fn_top.write_text("; wrong suffix\n")
    fn_gro.write_text("dummy gro\n")
    fn_trj.write_text("trajectory\n")

    fn_config = tmp_path / "reference.yaml"
    fn_config.write_text(
        yaml.safe_dump(
            {
                "mode": "import",
                "reference_dir": "./reference-assets",
                "systems": [
                    {
                        "topology": str(fn_top),
                        "coordinates": str(fn_gro),
                        "trajectory": str(fn_trj),
                    }
                ],
            }
        )
    )

    with pytest.raises(ValueError, match=r"\.top file"):
        ReferenceConfig.load(fn_config)


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

        def prepare_samples(self) -> None:
            self.prepared = True

    calls: dict[str, Path] = {}

    def fake_plot_marginals(results, specs, *, fn_out=None, **kwargs):
        assert results.prepared is True
        calls["marginals"] = Path(fn_out)
        Path(fn_out).write_text("marginals\n")

    def fake_plot_corner(results, *, fn_out=None, **kwargs):
        assert results.prepared is True
        calls["corner"] = Path(fn_out)
        Path(fn_out).write_text("corner\n")

    monkeypatch.setattr("bff.plotting.plot_marginals", fake_plot_marginals)
    monkeypatch.setattr("bff.plotting.plot_corner", fake_plot_corner)

    specs = tmp_path / "specs.yaml"
    specs.write_text(
        "mol_resname: ACE\n"
        "implicit_atoms: []\n"
        "bounds: {}\n"
        "total_charge: 0.0\n"
    )
    posterior = tmp_path / "posterior.pt"
    posterior.write_text("posterior\n")
    log = tmp_path / "learn.log"
    model = tmp_path / "model.pt"
    model.write_text("model\n")

    fn_config = tmp_path / "learn.yaml"
    fn_config.write_text(
        yaml.safe_dump(
            {
                "specs": str(specs),
                "models": {"qoi": str(model)},
                "mcmc": {"posterior": str(posterior)},
                "log": str(log),
            }
        )
    )

    config = LearnConfig.load(fn_config)
    warnings: list[str] = []
    logger = SimpleNamespace(
        kv=lambda *args, **kwargs: None,
        warn=lambda message, **kwargs: warnings.append(str(message)),
    )

    _write_default_plots(DummyResults(), config, logger)

    assert warnings == []
    assert calls["marginals"].read_text() == "marginals\n"
    assert calls["corner"].read_text() == "corner\n"
    assert calls["marginals"].name == "marginals.pdf"
    assert calls["corner"].name == "corner.pdf"
