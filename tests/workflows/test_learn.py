import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from bff.bayes.priors import Priors
from bff.bayes.results import Results
from bff.domain.specs import Specs
from bff.workflows.learn.config import LearnConfig
from bff.workflows.learn.main import _prepare_output, _write_plots


def _write(path: Path, text: str = "data\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _learn_config(tmp_path: Path, *, resume: bool = False, overwrite: bool = False):
    _write(tmp_path / "specs.yaml")
    _write(tmp_path / "model.lgp")
    config_path = tmp_path / "learn.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "specs": "specs.yaml",
                "models": {
                    "pmf": {
                        "model_path": "model.lgp",
                    }
                },
                "mcmc": {"resume": resume},
                "output": {"directory": "run", "overwrite": overwrite},
            }
        )
    )
    return LearnConfig.load(config_path)


def test_learning_collision_policy_only_removes_owned_files(tmp_path: Path) -> None:
    config = _learn_config(tmp_path)
    _write(config.output.log)
    with pytest.raises(ValueError, match="already exist"):
        _prepare_output(config)

    config = _learn_config(tmp_path, overwrite=True)
    unknown = _write(config.output.directory / "keep-me.txt")
    _prepare_output(config)
    assert not config.output.log.exists()
    assert unknown.exists()


def test_learning_resume_requires_matching_copied_specs(tmp_path: Path) -> None:
    config = _learn_config(tmp_path)
    _prepare_output(config)
    _write(config.output.checkpoint)

    config.output.specs.unlink()
    with pytest.raises(ValueError, match="requires the copied specifications"):
        _prepare_output(_learn_config(tmp_path, resume=True))

    shutil.copy2(config.specs, config.output.specs)
    config.output.specs.write_text("bounds: {}\n")
    with pytest.raises(ValueError, match="do not match"):
        _prepare_output(_learn_config(tmp_path, resume=True))


def test_write_plots_saves_the_three_figures(tmp_path: Path) -> None:
    values = np.linspace(-0.5, 0.5, 40)
    specs = Specs({"bounds": {"charge A": [-1.0, 1.0]}, "charge_constraints": []})
    results = Results(
        values.reshape(10, 4, 1),
        -(values.reshape(10, 4) ** 2),
        specs,
        prior=Priors.from_bounds(specs.explicit_bounds, names=["charge A"]),
        qoi_index=np.arange(40),
        qoi_log_likelihood={"rdf": -(values**2), "hb": -((values - 0.2) ** 2)},
    )
    fn_specs = tmp_path / "specs.yaml"
    specs.write(fn_specs)
    fn_model = _write(tmp_path / "model.lgp")
    fn_config = tmp_path / "learn.yaml"
    fn_config.write_text(
        yaml.safe_dump(
            {
                "specs": str(fn_specs),
                "models": {"rdf": {"model_path": str(fn_model)}},
                "plots": {
                    "max_corner_samples": 5,
                    "max_marginal_samples": 11,
                    "plot_metadata": {"charge A": {"xlabel": "A"}},
                },
                "output": {"directory": str(tmp_path / "learn")},
            }
        )
    )
    config = LearnConfig.load(fn_config)
    config.output.plots_dir.mkdir(parents=True)

    _write_plots(results, config)

    for fn_plot in (
        config.output.marginals,
        config.output.qoi_marginals,
        config.output.corner,
    ):
        assert fn_plot.stat().st_size > 1000
    assert config.output.marginals.name == "marginals.pdf"
    assert config.output.qoi_marginals.name == "qoi-marginals.pdf"
    assert config.output.corner.name == "corner.pdf"
