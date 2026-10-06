"""Small CPU runs of build-qoi-datasets, fit-lgp, and learn on real files."""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from bff.bayes.gaussian_process import LGPCommittee, LocalGaussianProcess
from bff.workflows.build_qoi_datasets.main import main as build_qoi_datasets_main
from bff.workflows.fit_lgp.main import main as fit_lgp_main
from bff.workflows.learn.main import main as learn_main

pytestmark = pytest.mark.slow


def _write(path: Path, text: str = "data\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_small_cpu_learning_run_writes_fixed_artifacts_and_resumes(
    tmp_path: Path,
) -> None:
    specs = tmp_path / "specs.yaml"
    specs.write_text(
        yaml.safe_dump({"bounds": {"sigma A": [0.5, 1.5]}, "charge_constraints": []})
    )
    x_train = torch.tensor([[0.5], [0.8], [1.2], [1.5]])
    y_train = torch.tensor([[0.6], [0.9], [1.1], [1.4]])
    lgp = LocalGaussianProcess(
        x_train,
        y_train,
        0.0,
        torch.ones(1),
        1.0,
        0.1,
        "cpu",
    )
    model = LGPCommittee(
        [lgp],
        np.asarray([1.0]),
        n_curves=1,
        nuisance=0.1,
        dataset_fingerprint="fixture",
    )
    model_path = tmp_path / "pmf.lgp"
    model.write(model_path)

    def write_config(*, steps: int, resume: bool) -> Path:
        path = tmp_path / "learn.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "specs": str(specs),
                    "models": {"pmf": {"model_path": str(model_path), "n_eff": 1.0}},
                    "mcmc": {
                        "total_steps": steps,
                        "warmup": 2,
                        "n_walkers": 4,
                        "progress_stride": 2,
                        "resume": resume,
                        "device": "cpu",
                    },
                    "output": {"directory": "run"},
                }
            )
        )
        return path

    learn_main(write_config(steps=8, resume=False))
    output = tmp_path / "run"
    mandatory = [
        output / "learn.log",
        output / "outputs" / "specs.yaml",
        output / "outputs" / "prior.pt",
        output / "outputs" / "posterior.pt",
        output / "outputs" / "mcmc.ckpt",
        output / "plots" / "marginals.pdf",
        output / "plots" / "qoi-marginals.pdf",
        output / "plots" / "corner.pdf",
    ]
    assert all(path.is_file() for path in mandatory)
    assert (output / "outputs" / "specs.yaml").read_bytes() == specs.read_bytes()
    posterior = torch.load(output / "outputs" / "posterior.pt", weights_only=False)
    assert posterior["metadata"]["specifications_fingerprint"]
    assert posterior["metadata"]["model_fingerprints"]["pmf"]
    learn_log = (output / "learn.log").read_text()
    assert "effective observations: 1.000\n\n" in learn_log
    assert "corner.pdf\n\n> Posterior learning" in learn_log

    learn_main(write_config(steps=10, resume=True))
    assert "Posterior Learning (resumed)" in (output / "learn.log").read_text()


def test_real_cpu_file_pipeline_build_qoi_fit_lgp_learn(tmp_path: Path) -> None:
    sample_dir = tmp_path / "sample"
    topology = _write(sample_dir / "systems" / "acetate" / "topology.top")
    coordinates = _write(sample_dir / "systems" / "acetate" / "coordinates.gro")
    specs = sample_dir / "specs.yaml"
    specs.write_text(
        yaml.safe_dump({"bounds": {"sigma A": [0.5, 1.5]}, "charge_constraints": []})
    )
    sample_records = {}
    for index, (parameter, value) in enumerate(
        [(0.55, 0.62), (0.8, 0.84), (1.2, 1.18), (1.45, 1.39)]
    ):
        sample_id = f"sample-{index}"
        trajectory = _write(
            sample_dir / "samples" / sample_id / "acetate" / "trajectory.xtc"
        )
        profile = _write(
            sample_dir / "samples" / sample_id / "acetate" / "profile.pmf",
            f"{value}\n",
        )
        sample_records[sample_id] = {
            "params": [parameter],
            "status": "completed",
            "outputs": [
                {
                    "system_id": "acetate",
                    "trajectory": str(trajectory.relative_to(sample_dir)),
                    "inputs": {"pmf": str(profile.relative_to(sample_dir))},
                }
            ],
        }
    (sample_dir / "samples.yaml").write_text(
        yaml.safe_dump(
            {
                "systems": [
                    {
                        "system_id": "acetate",
                        "topology": str(topology.relative_to(sample_dir)),
                        "coordinates": str(coordinates.relative_to(sample_dir)),
                    }
                ],
                "samples": sample_records,
            }
        )
    )

    reference_profile = _write(tmp_path / "reference.pmf", "1.0\n")
    routine_module = _write(
        tmp_path / "pmf.py",
        "from bff.qoi.dataset import QoI\n"
        "def load_profile(*, inputs, system_id, sample_id, options):\n"
        "    value = float(inputs['pmf'].read_text())\n"
        "    return QoI('ignored', [value], labels=('profile',))\n",
    )
    qoi_config = tmp_path / "build-qoi-datasets.yaml"
    qoi_config.write_text(
        yaml.safe_dump(
            {
                "training_samples": {
                    "manifest": str(sample_dir / "samples.yaml"),
                    "systems": [{"system_id": "acetate"}],
                    "workers": 2,
                },
                "reference": {
                    "systems": [
                        {
                            "system_id": "acetate",
                            "inputs": {"pmf": str(reference_profile)},
                        }
                    ]
                },
                "routines": [
                    {
                        "name": "pmf",
                        "callable": f"{routine_module}:load_profile",
                        "systems": ["acetate"],
                        "inputs": ["pmf"],
                    }
                ],
                "output": {"directory": "qoi"},
            }
        )
    )
    build_qoi_datasets_main(qoi_config)
    qoi_log = (tmp_path / "build-qoi-datasets.log").read_text()
    reference_start = qoi_log.index("Reference QoI: in progress")
    reference_end = qoi_log.index("Done. Finished", reference_start)
    training_start = qoi_log.index("Training QoI: in progress")
    training_end = qoi_log.index("Done. Finished", training_start)
    dataset_start = qoi_log.index("QoI dataset: Done.")
    assert qoi_log[reference_end:training_start].count("\n") >= 2
    assert qoi_log[training_end:dataset_start].count("\n") >= 2

    fit_config = tmp_path / "fit-lgp.yaml"
    fit_config.write_text(
        yaml.safe_dump(
            {
                "datasets": {"pmf": {"data": "qoi/pmf.pt", "nuisance": 0.1}},
                "fit": {
                    "model_dir": "models",
                    "reuse_models": False,
                    "n_hyper_max": 3,
                    "committee_size": 1,
                    "test_fraction": 0.25,
                    "device": "cpu",
                    "lr": 0.001,
                    "max_iter": 20,
                },
            }
        )
    )
    fit_lgp_main(fit_config)

    learn_config = tmp_path / "pipeline-learn.yaml"
    learn_config.write_text(
        yaml.safe_dump(
            {
                "specs": str(specs),
                "models": {"pmf": {"model_path": "models/pmf.lgp", "n_eff": 1.0}},
                "mcmc": {
                    "total_steps": 8,
                    "warmup": 2,
                    "n_walkers": 4,
                    "progress_stride": 2,
                    "device": "cpu",
                },
                "output": {"directory": "pipeline-learn"},
            }
        )
    )
    learn_main(learn_config)

    assert (tmp_path / "qoi" / "pmf.pt").is_file()
    assert (tmp_path / "models" / "pmf.lgp").is_file()
    assert (tmp_path / "pipeline-learn" / "outputs" / "posterior.pt").is_file()
