"""Convert verified M1_24 snapshots to one BFF Rg dataset per protein."""

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bff.qoi.dataset import QoIDataset
from examples.calvados.fetch_data import COMMIT, RAW, verified_bytes


def build_datasets(residues, snapshots):
    if residues.index.has_duplicates or set(residues.index) != set(
        "ACDEFGHIKLMNPQRSTVWY"
    ):
        raise ValueError("CALVADOS residues.csv: expected 20 unique amino-acid IDs")
    residues = residues.sort_index()
    cycles = sorted(snapshots)
    reference = snapshots[cycles[0]]
    if reference.index.has_duplicates:
        raise ValueError("CALVADOS snapshot: duplicate protein IDs")
    parameter_names = tuple(f"lambda {name}" for name in residues.index)
    parameters = np.array(
        [residues[f"lambdas_{cycle - 1}"].to_numpy(float) for cycle in cycles]
    )
    if not np.isfinite(parameters).all() or np.any(
        (parameters <= 0) | (parameters > 1)
    ):
        raise ValueError("CALVADOS residues.csv: lambdas must be finite in (0, 1]")
    for cycle, snapshot in snapshots.items():
        if snapshot.index.has_duplicates or set(snapshot.index) != set(reference.index):
            raise ValueError(f"CALVADOS cycle {cycle}: inconsistent protein IDs")
    datasets = []
    for protein_id in sorted(reference.index):
        original = reference.loc[protein_id]
        sequence = "".join(original.fasta)
        conditions = {key: float(original[key]) for key in ("temp", "ionic", "pH")}
        target, error = float(original.expRg), float(original.expRgErr)
        if not np.isfinite([*conditions.values(), target, error]).all():
            raise ValueError(f"CALVADOS {protein_id}: nonfinite conditions or target")
        if target <= 0 or error <= 0 or not sequence:
            raise ValueError(
                f"CALVADOS {protein_id}: invalid target, error or sequence"
            )
        outputs = []
        for cycle in cycles:
            row = snapshots[cycle].loc[protein_id]
            context = f"CALVADOS {protein_id}, cycle {cycle}"
            if str(row.path) != f"{protein_id}/{cycle}":
                raise ValueError(f"{context}: unexpected snapshot path {row.path}")
            if "".join(row.fasta) != sequence:
                raise ValueError(f"{context}: sequence changed")
            for key, value in {
                **conditions,
                "expRg": target,
                "expRgErr": error,
            }.items():
                if not np.isclose(float(row[key]), value, rtol=0, atol=1e-10):
                    raise ValueError(f"{context}: inconsistent {key}")
            frames = np.asarray(row.rgarray, dtype=float)
            value = float(row.Rg)
            if (
                frames.ndim != 1
                or not frames.size
                or not np.isfinite(frames).all()
                or np.any(frames <= 0)
                or not np.isclose(value, frames.mean())
            ):
                raise ValueError(f"{context}: Rg does not match finite per-frame mean")
            outputs.append([value])
        datasets.append(
            QoIDataset(
                name=f"rg_{protein_id}",
                X=parameters,
                y=np.asarray(outputs),
                y_ref=np.array([target]),
                nuisance=error,
                labels=(protein_id,),
                parameter_names=parameter_names,
                sample_ids=tuple(f"M1_24_cycle_{cycle}" for cycle in cycles),
                metadata={
                    "upstream_commit": COMMIT,
                    "run": "M1_24",
                    "cutoff_nm": 2.4,
                    "rg_units": "nm",
                    "temperature_units": "K",
                    "ionic_strength_units": "M",
                    "sequence": sequence,
                    "conditions": conditions,
                    "reference_source": "upstream snapshot expRg/expRgErr",
                    "parameter_columns": [f"lambdas_{cycle - 1}" for cycle in cycles],
                },
            )
        )
    return datasets


def main():
    residues = pd.read_csv(io.BytesIO(verified_bytes(RAW / "residues.csv")))
    snapshots = {
        cycle: pd.read_pickle(
            io.BytesIO(verified_bytes(RAW / f"{cycle}_init_proteinsRgs.pkl"))
        )
        for cycle in range(1, 12)
    }
    datasets = build_datasets(residues.set_index("one"), snapshots)
    output = Path(__file__).resolve().parent / "generated" / "qoi"
    output.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        dataset.write(str(output / f"{dataset.name}.pt"))
    summary = {
        "run": "M1_24",
        "upstream_commit": COMMIT,
        "proteins": len(datasets),
        "parameter_sets": datasets[0].n_samples,
        "parameters": len(datasets[0].parameter_names),
        "centered_parameter_rank": int(
            np.linalg.matrix_rank(datasets[0].X - datasets[0].X.mean(axis=0))
        ),
        "reference_source": "upstream snapshots; c2_rg.csv is not merged",
    }
    (output.parent / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
