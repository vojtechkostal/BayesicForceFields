import hashlib
import importlib

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

build_datasets = importlib.import_module(
    "examples.calvados.convert_data"
).build_datasets
fetch_data = importlib.import_module("examples.calvados.fetch_data")
CHECKSUMS = fetch_data.CHECKSUMS
verified_bytes = fetch_data.verified_bytes


def training_tables():
    residues = pd.DataFrame(
        {
            "lambdas_0": np.linspace(0.1, 0.8, 20),
            "lambdas_1": np.linspace(0.2, 0.9, 20),
        },
        index=list("ACDEFGHIKLMNPQRSTVWY"),
    )
    snapshots = {}
    for cycle in (1, 2):
        snapshots[cycle] = pd.DataFrame(
            [
                {
                    "fasta": list("AG"),
                    "temp": 298,
                    "ionic": 0.15,
                    "pH": 7.0,
                    "expRg": 1.2,
                    "expRgErr": 0.1,
                    "Rg": cycle + offset,
                    "rgarray": np.array([cycle + offset] * 3),
                    "path": f"{protein}/{cycle}",
                }
                for protein, offset in (("first", 0.1), ("second", 0.2))
            ],
            index=["first", "second"],
        )
    return residues, snapshots


def test_calvados_pairs_previous_parameters_and_protein_ids():
    residues, snapshots = training_tables()
    snapshots[2] = snapshots[2].iloc[::-1]
    datasets = build_datasets(residues.iloc[::-1], snapshots)
    first = datasets[0]
    np.testing.assert_allclose(first.X[0], residues.lambdas_0)
    np.testing.assert_allclose(first.X[1], residues.lambdas_1)
    np.testing.assert_allclose(first.y[:, 0], [1.1, 2.1])
    assert first.nuisance == 0.1
    assert first.labels == ("first",)
    assert first.sample_ids == ("M1_24_cycle_1", "M1_24_cycle_2")


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("temp", 310, "inconsistent temp"),
        ("path", "first/1", "snapshot path"),
        ("Rg", 99, "per-frame mean"),
    ],
)
def test_calvados_rejects_mismatched_snapshots(column, value, message):
    residues, snapshots = training_tables()
    snapshots[2].loc["first", column] = value
    with pytest.raises(ValueError, match=message):
        build_datasets(residues, snapshots)


def test_calvados_rejects_changed_download(tmp_path, monkeypatch):
    path = tmp_path / "residues.csv"
    data = b"original\n"
    monkeypatch.setitem(
        CHECKSUMS,
        path.name,
        hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest(),
    )
    path.write_bytes(data)
    assert verified_bytes(path) == data
    path.write_bytes(b"changed\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verified_bytes(path)
