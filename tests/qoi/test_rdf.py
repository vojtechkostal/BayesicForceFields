from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest

from bff.qoi.rdf import rdf

REFERENCE = np.load(Path(__file__).with_name("reference_values.npz"))


def _rdf(universe, **options):
    return rdf(universe, frames=slice(None), options=options)


def _three_atoms() -> mda.Universe:
    universe = mda.Universe.empty(3, trajectory=True)
    universe.add_TopologyAttr("types", ["O", "H", "OW"])
    universe.add_TopologyAttr("masses", [16.0, 1.0, 16.0])
    universe.load_new(
        np.asarray(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.5, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [2.8, 0.0, 0.0]],
            ]
        ),
        format="MEMORY",
        dimensions=np.asarray(
            [
                [10.0, 11.0, 12.0, 80.0, 95.0, 105.0],
                [12.0, 13.0, 14.0, 75.0, 90.0, 110.0],
            ]
        ),
    )
    return universe


@pytest.mark.parametrize("smooth", [False, True])
def test_rdf_reproduces_reference_values(solvated, smooth: bool) -> None:
    qoi = _rdf(
        solvated,
        group_a="resname LIG",
        group_b="type OW",
        range=[0.0, 8.0],
        bins=40,
        smooth=smooth,
    )
    assert qoi.labels == tuple(REFERENCE["rdf_labels"])
    assert qoi.values_per_label == 40
    np.testing.assert_array_equal(qoi.values, REFERENCE[f"rdf_smooth_{smooth}"])
    assert qoi.settings == {
        "group_a": "resname LIG",
        "group_b": "type OW",
        "range": (0.0, 8.0),
        "bins": 40,
        "pbc": True,
        "update_selections": False,
        "smooth": smooth,
    }


def test_rdf_with_updating_selections_reproduces_reference_values(solvated) -> None:
    qoi = _rdf(
        solvated,
        group_a="resname LIG and around 4 type OW",
        group_b="type OW",
        range=[0.0, 8.0],
        bins=40,
        update_selections=True,
    )
    assert qoi.labels == tuple(REFERENCE["rdf_updating_labels"])
    np.testing.assert_array_equal(qoi.values, REFERENCE["rdf_updating"])


def test_rdf_excludes_massless_centers() -> None:
    universe = _three_atoms()
    universe.atoms[1].mass = 0.0
    qoi = _rdf(universe, group_a="index 0 1", group_b="index 2", bins=10)
    assert qoi.labels == ("O",)


def test_rdf_rejects_frames_without_non_self_pairs() -> None:
    universe = _three_atoms()
    with pytest.raises(ValueError, match="no non-self atom pairs"):
        _rdf(universe, group_a="index 0", group_b="index 0", pbc=False)


def test_rdf_rejects_empty_frame_slice() -> None:
    universe = _three_atoms()
    with pytest.raises(ValueError, match="selects no trajectory frames"):
        rdf(
            universe,
            frames=slice(2, None),
            options={"group_a": "index 0", "group_b": "index 2"},
        )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"group_a": "index 0"}, "requires selection"),
        ({"group_a": "index 0", "group_b": "index 2", "binz": 3}, "unsupported"),
        ({"group_a": "index 0", "group_b": "index 2", "range": [5, 1]}, "range"),
        ({"group_a": "index 0", "group_b": "index 2", "bins": 0}, "bins"),
        ({"group_a": "index 0", "group_b": "index 2", "pbc": "yes"}, "booleans"),
        ({"group_a": "type XX", "group_b": "index 2"}, "empty"),
    ],
)
def test_rdf_validates_its_options(options, message) -> None:
    with pytest.raises(ValueError, match=message):
        _rdf(_three_atoms(), **options)


def test_rdf_requires_masses() -> None:
    universe = mda.Universe.empty(2, trajectory=True)
    universe.add_TopologyAttr("types", ["O", "OW"])
    with pytest.raises(ValueError, match="requires topology masses"):
        _rdf(universe, group_a="index 0", group_b="index 1")
