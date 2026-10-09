from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest

from bff.qoi.hbonds import hydrogen_bonds

REFERENCE = np.load(Path(__file__).with_name("reference_values.npz"))


def _hydrogen_bonds(universe, **options):
    return hydrogen_bonds(universe, frames=slice(None), options=options)


@pytest.mark.parametrize("update_selections", [False, True])
def test_hydrogen_bonds_reproduce_reference_values(solvated, update_selections) -> None:
    water = (
        "resname SOL and around 5 resname LIG" if update_selections else "resname SOL"
    )
    qoi = _hydrogen_bonds(
        solvated,
        selection="resname LIG",
        water_selection=water,
        update_selections=update_selections,
    )
    assert qoi.labels == tuple(REFERENCE[f"hb_labels_{update_selections}"])
    np.testing.assert_array_equal(
        qoi.values, REFERENCE[f"hb_values_{update_selections}"]
    )
    assert qoi.settings["elements"] == ("N", "O", "S")


def test_hydrogen_bonds_detect_ambident_nitrogen_in_both_directions() -> None:
    universe = mda.Universe.empty(
        6, n_residues=2, atom_resindex=[0, 0, 0, 1, 1, 1], trajectory=True
    )
    universe.add_TopologyAttr("names", ["N", "HN1", "HN2", "OW", "HW1", "HW2"])
    universe.add_TopologyAttr("types", ["N", "H", "H", "O", "H", "H"])
    universe.add_TopologyAttr("elements", ["N", "H", "H", "O", "H", "H"])
    universe.add_TopologyAttr("resnames", ["AMN", "SOL"])
    universe.add_TopologyAttr("bonds", [(0, 1), (0, 2), (3, 4), (3, 5)])
    universe.load_new(
        np.asarray(
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [2.5, 0.0, 0.0],
                    [1.5, 0.0, 0.0],
                    [2.5, 1.0, 0.0],
                ]
            ]
        ),
        format="MEMORY",
        dimensions=np.asarray([[10.0, 10.0, 10.0, 90.0, 90.0, 90.0]]),
    )

    qoi = _hydrogen_bonds(
        universe, selection="resname AMN and name N", water_selection="resname SOL"
    )

    assert qoi.labels == ("AMN(N) to SOL(O)", "SOL(O) to AMN(N)")
    assert qoi.values.tolist() == [1.0, 1.0]


def test_hydrogen_bonds_require_bonds(solvated) -> None:
    universe = mda.Universe.empty(2, trajectory=True)
    universe.add_TopologyAttr("elements", ["O", "O"])
    universe.add_TopologyAttr("resnames", ["X"])
    with pytest.raises(ValueError, match="bond"):
        _hydrogen_bonds(universe, selection="index 0", water_selection="index 1")


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({}, "'selection'"),
        ({"selection": "resname LIG", "cutoff": 3}, "unsupported"),
        ({"selection": "resname LIG", "elements": ["O", "H"]}, "heavy atoms"),
        ({"selection": "resname LIG", "angle_cutoff": 200}, "angle_cutoff"),
        ({"selection": "resname LIG", "water_selection": "resname XYZ"}, "empty"),
    ],
)
def test_hydrogen_bonds_validate_their_options(solvated, options, message) -> None:
    with pytest.raises(ValueError, match=message):
        _hydrogen_bonds(solvated, **options)
