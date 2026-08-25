import MDAnalysis as mda
import numpy as np
import pytest

from bff.qoi import rdf


def _trajectory_universe() -> mda.Universe:
    universe = mda.Universe.empty(3, trajectory=True)
    universe.add_TopologyAttr("names", ["O1", "H1", "OW"])
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
def test_compute_rdf_accepts_atomgroups(smooth: bool) -> None:
    universe = _trajectory_universe()

    centers, values = rdf.compute_rdf(
        universe,
        universe.atoms[:2],
        universe.atoms[1:],
        distance_range=(0.0, 5.0),
        bins=20,
        pbc=True,
        smooth=smooth,
    )

    assert centers.shape == (20,)
    assert values.shape == (20,)
    assert np.all(np.isfinite(values))


def test_compute_rdf_rejects_frames_without_non_self_pairs() -> None:
    universe = _trajectory_universe()
    atom = universe.atoms[:1]

    with pytest.raises(ValueError, match="no non-self atom pairs"):
        rdf.compute_rdf(universe, atom, atom, pbc=False)


def test_compute_rdf_rejects_empty_frame_slice() -> None:
    universe = _trajectory_universe()

    with pytest.raises(ValueError, match="selects no trajectory frames"):
        rdf.compute_rdf(
            universe,
            universe.atoms[:1],
            universe.atoms[2:],
            start=2,
        )


def test_compute_rdf_qoi_passes_typed_centers_and_shared_neighbors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    universe = _trajectory_universe()
    received: list[tuple[mda.AtomGroup, mda.AtomGroup]] = []

    def capture_groups(
        universe: mda.Universe,
        atoms_a: mda.AtomGroup,
        atoms_b: mda.AtomGroup,
        **options,
    ) -> tuple[np.ndarray, np.ndarray]:
        received.append((atoms_a, atoms_b))
        bins = options["bins"]
        return np.arange(bins, dtype=float), np.ones(bins, dtype=float)

    monkeypatch.setattr(rdf, "compute_rdf", capture_groups)

    qoi = rdf.compute_rdf_qoi(
        universe,
        group_a="index 0 1",
        group_b="index 2",
        bins=10,
    )

    assert qoi.labels == ("H", "O")
    assert [set(group.types) for group, _ in received] == [{"H"}, {"O"}]
    assert received[0][1] is received[1][1]
    assert received[0][1].indices.tolist() == [2]


def test_compute_rdf_qoi_preserves_updating_typed_centers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    universe = mda.Universe.empty(4, trajectory=True)
    universe.add_TopologyAttr("names", ["O1", "O2", "H1", "OW"])
    universe.add_TopologyAttr("types", ["O", "O", "H", "OW"])
    universe.add_TopologyAttr("masses", [16.0, 16.0, 1.0, 16.0])
    universe.load_new(
        np.asarray(
            [
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
                [[2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            ]
        ),
        format="MEMORY",
    )
    center_indices: list[tuple[int, ...]] = []

    def capture_updates(
        universe: mda.Universe,
        atoms_a: mda.AtomGroup,
        atoms_b: mda.AtomGroup,
        **options,
    ) -> tuple[np.ndarray, np.ndarray]:
        for _ in universe.trajectory:
            center_indices.append(tuple(atoms_a.indices))
        bins = options["bins"]
        return np.arange(bins, dtype=float), np.ones(bins, dtype=float)

    monkeypatch.setattr(rdf, "compute_rdf", capture_updates)

    qoi = rdf.compute_rdf_qoi(
        universe,
        group_a="type O and prop x < 1",
        group_b="index 3",
        bins=10,
        pbc=False,
        update_selections=True,
    )

    assert qoi.labels == ("O",)
    assert center_indices == [(0,), (1,)]


def test_compute_rdf_qoi_excludes_massless_centers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    universe = _trajectory_universe()
    universe.atoms[1].mass = 0.0
    received_types: list[tuple[str, ...]] = []

    def capture_types(
        universe: mda.Universe,
        atoms_a: mda.AtomGroup,
        atoms_b: mda.AtomGroup,
        **options,
    ) -> tuple[np.ndarray, np.ndarray]:
        received_types.append(tuple(atoms_a.types))
        bins = options["bins"]
        return np.arange(bins, dtype=float), np.ones(bins, dtype=float)

    monkeypatch.setattr(rdf, "compute_rdf", capture_types)

    qoi = rdf.compute_rdf_qoi(
        universe,
        group_a="index 0 1",
        group_b="index 2",
        bins=10,
    )

    assert qoi.labels == ("O",)
    assert received_types == [("O",)]


def test_compute_rdf_qoi_requires_masses() -> None:
    universe = mda.Universe.empty(2, trajectory=True)
    universe.add_TopologyAttr("types", ["O", "OW"])

    with pytest.raises(ValueError, match="requires topology masses"):
        rdf.compute_rdf_qoi(
            universe,
            group_a="index 0",
            group_b="index 1",
        )
