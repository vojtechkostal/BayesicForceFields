"""GROMACS topology reading and parameter modification."""

import warnings
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path

import MDAnalysis as mda
import numpy as np
from gmxtopology import Topology
from MDAnalysis.guesser.tables import masses as MDA_MASSES

from .domain.specs import parameter_kind

MASSES = np.array(list(MDA_MASSES.values()))
ELEMENTS = list(MDA_MASSES.keys())


def guess_elements(universe: mda.Universe) -> None:
    """Assign elements from atom masses.

    Force-field atom names make name-based guesses unreliable. Atoms whose
    mass matches no element within 0.01 (deuterium, repartitioned hydrogens)
    keep their name-based element.
    """
    masses, inverse = np.unique(universe.atoms.masses, return_inverse=True)
    nearest = np.abs(masses[:, None] - MASSES[None, :]).argmin(axis=1)
    matched = np.isclose(MASSES[nearest], masses, atol=1e-2)
    by_mass = np.asarray(ELEMENTS, dtype=object)[nearest]
    elements = np.where(
        matched[inverse], by_mass[inverse], universe.atoms.elements.astype(object)
    )
    universe.add_TopologyAttr("elements", elements)


@contextmanager
def quiet_itp_parser():
    """Silence the ITP parser's warnings about guessed or missing elements."""
    with warnings.catch_warnings():
        for category in (DeprecationWarning, UserWarning):
            warnings.filterwarnings(
                "ignore", category=category, module=r"MDAnalysis\.topology\.ITPParser"
            )
        yield


def prepare_universe(
    fn_topol: str, fn_coord: str = None, dt: float = 1.0
) -> mda.Universe:
    """
    Prepares an MDAnalysis universe
    from Gromacs .itp topology and coordinates.

    Parameters
    ----------
    fn_topol : str
        Path to the topology file.
    fn_coord : str
        Path to the coordinate file.
    dt : float, optional
        Time step in ps. Default is 1.0.

    Returns
    -------
    universe : MDAnalysis.Universe
        A universe object with the topology and coordinates loaded,
        and guessed elements.
    """

    with quiet_itp_parser():
        if fn_coord is None:
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message=r"No coordinate reader found for .*\. Skipping this file\.",
                module=r"MDAnalysis\.core\.universe",
            )
            universe = mda.Universe(
                fn_topol, topology_format="ITP", to_guess=("elements", "masses"), dt=dt
            )
        else:
            universe = mda.Universe(
                fn_topol,
                fn_coord,
                topology_format="ITP",
                to_guess=("elements", "masses"),
                dt=dt,
            )

    guess_elements(universe)

    return universe


class TopologyModifier(Topology):
    """A GROMACS topology whose parameters can be set by label.

    Labels are those of ``specs.yaml``: ``charge <names or types>``,
    ``sigma <types>``, ``epsilon <types>``, ``dihedraltype9_<mult>_<phase>``,
    and ``define <directive>``. Atom selections (MDAnalysis syntax) are
    resolved on a universe that is built the first time one is needed.
    """

    def __init__(self, fn_topol: Path | str) -> None:
        self.source = Path(fn_topol).resolve()
        super().__init__(self.source)

    @cached_property
    def universe(self) -> mda.Universe:
        universe = prepare_universe(self.source)
        if len(self.atoms) != len(universe.atoms) or any(
            atom.name != mda_atom.name
            for atom, mda_atom in zip(self.atoms, universe.atoms)
        ):
            raise ValueError(
                f"Topology and MDAnalysis atom ordering disagree for {self.source}."
            )
        return universe

    def selected_groups(self, selection: str, scope: str) -> list[set[int]]:
        """Atom indices of ``selection``: one group, or one per residue."""
        atoms = self.universe.select_atoms(selection)
        if scope not in {"system", "residue"}:
            raise ValueError(f"Unsupported charge-constraint scope {scope!r}.")
        if not len(atoms):
            return []
        if scope == "system":
            return [{int(index) for index in atoms.indices}]
        groups: dict[int, set[int]] = {}
        for atom in atoms:
            groups.setdefault(int(atom.resindex), set()).add(int(atom.index))
        return list(groups.values())

    def charge_parameter_matches(self, parameter: str) -> dict[str, set[int]]:
        """Atoms of each token of a charge label: by atom name, else by type."""
        kind, *tokens = parameter.split()
        if kind != "charge" or not tokens:
            raise ValueError(f"Invalid charge parameter {parameter!r}.")
        if len(tokens) != len(set(tokens)):
            raise ValueError(f"Duplicate atom name or type in {parameter!r}.")
        matches: dict[str, set[int]] = {}
        for token in tokens:
            indices = {i for i, atom in enumerate(self.atoms) if atom.name == token}
            indices = indices or {
                i for i, atom in enumerate(self.atoms) if atom.type == token
            }
            if indices:
                matches[token] = indices
        return matches

    def apply_parameters(self, params: dict[str, float]) -> None:
        """Set every labelled parameter to its value."""
        for name, value in params.items():
            kind, tokens = parameter_kind(name), name.split()[1:]
            if kind == "charge":
                for index in set().union(*self.charge_parameter_matches(name).values()):
                    self.atoms[index].update(charge=value)
            elif kind in {"sigma", "epsilon"}:
                for atomtype in tokens:
                    self._update_atomtype(atomtype, **{kind: value})
            elif kind == "dihedraltype9":
                self._update_dihedraltype9(name, value)
            elif kind == "define":
                for define in self.defines:
                    if define.directive == tokens[0]:
                        define.update(argument=value)
                        break
            else:
                raise ValueError(f"Unsupported parameter name '{kind}'.")

    def _update_atomtype(self, atomtype: str, **values: float) -> None:
        for candidate in self.atomtypes:
            if candidate.name == atomtype:
                candidate.update(**values)
                return
        raise ValueError(f"Atom type {atomtype} not found in topology.")

    def _update_dihedraltype9(self, name: str, k: float) -> None:
        """Set the force constant of the terms ``dihedraltype9_<mult>_<phase>``."""
        parts = name.split("_")
        try:
            if len(parts) != 3 or parts[0] != "dihedraltype9":
                raise ValueError
            multiplicity, phase = int(parts[1]), float(parts[2])
        except ValueError as exc:
            raise ValueError(
                f"Invalid dihedral type 9 parameter {name!r}; expected "
                "'dihedraltype9_<multiplicity>_<phase>' with an integer "
                "multiplicity and a numeric phase."
            ) from exc
        terms = [
            dihedral
            for molecule in self.moleculetypes
            for dihedral in molecule.dihedrals
            if dihedral.func == 9
            and dihedral.params["mult"] == multiplicity
            and np.isclose(dihedral.params["phi_s"], phase)
        ]
        if not terms:
            raise ValueError(
                f"Dihedral type 9 with multiplicity {multiplicity} and phase "
                f"{phase:g} not found in topology."
            )
        for dihedral in terms:
            dihedral.update(kphi=k)
