import warnings
from pathlib import Path

import MDAnalysis as mda
import numpy as np
from gmxtopology import Topology
from MDAnalysis.guesser.tables import masses as MDA_MASSES

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

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            module=r"MDAnalysis\.topology\.ITPParser",
        )
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
    """Modify parameters across a complete Gromacs topology."""

    def __init__(self, fn_topol: Path | str) -> None:
        source = Path(fn_topol).resolve()
        super().__init__(source)
        self.source = source
        self.universe = prepare_universe(source)
        if len(self.atoms) != len(self.universe.atoms) or any(
            atom.name != mda_atom.name
            for atom, mda_atom in zip(self.atoms, self.universe.atoms)
        ):
            raise ValueError(
                f"Topology and MDAnalysis atom ordering disagree for {self.source}."
            )

    def select_indices(self, selection: str) -> set[int]:
        """Resolve one MDAnalysis selection to expanded topology atom indices."""
        return {int(atom.index) for atom in self.universe.select_atoms(selection)}

    def selected_groups(self, selection: str, scope: str) -> list[set[int]]:
        """Resolve a system selection into one system-level or per-residue group."""
        indices = self.select_indices(selection)
        if not indices or scope == "system":
            return [indices] if indices else []
        if scope != "residue":
            raise ValueError(f"Unsupported charge-constraint scope {scope!r}.")

        groups: dict[int, set[int]] = {}
        for index in indices:
            residue_index = int(self.universe.atoms[index].resindex)
            groups.setdefault(residue_index, set()).add(index)
        return list(groups.values())

    def charge_parameter_matches(self, parameter: str) -> dict[str, set[int]]:
        """Resolve charge-label tokens by atom name, falling back to atom type."""
        kind, *tokens = parameter.split()
        if kind != "charge" or not tokens:
            raise ValueError(f"Invalid charge parameter {parameter!r}.")
        if len(tokens) != len(set(tokens)):
            raise ValueError(f"Duplicate atom name or type in {parameter!r}.")

        matches: dict[str, set[int]] = {}
        for token in tokens:
            indices = {
                index for index, atom in enumerate(self.atoms) if atom.name == token
            }
            if not indices:
                indices = {
                    index
                    for index, atom in enumerate(self.atoms)
                    if atom.type.name == token
                }
            if indices:
                matches[token] = indices
        return matches

    def _update_charge(self, parameter: str, value: float) -> None:
        indices = set().union(*self.charge_parameter_matches(parameter).values())
        updated: set[int] = set()
        for index in indices:
            atom = self.atoms[index]
            if id(atom) not in updated:
                atom.update(charge=value)
                updated.add(id(atom))

    def _update_sigma(self, atomtype: str, value: float) -> None:
        for at in self.atomtypes:
            if at.name == atomtype:
                at.update(sigma=value)
                return
        raise ValueError(f"Atom type {atomtype} not found in topology.")

    def _update_epsilon(self, atomtype: str, value: float) -> None:
        for at in self.atomtypes:
            if at.name == atomtype:
                at.update(epsilon=value)
                return
        raise ValueError(f"Atom type {atomtype} not found in topology.")

    def _update_dihedraltype9(self, k: float, phase: float, multiplicity: int) -> None:
        updated = 0
        for mol in self.moleculetypes:
            for d in mol.dihedrals:
                if (
                    d.func == 9
                    and d.params["mult"] == multiplicity
                    and np.isclose(d.params["phi_s"], phase)
                ):
                    d.update(kphi=k)
                    updated += 1
        if not updated:
            raise ValueError(
                "Dihedral type 9 with multiplicity "
                f"{multiplicity} and phase {phase:g} not found in topology."
            )

    def _update_define(self, directive: str, argument: float | int | str) -> None:
        for define in self.defines:
            if define.directive == directive:
                define.update(argument=argument)
                return

    def apply_parameters(
        self,
        params: dict[str, float | list[float]],
    ) -> None:
        for p, value in params.items():
            if p.startswith("dihedraltype9"):
                parts = p.split("_")
                if len(parts) != 3 or parts[0] != "dihedraltype9":
                    raise ValueError(
                        f"Invalid dihedral type 9 parameter {p!r}; expected "
                        "'dihedraltype9_<multiplicity>_<phase>'."
                    )
                try:
                    multiplicity = int(parts[1])
                    phase = float(parts[2])
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid dihedral type 9 parameter {p!r}; multiplicity "
                        "must be an integer and phase must be a number."
                    ) from exc
                self._update_dihedraltype9(
                    k=value,
                    phase=phase,
                    multiplicity=multiplicity,
                )

            elif p.startswith("define"):
                p_name, directive = p.split(" ")
                self._update_define(directive, value)

            else:
                p_name, *atoms = p.split(" ")
                if p_name == "charge":
                    self._update_charge(p, value)
                elif p_name == "sigma":
                    for atom in atoms:
                        self._update_sigma(atom, value)
                elif p_name == "epsilon":
                    for atom in atoms:
                        self._update_epsilon(atom, value)
                else:
                    raise ValueError(f"Unsupported parameter name '{p_name}'.")
