from copy import deepcopy
from pathlib import Path
from typing import Any

import MDAnalysis as mda
import numpy as np

from ..data import CATIONS, CP2K_INPUT_TEMPLATE, CP2K_KIND_DEFAULTS


def format_cp2k_input(data: dict[str, Any], indent: int = 0) -> list[str]:
    """Format a CP2K input tree as text.

    Parameters
    ----------
    data
        Nested mapping representing the CP2K input tree.
    indent
        Current indentation level.

    Returns
    -------
    list of str
        Formatted CP2K input lines.
    """
    lines: list[str] = []
    prefix = " " * indent

    for key, value in data.items():
        section_name = key.upper()
        section_end = key.split()[0].upper()

        if isinstance(value, dict):
            lines.append(f"{prefix}&{section_name}")
            lines.extend(format_cp2k_input(value, indent + 2))
            lines.append(f"{prefix}&END {section_end}")
            continue

        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    lines.extend(format_cp2k_input({key: item}, indent))
                elif isinstance(item, list):
                    lines.append(f"{prefix}{section_name} {' '.join(map(str, item))}")
                else:
                    lines.append(f"{prefix}{section_name} {item}")
            continue

        if isinstance(value, bool):
            value = ".TRUE." if value else ".FALSE."

        lines.append(f"{prefix}{section_name} {value}")

    return lines


def get_cp2k_elements(fn_pos: str | Path) -> np.ndarray:
    """Return element symbols in coordinate-file order without duplicates."""
    universe = mda.Universe(fn_pos)
    elements, index = np.unique(universe.atoms.elements, return_index=True)
    return elements[np.argsort(index)]


def build_cp2k_kinds(elements: np.ndarray) -> dict[str, dict[str, str]]:
    """Build CP2K KIND sections for the elements present in a system."""
    return {
        f"kind {element.capitalize()}": CP2K_KIND_DEFAULTS[element.lower()]
        for element in elements
    }


def build_cp2k_d3_exclusions(elements: np.ndarray) -> list[list[int]]:
    """Build D3 kind-pair exclusions for cation-containing systems."""
    return [
        [i + 1, j + 1]
        for i, element in enumerate(elements)
        if element.lower() in CATIONS
        for j in range(len(elements))
    ]


def configure_cp2k_restart(
    tree: dict[str, Any],
    *,
    project: str,
    restart: bool,
) -> None:
    """Configure CP2K SCF and restart sections."""
    if restart:
        tree["ext_restart"] = {"restart_file_name": f"{project}-1.restart"}
        tree["force_eval"]["dft"]["scf"]["scf_guess"] = "RESTART"
        tree["force_eval"]["dft"]["wfn_restart_file_name"] = (
            f"{project}-RESTART.wfn"
        )
        return

    tree["force_eval"]["dft"]["scf"]["scf_guess"] = "ATOMIC"


def configure_cp2k_ensemble(
    tree: dict[str, Any],
    *,
    equilibration: bool,
) -> None:
    """Configure the MD ensemble for equilibration or production."""
    tree["motion"]["md"]["ensemble"] = "LANGEVIN" if equilibration else "NVT"
    if equilibration:
        tree["motion"]["md"]["langevin"] = {"gamma": 0.02}
        tree["motion"]["md"].pop("thermostat", None)
        return

    tree["motion"]["md"]["thermostat"] = {
        "region": "GLOBAL",
        "type": "CSVR",
        "csvr": {"timecon": "[fs] 100"},
    }
    tree["motion"]["md"].pop("langevin", None)


def configure_cp2k_plumed(
    tree: dict[str, Any],
    *,
    plumed_input_file: str | Path,
) -> None:
    """Configure CP2K to use a PLUMED input file for biased MD.

    Parameters
    ----------
    tree
        CP2K input tree to update.
    plumed_input_file
        PLUMED input filename referenced from the CP2K input.
    """
    tree["motion"]["free_energy"] = {
        "metadyn": {
            "use_plumed": True,
            "plumed_input_file": Path(plumed_input_file).name,
        }
    }


def configure_cp2k_single_point(tree: dict[str, Any]) -> None:
    """Configure a CP2K input tree for single-point energy/forces."""
    tree["global"]["run_type"] = "ENERGY_FORCE"
    tree.pop("motion", None)


def build_cp2k_tree(
    *,
    project: str,
    charge: int,
    multiplicity: int,
    unitcell: list[float],
    fn_pos: str | Path,
) -> dict[str, Any]:
    """Build the common CP2K input tree for one structure."""
    fn_pos = Path(fn_pos)
    elements = get_cp2k_elements(fn_pos)
    tree = deepcopy(CP2K_INPUT_TEMPLATE)

    tree["global"]["project_name"] = project
    tree["force_eval"]["subsys"]["topology"] = {
        "coord_file_name": fn_pos.name,
        "coord_file_format": "XYZ",
    }
    tree["force_eval"]["subsys"]["cell"] = {
        "ABC": f"[angstrom] {unitcell[0]} {unitcell[1]} {unitcell[2]}",
        "periodic": "XYZ",
    }
    tree["force_eval"]["subsys"].update(build_cp2k_kinds(elements))

    tree["force_eval"]["dft"]["charge"] = charge
    tree["force_eval"]["dft"]["multiplicity"] = multiplicity
    tree["force_eval"]["dft"]["uks"] = multiplicity != 1

    if any(element.lower() in CATIONS for element in elements):
        tree["force_eval"]["dft"]["xc"]["vdw_potential"]["pair_potential"][
            "d3_exclude_kind_pair"
        ] = build_cp2k_d3_exclusions(elements)

    return tree


def make_cp2k_input(
    project: str,
    charge: int,
    multiplicity: int,
    unitcell: list[float],
    fn_pos: str | Path,
    equilibration: bool,
    restart: bool,
    fn_out: str | Path,
    plumed_input_file: str | Path | None = None,
) -> None:
    """Generate a CP2K input file for one simulation window.

    Parameters
    ----------
    project
        Base project name.
    charge
        Total charge of the system.
    multiplicity
        Spin multiplicity.
    unitcell
        Unit-cell dimensions in Angstrom.
    fn_pos
        Path to the XYZ coordinate file.
    equilibration
        Whether to generate an equilibration input.
    restart
        Whether the CP2K run restarts from a previous step.
    fn_out
        Output CP2K input file.
    plumed_input_file
        Optional PLUMED input file used for biased CP2K MD.
    """
    tree = build_cp2k_tree(
        project=project,
        charge=charge,
        multiplicity=multiplicity,
        unitcell=unitcell,
        fn_pos=fn_pos,
    )
    configure_cp2k_restart(tree, project=project, restart=restart)
    configure_cp2k_ensemble(tree, equilibration=equilibration)
    if plumed_input_file is not None:
        configure_cp2k_plumed(tree, plumed_input_file=plumed_input_file)

    with open(fn_out, "w") as handle:
        handle.write("\n".join(format_cp2k_input(tree)) + "\n")


def make_cp2k_single_point_input(
    project: str,
    charge: int,
    multiplicity: int,
    unitcell: list[float],
    fn_pos: str | Path,
    fn_out: str | Path,
) -> None:
    """Generate a CP2K single-point input file."""
    tree = build_cp2k_tree(
        project=project,
        charge=charge,
        multiplicity=multiplicity,
        unitcell=unitcell,
        fn_pos=fn_pos,
    )
    configure_cp2k_single_point(tree)
    tree["global"]["print_level"] = "MEDIUM"

    with open(fn_out, "w") as handle:
        handle.write("\n".join(format_cp2k_input(tree)) + "\n")


def write_cp2k_md_slurm_script(
    fn_out: str | Path,
    *,
    uses_plumed: bool = False,
) -> None:
    """Write a restart-aware Slurm script for CP2K MD."""
    plumed_block = ""
    if uses_plumed:
        plumed_block = """
if [ -f plumed.dat ]; then
  echo "Using PLUMED input: plumed.dat"
fi
"""

    script = f"""#!/bin/bash
#SBATCH --job-name=cp2k-md
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

CP2K_CMD="${{CP2K_CMD:-cp2k.psmp}}"
{plumed_block}

if [ ! -f .md-eq-start.done ]; then
  ${{CP2K_CMD}} -i md-eq-start.inp -o md-eq-start.out
  touch .md-eq-start.done
fi

if [ ! -f .md-eq-restart.done ]; then
  ${{CP2K_CMD}} -i md-eq-restart.inp -o md-eq-restart.out
  touch .md-eq-restart.done
fi

${{CP2K_CMD}} -i md-prod.inp -o md-prod.out
"""
    Path(fn_out).write_text(script)


def write_cp2k_single_point_slurm_script(
    fn_out: str | Path,
    *,
    snapshots_dirname: str = "snapshots",
) -> None:
    """Write a Slurm script for batched CP2K single-point evaluations."""
    script = f"""#!/bin/bash
#SBATCH --job-name=cp2k-sp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

CP2K_CMD="${{CP2K_CMD:-cp2k.psmp}}"
SNAPSHOT_DIR="{snapshots_dirname}"
INPUT_TEMPLATE="single-point.inp"

for xyz in "${{SNAPSHOT_DIR}}"/snapshot-*.xyz; do
  [ -e "${{xyz}}" ] || continue
  stem=$(basename "${{xyz}}" .xyz)
  run_dir="runs/${{stem}}"
  mkdir -p "${{run_dir}}"
  cp "${{INPUT_TEMPLATE}}" "${{run_dir}}/single-point.inp"
  cp "${{xyz}}" "${{run_dir}}/snapshot.xyz"
  (
    cd "${{run_dir}}"
    ${{CP2K_CMD}} -i single-point.inp -o single-point.out
  )
done
"""
    Path(fn_out).write_text(script)
