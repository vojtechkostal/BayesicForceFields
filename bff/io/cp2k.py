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
    coord_filename: str | None = None,
) -> dict[str, Any]:
    """Build the common CP2K input tree for one structure."""
    fn_pos = Path(fn_pos)
    elements = get_cp2k_elements(fn_pos)
    tree = deepcopy(CP2K_INPUT_TEMPLATE)

    tree["global"]["project_name"] = project
    tree["force_eval"]["subsys"]["topology"] = {
        "coord_file_name": coord_filename or fn_pos.name,
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
    steps: int | None = None,
    coord_filename: str | None = None,
    xyz_output: bool = False,
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
        coord_filename=coord_filename,
    )
    configure_cp2k_restart(tree, project=project, restart=restart)
    configure_cp2k_ensemble(tree, equilibration=equilibration)
    if steps is not None:
        tree["motion"]["md"]["steps"] = int(steps)
    if xyz_output:
        configure_cp2k_xyz_output(tree)
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
    coord_filename: str | None = None,
) -> None:
    """Generate a CP2K single-point input file."""
    tree = build_cp2k_tree(
        project=project,
        charge=charge,
        multiplicity=multiplicity,
        unitcell=unitcell,
        fn_pos=fn_pos,
        coord_filename=coord_filename,
    )
    configure_cp2k_single_point(tree)
    tree["global"]["print_level"] = "MEDIUM"

    with open(fn_out, "w") as handle:
        handle.write("\n".join(format_cp2k_input(tree)) + "\n")


def configure_cp2k_xyz_output(tree: dict[str, Any]) -> None:
    """Ask CP2K MD to write XYZ positions, forces, and energies every step."""
    print_section = tree["motion"].setdefault("print", {})
    for key in ("trajectory", "forces"):
        print_section.setdefault(key, {})["format"] = "XYZ"
        print_section[key]["each"] = {"md": 1}
    print_section["energy"] = {"each": {"md": 1}}


def make_cp2k_short_md_input(
    project: str,
    charge: int,
    multiplicity: int,
    unitcell: list[float],
    fn_pos: str | Path,
    fn_out: str | Path,
    *,
    steps: int = 100,
    coord_filename: str = "pos.xyz",
) -> None:
    """Generate a short CP2K MD input for one snapshot."""
    make_cp2k_input(
        project,
        charge,
        multiplicity,
        unitcell,
        fn_pos,
        equilibration=False,
        restart=False,
        fn_out=fn_out,
        steps=steps,
        coord_filename=coord_filename,
        xyz_output=True,
    )


def write_cp2k_md_slurm_script(
    fn_out: str | Path,
    *,
    uses_plumed: bool = False,
    project: str = "project",
) -> None:
    """Write a restart-aware Slurm script for CP2K MD."""
    plumed_block = ""
    if uses_plumed:
        plumed_block = 'echo "Using PLUMED input: plumed.dat"\n'

    script = f"""#!/bin/bash
#SBATCH --job-name=cp2k-md
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=1G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

CP2K_CMD="${{CP2K_CMD:-cp2k.psmp}}"
CP2K_PROJECT="${{CP2K_PROJECT:-{project}}}"
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
cd "${{SCRIPT_DIR}}"

if [[ -f setup-env.sh ]]; then
  # Optional user/site-specific environment setup.
  source setup-env.sh
fi
export OMP_NUM_THREADS="${{OMP_NUM_THREADS:-1}}"
{plumed_block}

if [[ ! -f .eq.done ]]; then
  restart_file="${{CP2K_PROJECT}}-1.restart"
  wfn_file="${{CP2K_PROJECT}}-RESTART.wfn"
  if [[ -f "${{restart_file}}" || -f "${{wfn_file}}" ]]; then
    eq_input="md-eq-restart.inp"
  else
    eq_input="md-eq-start.inp"
  fi

  echo "Running equilibration: ${{eq_input}}"
  srun "${{CP2K_CMD}}" -i "${{eq_input}}" -o "${{eq_input%.inp}}.out"
  touch .eq.done
else
  echo "Skipping equilibration; found .eq.done."
fi

if [[ ! -f .prod.done ]]; then
  echo "Running production: md-prod.inp"
  srun "${{CP2K_CMD}}" -i md-prod.inp -o md-prod.out
  touch .prod.done
else
  echo "Skipping production; found .prod.done."
fi
"""
    Path(fn_out).write_text(script)


def write_cp2k_snapshot_run_script(
    fn_out: str | Path,
) -> None:
    """Write the per-snapshot Slurm script for CP2K reference jobs."""
    script = """#!/bin/bash
#SBATCH --job-name=cp2k-snapshot
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=1G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

CP2K_CMD="${CP2K_CMD:-cp2k.psmp}"
RUN_DIR="$(pwd)"

if [[ -f setup-env.sh ]]; then
  # Optional user/site-specific environment setup.
  source setup-env.sh
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

for input in pos.xyz single-point.inp md.inp; do
  if [[ ! -f "${input}" ]]; then
    echo "Missing ${input} in ${RUN_DIR}" >&2
    exit 1
  fi
done

if [[ ! -f .single-point.done ]]; then
  echo "Running single-point calculation"
  srun "${CP2K_CMD}" -i single-point.inp -o single-point.out
  touch .single-point.done
else
  echo "Skipping single-point calculation; found .single-point.done."
fi

if [[ ! -f .md.done ]]; then
  echo "Running short MD"
  srun "${CP2K_CMD}" -i md.inp -o md.out
  touch .md.done
else
  echo "Skipping short MD; found .md.done."
fi
"""
    Path(fn_out).write_text(script)


def write_cp2k_snapshot_submit_script(
    fn_out: str | Path,
    *,
    snapshots_dirname: str = "xyz",
) -> None:
    """Write the submission helper for all prepared CP2K snapshots."""
    script = f"""#!/usr/bin/env bash
set -euo pipefail

main_dir="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
cd "${{main_dir}}"
snapshot_dir="${{1:-{snapshots_dirname}}}"
mkdir -p runs

for xyz in "${{snapshot_dir}}"/snapshot-*.xyz; do
  [[ -e "${{xyz}}" ]] || continue
  stem="$(basename "${{xyz}}" .xyz)"
  run_dir="${{main_dir}}/runs/${{stem}}"

  mkdir -p "${{run_dir}}"
  cp "${{main_dir}}/single-point.inp" "${{run_dir}}/single-point.inp"
  cp "${{main_dir}}/md.inp" "${{run_dir}}/md.inp"
  cp "${{main_dir}}/run.sh" "${{run_dir}}/run.sh"
  cp "${{xyz}}" "${{run_dir}}/pos.xyz"
  if [[ -f "${{main_dir}}/setup-env.sh" ]]; then
    cp "${{main_dir}}/setup-env.sh" "${{run_dir}}/setup-env.sh"
  fi
  echo "Submitting ${{stem}}"
  sbatch --chdir="${{run_dir}}" "${{run_dir}}/run.sh"
done
"""
    Path(fn_out).write_text(script)
