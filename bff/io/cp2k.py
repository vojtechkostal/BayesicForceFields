from copy import deepcopy
from pathlib import Path
from typing import Any

import MDAnalysis as mda
import numpy as np

from ..data import (
    CATIONS,
    CP2K_INPUT_TEMPLATE,
    CP2K_KIND_DEFAULTS,
    CP2K_SINGLE_ATOM_DIRECTORY_NAMES,
    CP2K_SINGLE_ATOM_MULTIPLICITIES,
)


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


def get_cp2k_kind_defaults(element: str) -> dict[str, str]:
    """Return CP2K KIND defaults for one element or raise a clear error."""
    element_key = element.lower()
    if element_key not in CP2K_KIND_DEFAULTS:
        raise ValueError(
            f"No CP2K KIND defaults defined for element '{element}'. "
            "Extend bff.data.CP2K_KIND_DEFAULTS before preparing CP2K inputs."
        )
    return CP2K_KIND_DEFAULTS[element_key]


def build_cp2k_kinds(elements: np.ndarray) -> dict[str, dict[str, str]]:
    """Build CP2K KIND sections for the elements present in a system."""
    return {
        f"kind {element.capitalize()}": get_cp2k_kind_defaults(element)
        for element in elements
    }


def get_cp2k_single_atom_directory_name(element: str) -> str:
    """Return the directory name used for one isolated-atom reference job."""
    element_key = element.lower()
    if element_key not in CP2K_SINGLE_ATOM_DIRECTORY_NAMES:
        raise ValueError(
            f"No isolated-atom directory name defined for element '{element}'."
        )
    return CP2K_SINGLE_ATOM_DIRECTORY_NAMES[element_key]


def get_cp2k_single_atom_multiplicity(element: str) -> int:
    """Return the neutral isolated-atom spin multiplicity."""
    element_key = element.lower()
    if element_key not in CP2K_SINGLE_ATOM_MULTIPLICITIES:
        raise ValueError(
            f"No isolated-atom multiplicity defined for element '{element}'."
        )
    return CP2K_SINGLE_ATOM_MULTIPLICITIES[element_key]


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


def configure_cp2k_isolated_atom(
    tree: dict[str, Any],
    *,
    box_length: float,
) -> None:
    """Configure a CP2K input tree for an isolated-atom energy calculation."""
    tree["global"]["run_type"] = "ENERGY"
    tree["global"]["print_level"] = "LOW"
    tree["global"]["preferred_fft_library"] = "FFTW"
    tree["global"].pop("walltime", None)
    tree.pop("motion", None)

    dft = tree["force_eval"]["dft"]
    dft["poisson"] = {"periodic": "NONE", "psolver": "MT"}
    dft["scf"]["scf_guess"] = "ATOMIC"
    dft["scf"]["max_scf"] = 100
    dft["scf"]["eps_scf"] = 1.0e-6
    dft["scf"]["ot"] = {
        "preconditioner": "FULL_SINGLE_INVERSE",
        "minimizer": "DIIS",
        "energy_gap": 0.05,
    }
    dft["scf"]["outer_scf"] = {"max_scf": 20, "eps_scf": 1.0e-6}
    dft["qs"].pop("extrapolation_order", None)
    dft["qs"].pop("extrapolation", None)
    dft["xc"].pop("vdw_potential", None)

    tree["force_eval"]["subsys"]["cell"] = {
        "ABC": f"[angstrom] {box_length} {box_length} {box_length}",
        "periodic": "NONE",
    }


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


def write_cp2k_single_atom_xyz(
    element: str,
    fn_out: str | Path,
    *,
    box_length: float = 10.0,
) -> None:
    """Write a centered one-atom XYZ file for isolated vacuum calculations."""
    center = float(box_length) / 2.0
    symbol = element.capitalize()
    xyz = (
        "1\n"
        f"{symbol} isolated in vacuum\n"
        f"{symbol} {center:.6f} {center:.6f} {center:.6f}\n"
    )
    Path(fn_out).write_text(xyz)


def make_cp2k_isolated_atom_input(
    element: str,
    fn_out: str | Path,
    *,
    box_length: float = 10.0,
    coord_filename: str = "pos.xyz",
) -> None:
    """Generate a vacuum single-atom CP2K input with neutral multiplicity."""
    symbol = element.capitalize()
    multiplicity = get_cp2k_single_atom_multiplicity(symbol)
    fn_out = Path(fn_out)
    fn_pos = fn_out.with_name(coord_filename)

    write_cp2k_single_atom_xyz(symbol, fn_pos, box_length=box_length)
    tree = build_cp2k_tree(
        project=f"{symbol}_vacuum",
        charge=0,
        multiplicity=multiplicity,
        unitcell=[box_length, box_length, box_length],
        fn_pos=fn_pos,
        coord_filename=coord_filename,
    )
    configure_cp2k_isolated_atom(tree, box_length=box_length)

    with open(fn_out, "w") as handle:
        handle.write("\n".join(format_cp2k_input(tree)) + "\n")


def configure_cp2k_xyz_output(tree: dict[str, Any]) -> None:
    """Ask CP2K MD to write XYZ positions and forces every step."""
    print_section = tree["motion"].setdefault("print", {})
    for key in ("trajectory", "forces"):
        print_section.setdefault(key, {})["format"] = "XYZ"
        print_section[key]["each"] = {"md": 1}


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
#SBATCH --mem=10G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

CP2K_CMD="${{CP2K_CMD:-cp2k.psmp}}"
CP2K_PROJECT="${{CP2K_PROJECT:-{project}}}"

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
#SBATCH --mem=10G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

CP2K_CMD="${CP2K_CMD:-cp2k.psmp}"

if [[ -f setup-env.sh ]]; then
  source setup-env.sh
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "Running short MD in $(pwd)"
srun "${CP2K_CMD}" -i md.inp -o md.out

# The single-point input is staged for optional manual use:
# srun "${CP2K_CMD}" -i single-point.inp -o single-point.out
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
snapshot_dir="${{1:-{snapshots_dirname}}}"
mkdir -p runs

for xyz in "${{snapshot_dir}}"/snapshot-*.xyz; do
  [[ -e "${{xyz}}" ]] || continue
  stem="$(basename "${{xyz}}" .xyz)"
  run_dir="runs/${{stem}}"

  mkdir -p "${{run_dir}}"
  cp single-point.inp md.inp run.sh "${{run_dir}}"/
  cp "${{xyz}}" "${{run_dir}}/pos.xyz"
  if [[ -f setup-env.sh ]]; then
    cp setup-env.sh "${{run_dir}}"/
  fi
  echo "Submitting ${{stem}}"
  sbatch --chdir="${{run_dir}}" "${{run_dir}}/run.sh"
done
"""
    Path(fn_out).write_text(script)


def write_cp2k_single_atom_run_script(
    fn_out: str | Path,
) -> None:
    """Write the Slurm script for one isolated-atom CP2K job."""
    energy_command = (
        'grep "Total FORCE_EVAL" out.log | '
        """awk '{gsub(/\\.$/, "", $NF); """
        """printf "%.12f\\n", $NF * 27.211386245988}' > energy.dat"""
    )
    script = """#!/bin/bash
#SBATCH --job-name=cp2k-atom
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=10G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

CP2K_CMD="${CP2K_CMD:-cp2k.psmp}"

if [[ -f setup-env.sh ]]; then
  source setup-env.sh
elif [[ -f ../setup-env.sh ]]; then
  source ../setup-env.sh
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "Running isolated-atom single point in $(pwd)"
srun "${CP2K_CMD}" -i input.inp -o out.log
""" + energy_command + """
"""
    Path(fn_out).write_text(script)


def write_cp2k_single_atom_submit_script(
    fn_out: str | Path,
) -> None:
    """Write the submission helper for isolated-atom CP2K jobs."""
    script = """#!/usr/bin/env bash
set -euo pipefail

for run_dir in */; do
  [[ -d "${run_dir}" ]] || continue
  echo "Submitting ${run_dir%/}"
  sbatch --chdir="${run_dir}" run.sh
done
"""
    Path(fn_out).write_text(script)
