import re
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .extxyz import last_xyz_frame, read_extxyz_frame, write_extxyz_frames

CATIONS = {'li', 'na', 'k', 'rb', 'ca', 'mg'}

CP2K_KIND_DEFAULTS = {
    'h': {'basis_set': 'TZV2P-GTH-q1', 'potential': 'GTH-PBE-q1'},
    'o': {'basis_set': 'TZV2P-GTH-q6', 'potential': 'GTH-PBE-q6'},
    'c': {'basis_set': 'TZV2P-GTH-q4', 'potential': 'GTH-PBE-q4'},
    'p': {'basis_set': 'TZV2P-GTH-q5', 'potential': 'GTH-PBE-q5'},
    'n': {'basis_set': 'TZV2P-GTH-q5', 'potential': 'GTH-PBE-q5'},
    'f': {'basis_set': 'TZV2P-GTH-q7', 'potential': 'GTH-PBE-q7'},
    'na': {'basis_set': 'TZV2P-GTH-q9', 'potential': 'GTH-PBE-q9'},
    'k': {'basis_set': 'TZV2P-MOLOPT-SR-GTH-q9', 'potential': 'GTH-PBE-q9'},
    'ca': {'basis_set': 'TZV2P-MOLOPT-PBE-GTH-q10', 'potential': 'GTH-PBE-q10'},
    'cl': {'basis_set': 'TZV2P-GTH-q7', 'potential': 'GTH-PBE-q7'},
    's': {'basis_set': 'TZV2P-GTH-q6', 'potential': 'GTH-PBE-q6'},
}

CP2K_SINGLE_ATOM_DIRECTORY_NAMES = {
    'h': 'hydrogen',
    'c': 'carbon',
    'n': 'nitrogen',
    'o': 'oxygen',
    'f': 'fluorine',
    'na': 'sodium',
    'mg': 'magnesium',
    'p': 'phosphorus',
    's': 'sulfur',
    'cl': 'chlorine',
    'k': 'potassium',
    'ca': 'calcium',
    'li': 'lithium',
    'rb': 'rubidium',
    'br': 'bromine',
    'i': 'iodine',
}

CP2K_SINGLE_ATOM_MULTIPLICITIES = {
    'h': 2,
    'c': 3,
    'n': 4,
    'o': 3,
    'f': 2,
    'na': 2,
    'mg': 1,
    'p': 4,
    's': 3,
    'cl': 2,
    'k': 2,
    'ca': 1,
    'li': 2,
    'rb': 2,
    'br': 2,
    'i': 2,
}

ATOMIC_NUMBERS = {
    'h': 1,
    'li': 3,
    'c': 6,
    'n': 7,
    'o': 8,
    'f': 9,
    'na': 11,
    'mg': 12,
    'p': 15,
    's': 16,
    'cl': 17,
    'k': 19,
    'ca': 20,
    'br': 35,
    'rb': 37,
    'i': 53,
}

CP2K_VERSION_RE = re.compile(
    r"\b(?:CP2K(?:\s+version)?|version)\s+(\d{4}|\d+)(?:[._](\d+))?",
    re.IGNORECASE,
)
CP2K_FALLBACK_VERSION_RE = re.compile(r"\b(\d{4}|\d+)\.(\d+)\b")


@lru_cache(maxsize=None)
def detect_cp2k_version(cp2k_cmd: str) -> tuple[int, int] | None:
    """Detect the CP2K major/minor version from the executable output."""
    for flag in ("--version", "-v"):
        try:
            result = subprocess.run(
                [cp2k_cmd, flag],
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            raise

        text = "\n".join(part for part in (result.stdout, result.stderr) if part)
        match = CP2K_VERSION_RE.search(text) or CP2K_FALLBACK_VERSION_RE.search(text)
        if match is not None:
            return int(match.group(1)), int(match.group(2) or 0)
    return None


def cp2k_supports_gfn_type(cp2k_cmd: str) -> bool | None:
    """Return whether the selected CP2K build supports the XTB GFN_TYPE keyword."""
    version = detect_cp2k_version(cp2k_cmd)
    if version is None:
        return None
    return version[0] >= 2025


def strip_cp2k_gfn_type(fn_input: str | Path) -> bool:
    """Remove the XTB GFN_TYPE line from one CP2K input file if present."""
    fn_input = Path(fn_input)
    text = fn_input.read_text(encoding='utf-8')
    updated, n_removed = re.subn(
        r"(?im)^[ \t]*GFN_TYPE\b.*(?:\n|$)",
        "",
        text,
    )
    if n_removed == 0:
        return False
    fn_input.write_text(updated, encoding='utf-8')
    return True


def format_cp2k_input(data: dict[str, Any], indent: int = 0) -> list[str]:
    """Format a nested CP2K input tree as text."""
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
    import MDAnalysis as mda

    universe = mda.Universe(fn_pos)
    elements, index = np.unique(universe.atoms.elements, return_index=True)
    return elements[np.argsort(index)]


def get_cp2k_kind_defaults(element: str) -> dict[str, str]:
    """Return CP2K KIND defaults for one element or raise a clear error."""
    element_key = element.lower()
    if element_key not in CP2K_KIND_DEFAULTS:
        raise ValueError(
            f"No CP2K KIND defaults defined for element '{element}'. "
            "Extend CP2K_KIND_DEFAULTS in bff.io.cp2k before preparing CP2K inputs."
        )
    return CP2K_KIND_DEFAULTS[element_key]


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


def build_cp2k_tree(
    *,
    project: str,
    charge: int,
    multiplicity: int,
    unitcell: list[float],
    fn_pos: str | Path,
    coord_filename: str | None = None,
) -> dict[str, Any]:
    """Build the default DFT CP2K input tree for one structure."""
    fn_pos = Path(fn_pos)
    elements = get_cp2k_elements(fn_pos)
    subsys = {
        'topology': {
            'coord_file_name': coord_filename or fn_pos.name,
            'coord_file_format': 'XYZ',
        },
        'cell': {
            'ABC': f'[angstrom] {unitcell[0]} {unitcell[1]} {unitcell[2]}',
            'periodic': 'XYZ',
        },
    }
    for element in elements:
        subsys[f'kind {element.capitalize()}'] = get_cp2k_kind_defaults(element)

    force_eval = {
        'method': 'Quickstep',
        'subsys': subsys,
        'dft': {
            'basis_set_file_name': ['GTH_BASIS_SETS', 'BASIS_MOLOPT_UZH'],
            'potential_file_name': 'GTH_POTENTIALS',
            'scf': {
                'max_scf': 20,
                'eps_scf': 5e-07,
                'outer_scf': {'max_scf': 20, 'eps_scf': 5e-07},
                'ot': {
                    'preconditioner': 'FULL_ALL',
                    'energy_gap': -1.0,
                    'minimizer': 'DIIS',
                },
            },
            'mgrid': {'cutoff': '[Ry] 400.0'},
            'qs': {
                'method': 'GPW',
                'eps_default': 1e-12,
                'extrapolation_order': 4,
                'extrapolation': 'ASPC',
            },
            'xc': {
                'xc_functional': {'pbe': {'parametrization': 'REVPBE'}},
                'xc_grid': {'xc_deriv': 'NN50_SMOOTH'},
                'vdw_potential': {
                    'potential_type': 'PAIR_POTENTIAL',
                    'pair_potential': {
                        'type': 'DFTD3',
                        'r_cutoff': '[angstrom] 16.0',
                        'reference_functional': 'revPBE',
                        'parameter_file_name': 'dftd3.dat',
                    },
                },
            },
            'charge': charge,
            'multiplicity': multiplicity,
            'uks': multiplicity != 1,
        },
    }
    if any(element.lower() in CATIONS for element in elements):
        pair_potential = force_eval['dft']['xc']['vdw_potential']['pair_potential']
        pair_potential['d3_exclude_kind_pair'] = [
            [i + 1, j + 1]
            for i, element in enumerate(elements)
            if element.lower() in CATIONS
            for j in range(len(elements))
        ]

    return {
        'global': {
            'project_name': project,
            'print_level': 'LOW',
            'run_type': 'MD',
            'walltime': '03:50:00',
        },
        'motion': {
            'md': {
                'temperature': 300.0,
                'timestep': '[fs] 0.5',
                'steps': 500,
            },
            'print': {
                'forces': {
                    'format': 'DCD',
                    'each': {'md': 1},
                },
                'trajectory': {
                    'format': 'DCD',
                    'each': {'md': 1},
                },
                'restart': {
                    'backup_copies': 1,
                    'each': {'md': 1},
                },
            },
        },
        'force_eval': force_eval,
    }


def make_cp2k_input(
    project: str,
    charge: int,
    multiplicity: int,
    unitcell: list[float],
    fn_pos: str | Path,
    fn_out: str | Path,
    *,
    kind: str = 'md',
    restart: bool = False,
    equilibration: bool = False,
    steps: int | None = None,
    plumed_input_file: str | Path | None = None,
    coord_filename: str | None = None,
) -> None:
    """Generate one CP2K input file.

    `kind` can be `md`, `single_point`, `isolated_atom`, or `xtb_md`.
    """
    tree = build_cp2k_tree(
        project=project,
        charge=charge,
        multiplicity=multiplicity,
        unitcell=unitcell,
        fn_pos=fn_pos,
        coord_filename=coord_filename,
    )

    if kind == 'single_point':
        tree['global']['run_type'] = 'ENERGY_FORCE'
        tree['global']['print_level'] = 'MEDIUM'
        tree.pop('motion', None)
        tree['force_eval']['print'] = {'forces on': {'filename': '__STD_OUT__'}}
    elif kind == 'isolated_atom':
        tree['global']['run_type'] = 'ENERGY'
        tree['global']['print_level'] = 'LOW'
        tree['global']['preferred_fft_library'] = 'FFTW'
        tree['global'].pop('walltime', None)
        tree.pop('motion', None)

        dft = tree['force_eval']['dft']
        dft['poisson'] = {'periodic': 'NONE', 'psolver': 'MT'}
        dft['scf']['scf_guess'] = 'ATOMIC'
        dft['scf']['max_scf'] = 200
        dft['scf']['eps_scf'] = 1.0e-6
        dft['scf']['ot'] = {
            'preconditioner': 'FULL_SINGLE_INVERSE',
            'minimizer': 'DIIS',
            'energy_gap': 0.05,
        }
        dft['scf'].pop('outer_scf', None)
        dft['qs'].pop('extrapolation_order', None)
        dft['qs'].pop('extrapolation', None)
        dft['xc'].pop('vdw_potential', None)
        tree['force_eval']['subsys']['cell'] = {
            'ABC': f'[angstrom] {unitcell[0]} {unitcell[1]} {unitcell[2]}',
            'periodic': 'NONE',
        }
    elif kind == 'xtb_md':
        dft = tree['force_eval']['dft']
        topology = dict(tree['force_eval']['subsys']['topology'])
        cell = dict(tree['force_eval']['subsys']['cell'])
        tree['force_eval'] = {
            'method': 'QS',
            'subsys': {
                'topology': topology,
                'cell': cell,
            },
            'dft': {
                'scf': {
                    'max_scf': 50,
                    'eps_scf': 1.0e-6,
                    'outer_scf': {'max_scf': 200, 'eps_scf': 1.0e-6},
                    'ot': {
                        'preconditioner': 'FULL_SINGLE_INVERSE',
                        'minimizer': 'DIIS',
                    },
                    'scf_guess': 'ATOMIC',
                },
                'qs': {
                    'method': 'XTB',
                    'xtb': {
                        'gfn_type': 1,
                        'check_atomic_charges': False,
                        'do_ewald': True,
                        'use_halogen_correction': True,
                    },
                },
                'charge': dft['charge'],
                'multiplicity': dft['multiplicity'],
                'uks': dft['uks'],
            },
        }
        md = tree['motion']['md']
        md['ensemble'] = 'NVT'
        md['thermostat'] = {
            'region': 'GLOBAL',
            'type': 'CSVR',
            'csvr': {'timecon': '[fs] 100'},
        }
        md.pop('langevin', None)
        md['steps'] = int(steps or md['steps'])
        tree['motion']['print'] = {
            'trajectory': {
                'format': 'XYZ',
                'each': {'md': md['steps']},
            }
        }
    elif kind == 'md':
        dft = tree['force_eval']['dft']
        md = tree['motion']['md']

        if restart:
            tree['ext_restart'] = {'restart_file_name': f'{project}-1.restart'}
            dft['scf']['scf_guess'] = 'RESTART'
            dft['wfn_restart_file_name'] = f'{project}-RESTART.wfn'
        else:
            dft['scf']['scf_guess'] = 'ATOMIC'

        md['ensemble'] = 'LANGEVIN' if equilibration else 'NVT'
        if equilibration:
            md['langevin'] = {'gamma': 0.02}
            md.pop('thermostat', None)
        else:
            md['thermostat'] = {
                'region': 'GLOBAL',
                'type': 'CSVR',
                'csvr': {'timecon': '[fs] 100'},
            }
            md.pop('langevin', None)

        if steps is not None:
            md['steps'] = int(steps)
        if plumed_input_file is not None:
            tree['motion']['free_energy'] = {
                'metadyn': {
                    'use_plumed': True,
                    'plumed_input_file': Path(plumed_input_file).name,
                }
            }
    else:
        raise ValueError(f'Unsupported CP2K input kind: {kind}')

    Path(fn_out).write_text("\n".join(format_cp2k_input(tree)) + "\n")


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
    coord_filename: str = 'pos.xyz',
) -> None:
    """Generate a vacuum single-atom CP2K input with neutral multiplicity."""
    symbol = element.capitalize()
    fn_out = Path(fn_out)
    fn_pos = fn_out.with_name(coord_filename)

    write_cp2k_single_atom_xyz(symbol, fn_pos, box_length=box_length)
    make_cp2k_input(
        project=f'{symbol}_vacuum',
        charge=0,
        multiplicity=get_cp2k_single_atom_multiplicity(symbol),
        unitcell=[box_length, box_length, box_length],
        fn_pos=fn_pos,
        fn_out=fn_out,
        kind='isolated_atom',
        coord_filename=coord_filename,
    )


HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM = HARTREE_TO_EV / BOHR_TO_ANGSTROM
FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
ENERGY_RE = re.compile(
    rf"ENERGY\| Total FORCE_EVAL .*?({FLOAT_RE})\s*$",
    re.MULTILINE,
)
PIPE_FORCE_RE = re.compile(
    r"^\s*FORCES\|\s+\d+\s+"
    rf"({FLOAT_RE})\s+"
    rf"({FLOAT_RE})\s+"
    rf"({FLOAT_RE})\s+"
    rf"{FLOAT_RE}\s*$",
    re.MULTILINE,
)
LEGACY_FORCE_ROW_RE = re.compile(
    r"^\s*\d+\s+\d+\s+\S+\s+"
    rf"({FLOAT_RE})\s+"
    rf"({FLOAT_RE})\s+"
    rf"({FLOAT_RE})\s*$"
)


def read_cp2k_energy(
    fn_output: str | Path,
    *,
    context: str = 'energy',
) -> float:
    """Read a CP2K total energy from standard output and return it in eV."""
    fn_output = Path(fn_output)
    match = ENERGY_RE.search(fn_output.read_text(encoding='utf-8', errors='ignore'))
    if match is None:
        raise ValueError(f'Could not extract {context} from {fn_output}.')
    return float(match.group(1)) * HARTREE_TO_EV


def _convert_force(values: tuple[str, str, str]) -> list[float]:
    return [
        float(value) * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM
        for value in values
    ]


def _parse_pipe_forces(text: str) -> list[list[float]]:
    return [
        _convert_force(match.group(1, 2, 3))
        for match in PIPE_FORCE_RE.finditer(text)
    ]


def _parse_legacy_atomic_forces(text: str) -> list[list[float]]:
    tables: list[list[list[float]]] = []
    lines = text.splitlines()
    index = 0

    while index < len(lines):
        if not re.match(r"^\s*ATOMIC FORCES\b", lines[index]):
            index += 1
            continue

        table: list[list[float]] = []
        index += 1
        while index < len(lines):
            line = lines[index]
            if re.match(r"^\s*SUM OF ATOMIC FORCES\b", line):
                break

            match = LEGACY_FORCE_ROW_RE.match(line)
            if match is not None:
                table.append(_convert_force(match.group(1, 2, 3)))
            index += 1

        if table:
            tables.append(table)
        index += 1

    return tables[-1] if tables else []


def read_cp2k_forces(fn_output: str | Path) -> list[list[float]]:
    """Read atomic forces from CP2K output and return them in eV/Angstrom."""
    fn_output = Path(fn_output)
    text = fn_output.read_text(encoding='utf-8', errors='ignore')
    forces = _parse_pipe_forces(text)
    if not forces:
        forces = _parse_legacy_atomic_forces(text)
    if not forces:
        raise ValueError(f'Could not extract atomic forces from {fn_output}.')
    return forces


def write_cp2k_snapshot_extxyz(
    run_dir: str | Path,
    *,
    snapshot_filename: str = 'pos.xyz',
    trajectory_filename: str = 'md-pos-1.xyz',
    output_filename: str = 'sp.out',
    extxyz_filename: str = 'sp.extxyz',
) -> Path:
    """Write one reference extxyz frame from a staged CP2K snapshot job."""
    run_dir = Path(run_dir)
    snapshot_frame = read_extxyz_frame(run_dir / snapshot_filename)
    symbols, positions = last_xyz_frame(run_dir / trajectory_filename)
    forces = read_cp2k_forces(run_dir / output_filename)
    energy = read_cp2k_energy(run_dir / output_filename, context='energy')
    if len(symbols) != len(forces):
        raise ValueError(
            f'Force extraction failed for {run_dir}: expected {len(symbols)} rows, '
            f'got {len(forces)}.'
        )

    frame = {
        'atoms': symbols,
        'positions': positions,
        'forces': forces,
        'energy': energy,
        'lattice': snapshot_frame.get('lattice'),
        'pbc': snapshot_frame.get('pbc') or 'T T T',
        'source': 'sp',
    }
    fn_extxyz = run_dir / extxyz_filename
    write_extxyz_frames([frame], fn_extxyz)
    return fn_extxyz


def collect_single_atom_energies(
    single_atom_dirs: Sequence[str | Path],
) -> dict[int, float]:
    """Collect isolated-atom CP2K energies from staged single-atom jobs."""
    energies: dict[int, float] = {}
    for atom_dir in sorted(Path(path) for path in single_atom_dirs):
        fn_output = atom_dir / 'atom.out'
        if not fn_output.exists():
            raise FileNotFoundError(f'Missing isolated-atom output file: {fn_output}')
        lines = (atom_dir / 'pos.xyz').read_text(encoding='utf-8').splitlines()
        if len(lines) < 3:
            raise ValueError(f'Invalid isolated-atom XYZ file: {atom_dir / "pos.xyz"}')
        symbol = lines[2].split()[0].lower()
        if symbol not in ATOMIC_NUMBERS:
            raise ValueError(
                f'Unsupported isolated-atom element in {atom_dir / "pos.xyz"}: {symbol}'
            )
        energies[ATOMIC_NUMBERS[symbol]] = read_cp2k_energy(
            fn_output,
            context='isolated-atom energy',
        )
    return energies
