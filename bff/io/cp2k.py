import numpy as np
import MDAnalysis as mda
from ..data import CATIONS, CP2K_KIND_DEFAULTS, CP2K_INPUT_TEMPLATE


def format_cp2k_input(data: dict, indent: int = 0) -> list:
    """ Recursively format CP2K input from a dictionary. """
    lines = []
    for key, value in data.items():
        key_upper = key.upper()
        key_end_upper = key.split()[0].upper()
        prefix = ' ' * indent

        if isinstance(value, dict):
            lines.append(f"{prefix}&{key_upper}")
            lines.extend(format_cp2k_input(value, indent + 2))
            lines.append(f"{prefix}&END {key_end_upper}")
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    lines.extend(format_cp2k_input({key: item}, indent))
                elif isinstance(item, list):
                    formatted_item = ' '.join(map(str, item))
                    lines.append(f"{prefix}{key_upper} {formatted_item}")
                else:
                    lines.append(f"{prefix}{key_upper} {item}")
        elif isinstance(value, bool):
            bool_value = '.TRUE.' if value else '.FALSE.'
            lines.append(f"{prefix}{key_upper} {bool_value}")
        else:
            lines.append(f"{prefix}{key_upper} {value}")
    return lines


def make_cp2k_input(
    project: str,
    charge: int, multiplicity: int, unitcell: list[float], fn_pos: str,
    restraints: list[dict],
    equlibration: bool, restart: bool,
    fn_out: str
) -> None:
    """
    Generates a CP2K input file.

    Parameters
    ----------
    project : str
        Base name of the project.
    charge : int
        Total charge of the system.
    multiplicity : int
        Spin multiplicity of the system 2S+1 (S = 0.5 * N),
        where N is number of unpaired electrons.
    unitcell : list of floats
        Unit cell dimensions (a, b, c) in angstroms.
    fn_pos : str
        Path to the input coordinate file.
    restraints : list of dict, optional
        List of restraints, where each dict contains 'atoms', 'target', and 'k'
        representing the atomic pair which distance is restined, position
        of the restraint and the force constant in kJ/mol/nm^2.
    equilibration : bool
        If True, Langevin dynamics is enabled
        for faster temperature equilibration.
    restart : bool
        Whether to restart from a previous simulation.
    fn_out : str
        Output file name for the CP2K input.
    Returns
    -------
    None
    """
    u = mda.Universe(fn_pos)
    elements, idx = np.unique(u.atoms.elements, return_index=True)
    elements = elements[np.argsort(idx)]
    contains_cations = any(e.lower() in CATIONS for e in elements)
    kinds = {
        f'kind {e.capitalize()}': CP2K_KIND_DEFAULTS[e.lower()]
        for e in elements
    }
    topology = {'coord_file_name': fn_pos.name, 'coord_file_format': 'XYZ'}
    cell = {
        'ABC': f'[angstrom] {unitcell[0]} {unitcell[1]} {unitcell[2]}',
        'periodic': 'XYZ'
    }

    tree = CP2K_INPUT_TEMPLATE.copy()

    tree['global']['project_name'] = project

    tree['force_eval']['subsys']['topology'] = topology
    tree['force_eval']['subsys']['cell'] = cell
    tree['force_eval']['subsys'].update(**kinds)

    tree['force_eval']['dft']['charge'] = charge
    tree['force_eval']['dft']['multiplicity'] = multiplicity
    tree['force_eval']['dft']['uks'] = multiplicity != 1

    if contains_cations:
        exclusions = [
            [i + 1, j + 1]
            for i, e in enumerate(elements)
            if e.lower() in CATIONS
            for j in range(len(elements))
        ]
        tree['force_eval']['dft']['xc']['vdw_potential']['pair_potential']['d3_exclude_kind_pair'] = exclusions

    if restart:
        tree['ext_restart'] = {'restart_file_name': f'{project}-1.restart'}
        tree['force_eval']['dft']['scf']['scf_guess'] = 'RESTART'
        tree['force_eval']['dft']['wfn_restart_file_name'] = \
            f'{project}-RESTART.wfn'
    else:
        tree['force_eval']['dft']['scf']['scf_guess'] = 'ATOMIC'

    if restraints:
        tree['force_eval']['subsys']['colvar'] = [
            {'distance': {'atoms': r['atoms']}} for r in restraints
        ]
        tree['motion']['constraint'] = {'collective': [
            {
                'colvar': i + 1,
                'intermolecular': True,
                'target': f"[nm] {r['target']}",
                'restraint': {'k': f"[kjmol/nm^2] {r['k']}"}
            }
            for i, r in enumerate(restraints)
        ]}

    tree['motion']['md']['ensemble'] = 'LANGEVIN' if equlibration else 'NVT'
    if equlibration:
        tree['motion']['md']['langevin'] = {'gamma': 0.02}
    else:
        tree['motion']['md']['thermostat'] = {
            'region': 'GLOBAL',
            'type': 'CSVR',
            'csvr': {'timecon': '[fs] 100'}
        }

    with open(fn_out, 'w') as f:
        f.write('\n'.join(format_cp2k_input(tree)) + '\n')
