import numpy as np

WATER_4SITE = np.array(
    [[0.02909557,  0.03692916, -0.04588176],
     [0.43909732, -0.54306981,  0.60411833],
     [-0.90090283, -0.04307076,  0.12411808],
     [-0.0309039,  -0.04307076,  0.05411814]]
)

WATER_3SITE = np.array(
    [[0.02909557,  0.03692916, -0.04588176],
     [0.43909732, -0.54306981,  0.60411833],
     [-0.90090283, -0.04307076,  0.12411808]]
)

IONS = ['li', 'na', 'k', 'ca', 'cal', 'mg', 'f', 'cl', 'br', 'i']
CATIONS = {'li', 'na', 'k', 'rb', 'ca', 'mg'}
ANIONS = {'f', 'cl', 'br', 'i'}

WATERS = ['SOL', 'water', 'HOH']


CP2K_KIND_DEFAULTS = {
    'h': {'basis_set': 'TZV2P-GTH-q1', 'potential': 'GTH-PBE-q1'},
    'o': {'basis_set': 'TZV2P-GTH-q6', 'potential': 'GTH-PBE-q6'},
    'c': {'basis_set': 'TZV2P-GTH-q4', 'potential': 'GTH-PBE-q4'},
    'p': {'basis_set': 'TZV2P-GTH-q5', 'potential': 'GTH-PBE-q5'},
    'n': {'basis_set': 'TZV2P-GTH-q5', 'potential': 'GTH-PBE-q5'},
    'na': {'basis_set': 'TZV2P-GTH-q9', 'potential': 'GTH-PBE-q9'},
    'ca': {'basis_set': 'TZV2P-MOLOPT-PBE-GTH-q10', 'potential': 'GTH-PBE-q10'},
    'cl': {'basis_set': 'TZV2P-GTH-q7', 'potential': 'GTH-PBE-q7'},
    's': {'basis_set': 'TZV2P-GTH-q6', 'potential': 'GTH-PBE-q6'},
}

CP2K_INPUT_TEMPLATE = {
    'global': {
        'project_name': 'project',
        'print_level': 'LOW',
        'run_type': 'MD',
        'walltime': '03:50:00'
    },
    'motion': {
        'md': {'temperature': 300.0, 'timestep': '[fs] 0.5', 'steps': 500},
        'print': {
            'forces': {
                'format': 'DCD',
                'each': {'md': 1}
            },
            'trajectory': {
                'format': 'DCD',
                'each': {'md': 1}
            },
            'restart': {
                'backup_copies': 1,
                'each': {'md': 1}
            }
        }
    },
    'force_eval': {
        'method': 'Quickstep',
        'subsys': {},
        'dft': {
            'basis_set_file_name': ['GTH_BASIS_SETS', 'BASIS_MOLOPT_UZH'],
            'potential_file_name': 'GTH_POTENTIALS',
            'scf': {
                'max_scf': 20, 'eps_scf': 5e-07,
                'outer_scf': {'max_scf': 20, 'eps_scf': 5e-07},
                'ot': {
                    'preconditioner': 'FULL_ALL',
                    'energy_gap': -1.0,
                    'minimizer': 'DIIS'}
            },
            'mgrid': {'cutoff': '[Ry] 400.0'},
            'qs': {
                'method': 'GPW',
                'eps_default': 1e-12,
                'extrapolation_order': 4,
                'extrapolation': 'ASPC'
            },
            'xc': {
                'xc_functional': {'pbe': {'parametrization': 'REVPBE'}},
                'xc_grid': {'xc_deriv': 'NN50_SMOOTH'},
                'vdw_potential': {
                    'potential_type': 'PAIR_POTENTIAL',
                    'pair_potential': {
                        'type': 'DFTD3', 'r_cutoff': '[angstrom] 16.0',
                        'reference_functional': 'revPBE',
                        'parameter_file_name': 'dftd3.dat'
                    }
                }
            }
        }
    }
}
