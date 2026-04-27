import argparse

from .main import main

parser = argparse.ArgumentParser(description='Run Gromacs simulations.')
parser.add_argument(
    '-f',
    '--fn_config',
    required=True,
    help='Path to the configuration file.',
)
args = parser.parse_args()
main(args.fn_config)
