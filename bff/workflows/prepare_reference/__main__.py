import argparse

from .main import main

parser = argparse.ArgumentParser(
    description="Prepare CP2K reference inputs from a BFF build stage."
)
parser.add_argument("fn_config", help="Path to the config file [YAML].")
args = parser.parse_args()
main(args.fn_config)
