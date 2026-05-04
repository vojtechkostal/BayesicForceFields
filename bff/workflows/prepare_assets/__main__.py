import argparse

from .main import main

parser = argparse.ArgumentParser(
    description="Prepare FFMD and CP2K reference assets from a BFF build manifest."
)
parser.add_argument("fn_config", help="Path to the config file [YAML].")
args = parser.parse_args()
main(args.fn_config)
