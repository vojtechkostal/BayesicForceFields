import argparse

from .main import main

parser = argparse.ArgumentParser(
    description="Build reusable FFMD and reference assets."
)
parser.add_argument("fn_config", help="Path to the config file [YAML].")
args = parser.parse_args()
main(args.fn_config)
