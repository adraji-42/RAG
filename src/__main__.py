"""Command-line interface entrypoint for the RAG package."""

import sys

import fire

from .cli import Cli


def main() -> None:
    """Run the command-line interface entrypoint."""
    try:
        fire.Fire(Cli)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)


main()
