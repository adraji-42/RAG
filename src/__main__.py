import sys
import fire

from .cli import Cli


def main() -> None:
    try:
        fire.Fire(Cli)
    except KeyboardInterrupt:
        sys.exit(130)


main()
