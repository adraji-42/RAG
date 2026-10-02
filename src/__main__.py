import sys
import fire

from .cli import Cli


def main() -> None:
    try:
        fire.Fire(Cli)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)


main()
