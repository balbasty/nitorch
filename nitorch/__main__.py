"""Allow ``python -m nitorch ...`` (same as the ``nitorch`` console script)."""
import sys

from nitorch.cli import cli

if __name__ == '__main__':
    sys.exit(cli())
