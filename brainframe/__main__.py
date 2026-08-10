"""Allow ``python -m brainframe`` to invoke the CLI."""

from __future__ import annotations

import sys

from brainframe.cli import main

if __name__ == "__main__":
    sys.exit(main())
