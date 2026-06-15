"""Module entry point so ``python -m injectkit`` works like the console script.

Running ``python -m injectkit ...`` is the canonical way to invoke a package
without relying on the installed console script being on ``PATH`` — handy in CI,
virtualenvs, and Docker. It simply delegates to the same :func:`injectkit.cli.main`
that the ``injectkit`` console script (declared in ``[project.scripts]``) calls,
so both invocation styles behave identically and share the same exit codes.
"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
