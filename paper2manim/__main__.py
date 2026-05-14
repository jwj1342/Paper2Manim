"""Allow ``python -m paper2manim`` so subprocess-based runners
(``scripts/run_experiment.py``, ``scripts/cross_domain.py``) don't depend on
the ``paper2manim`` console script being on PATH.
"""

from paper2manim.cli import cli

if __name__ == "__main__":
    cli()
