"""Nox sessions for testing, linting, and type checking."""

import os
from pathlib import Path

import nox

nox.options.default_venv_backend = "uv"
nox.options.sessions = ["test", "type", "lint"]

PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13"]


def _install_group(session: nox.Session, group: str) -> None:
    """Install a dependency group using uv."""
    session.run_install("uv", "sync", "--group", group, env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location})


@nox.session(python=PYTHON_VERSIONS)
def test(session: nox.Session) -> None:
    """Run unit tests with coverage."""
    _install_group(session, "test")

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    session.run(
        "pytest",
        "--cov=src/dataeval/plots",
        "--cov-report=term-missing",
        "--cov-report=xml:output/coverage.xml",
        "--cov-report=html:output/htmlcov",
        f"--junitxml=output/junit-{session.python}.xml",
        "tests/",
    )


@nox.session(python=PYTHON_VERSIONS)
def type(session: nox.Session) -> None:
    """Run type checking with pyright."""
    _install_group(session, "type")
    session.run("pyright", "src/dataeval/plots")


@nox.session
def lint(session: nox.Session) -> None:
    """Run linting and spell checking."""
    _install_group(session, "lint")

    ci = os.getenv("CI")

    if ci:
        # In CI, only check formatting (don't auto-fix)
        session.run("ruff", "check", "src/", "tests/")
        session.run("ruff", "format", "--check", "src/", "tests/")
    else:
        # Locally, auto-fix issues
        session.run("ruff", "check", "--fix", "src/", "tests/")
        session.run("ruff", "format", "src/", "tests/")

    session.run("codespell", "src/", "tests/")
