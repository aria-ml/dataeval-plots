"""Nox sessions for testing, linting, and type checking."""

import os
import re
from sys import version_info

import nox
import nox_uv

PYTHON_VERSION = f"{version_info[0]}.{version_info[1]}"
PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13"]
PYTHON_RE_PATTERN = re.compile(r"\d\.\d{1,2}")
IS_CI = bool(os.environ.get("CI"))

nox.options.default_venv_backend = "uv"
nox.options.sessions = ["test", "type", "lint"]


def get_python_version(session: nox.Session) -> str:
    matches = PYTHON_RE_PATTERN.search(session.name)
    return matches.group(0) if matches else PYTHON_VERSION


@nox_uv.session(uv_groups=["test"])
def test(session: nox.Session) -> None:
    """Run unit tests with coverage reporting. Specify version using `nox -P {version} -e test`."""
    python_version = get_python_version(session)
    cov_args = ["--cov", f"--junitxml=output/junit.{python_version}.xml"]
    cov_term_args = ["--cov-report", "term"]
    cov_xml_args = ["--cov-report", f"xml:output/coverage.{python_version}.xml"]
    cov_html_args = ["--cov-report", f"html:output/htmlcov.{python_version}"]

    session.run_install("uv", "sync", "--no-dev", "--all-extras", "--group=test")
    session.run(
        "pytest",
        *cov_args,
        *cov_term_args,
        *cov_xml_args,
        *cov_html_args,
        *session.posargs,
    )
    session.run("mv", ".coverage", f"output/.coverage.{python_version}", external=True)


@nox_uv.session(uv_groups=["test"])
def base(session: nox.Session) -> None:
    """Run tests for base installation (matplotlib only, no optional backends)."""
    python_version = get_python_version(session)
    cov_args = ["--cov", "--cov-fail-under=0", f"--junitxml=output/junit.base.{python_version}.xml"]
    cov_term_args = ["--cov-report", "term"]
    cov_xml_args = ["--cov-report", f"xml:output/coverage.base.{python_version}.xml"]
    cov_html_args = ["--cov-report", f"html:output/htmlcov.base.{python_version}"]

    # Install only base dependencies (no extras)
    session.run_install("uv", "sync", "--no-dev", "--group=test")
    session.run(
        "pytest",
        "tests/test_base_installation.py",
        *cov_args,
        *cov_term_args,
        *cov_xml_args,
        *cov_html_args,
        *session.posargs,
    )
    session.run("mv", ".coverage", f"output/.coverage.base.{python_version}", external=True)


@nox_uv.session(uv_groups=["type"])
def type(session: nox.Session) -> None:  # noqa: A001
    """Run type checks and verify external types. Specify version using `nox -P {version} -e type`."""
    session.run_install("uv", "sync", "--no-dev", "--all-extras", "--group=type")
    session.run("pyright", "--stats", "src/")
    session.run("pyright", "--ignoreexternal", "--verifytypes", "dataeval_plots")


@nox_uv.session(uv_groups=["lint"])
def lint(session: nox.Session) -> None:
    """Perform linting and spellcheck."""
    session.run_install("uv", "sync", "--only-group=lint")
    session.run("ruff", "check", "--show-fixes", "--exit-non-zero-on-fix", "--fix")
    session.run("ruff", "format", "--check" if IS_CI else ".")
    session.run("codespell")
