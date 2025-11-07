"""Backend registry and selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dataeval_plots.backends._base import PlottingBackend

_BACKENDS: dict[str, PlottingBackend] = {}
_DEFAULT_BACKEND: str = "matplotlib"


def register_backend(name: str, backend: PlottingBackend) -> None:
    """
    Register a plotting backend.

    Parameters
    ----------
    name : str
        Backend name (e.g., 'matplotlib', 'seaborn', 'plotly')
    backend : PlottingBackend
        Backend instance implementing the PlottingBackend protocol
    """
    _BACKENDS[name] = backend


def set_default_backend(name: str) -> None:
    """
    Set default plotting backend.

    Parameters
    ----------
    name : str
        Name of registered backend to use as default

    Raises
    ------
    ValueError
        If backend is not registered
    ImportError
        If backend dependencies are not installed
    """
    # Trigger lazy import if not already registered
    get_backend(name)

    global _DEFAULT_BACKEND
    _DEFAULT_BACKEND = name


def get_backend(name: str | None = None) -> PlottingBackend:
    """
    Get plotting backend by name.

    Performs lazy import of backends to avoid unnecessary dependencies.

    Parameters
    ----------
    name : str or None, default None
        Backend name. If None, uses default backend.

    Returns
    -------
    PlottingBackend
        Backend instance

    Raises
    ------
    ValueError
        If backend name is unknown
    ImportError
        If backend dependencies are not installed
    """
    backend_name = name or _DEFAULT_BACKEND

    if backend_name not in _BACKENDS:
        # Lazy import
        try:
            if backend_name == "matplotlib":
                from dataeval_plots.backends._matplotlib import MatplotlibBackend

                register_backend("matplotlib", MatplotlibBackend())
            elif backend_name == "seaborn":
                from dataeval_plots.backends._seaborn import SeabornBackend

                register_backend("seaborn", SeabornBackend())
            elif backend_name == "plotly":
                from dataeval_plots.backends._plotly import PlotlyBackend

                register_backend("plotly", PlotlyBackend())
            elif backend_name == "altair":
                from dataeval_plots.backends._altair import AltairBackend

                register_backend("altair", AltairBackend())
            else:
                raise ValueError(f"Unknown backend: {backend_name}")
        except ImportError as e:
            raise ImportError(
                f"Backend '{backend_name}' requires additional dependencies. "
                f"Install with: pip install dataeval-plots[{backend_name}]"
            ) from e

    return _BACKENDS[backend_name]
