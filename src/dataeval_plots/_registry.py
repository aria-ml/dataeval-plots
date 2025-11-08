"""Backend registry and selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dataeval_plots.backends._base import PlottingBackend

_BACKENDS: dict[str, PlottingBackend] = {}
_DEFAULT_BACKEND: str = "matplotlib"
_AVAILABLE_BACKENDS: set[str] | None = None

# Mapping of backend names to their module and class
_BACKEND_MAP = {
    "matplotlib": ("dataeval_plots.backends._matplotlib", "MatplotlibBackend"),
    "seaborn": ("dataeval_plots.backends._seaborn", "SeabornBackend"),
    "plotly": ("dataeval_plots.backends._plotly", "PlotlyBackend"),
    "altair": ("dataeval_plots.backends._altair", "AltairBackend"),
}


def _discover_available_backends() -> set[str]:
    """
    Discover which backends are available based on installed dependencies.

    Returns
    -------
    set[str]
        Set of available backend names
    """
    # Map backend names to their actual library dependencies
    dependency_map = {
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "plotly": "plotly",
        "altair": "altair",
    }

    available = set()
    for name, lib in dependency_map.items():
        try:
            __import__(lib)
            available.add(name)
        except ImportError:
            pass
    return available


def get_available_backends() -> set[str]:
    """
    Get set of available backends based on installed dependencies.

    Returns
    -------
    set[str]
        Set of available backend names

    Examples
    --------
    >>> from dataeval_plots import get_available_backends
    >>> available = get_available_backends()
    >>> print(available)
    {'matplotlib', 'seaborn'}
    """
    global _AVAILABLE_BACKENDS
    if _AVAILABLE_BACKENDS is None:
        _AVAILABLE_BACKENDS = _discover_available_backends()
    return _AVAILABLE_BACKENDS.copy()


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

    # Check if backend is known
    if backend_name not in _BACKEND_MAP:
        available = get_available_backends()
        known_backends = ", ".join(sorted(_BACKEND_MAP.keys()))
        raise ValueError(
            f"Unknown backend: '{backend_name}'. "
            f"Known backends: {known_backends}. "
            f"Available backends: {', '.join(sorted(available)) if available else 'none'}"
        )

    # Lazy import if not already registered
    if backend_name not in _BACKENDS:
        module_path, class_name = _BACKEND_MAP[backend_name]
        try:
            module = __import__(module_path, fromlist=[class_name])
            backend_class = getattr(module, class_name)
            register_backend(backend_name, backend_class())
        except ImportError as e:
            available = get_available_backends()
            available_str = ", ".join(sorted(available)) if available else "none"

            # Determine installation command based on backend
            if backend_name == "matplotlib":
                install_cmd = "pip install dataeval-plots"
            else:
                install_cmd = f"pip install dataeval-plots[{backend_name}]"

            raise ImportError(
                f"Backend '{backend_name}' requires additional dependencies. "
                f"Install with: {install_cmd}\n"
                f"Available backends: {available_str}"
            ) from e

    return _BACKENDS[backend_name]
