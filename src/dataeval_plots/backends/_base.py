"""Protocol for plotting backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from dataeval.outputs import Output


class PlottingBackend(Protocol):
    """Protocol that all plotting backends must implement."""

    def plot(self, output: Output, **kwargs: Any) -> Any:
        """
        Plot output using this backend.

        Parameters
        ----------
        output : Output
            DataEval output to visualize
        **kwargs
            Backend-specific parameters

        Returns
        -------
        Figure
            Backend-specific figure object
        """
        ...
