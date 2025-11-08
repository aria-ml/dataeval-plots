"""Tests for base installation (matplotlib only, no optional backends).

These tests verify that the package works correctly when installed without
optional dependencies (seaborn, plotly, altair). They should be run with
`nox -e base` to ensure the test environment matches the base installation.
"""

from __future__ import annotations

import sys

import pytest

from dataeval_plots import get_available_backends, get_backend, plot, set_default_backend


class TestBaseInstallation:
    """Test suite for base installation scenarios."""

    def test_matplotlib_available(self) -> None:
        """Test that matplotlib backend is available in base installation."""
        available = get_available_backends()
        assert "matplotlib" in available, "matplotlib should be available in base installation"

    def test_optional_backends_not_available(self) -> None:
        """Test that optional backends are not available in base installation."""
        available = get_available_backends()

        # Check that optional backends are not installed
        # Note: This test will only pass when run with `nox -e base`
        optional_backends = {"seaborn", "plotly", "altair"}

        # Only enforce this check if we're actually in a base-only environment
        # (detect by checking if any of these can be imported)
        has_optional = False
        for backend_name in optional_backends:
            try:
                if backend_name == "seaborn":
                    __import__("seaborn")
                elif backend_name == "plotly":
                    __import__("plotly")
                elif backend_name == "altair":
                    __import__("altair")
                has_optional = True
                break
            except ImportError:
                pass

        if not has_optional:
            # We're in a base-only environment, verify optional backends are not available
            for backend_name in optional_backends:
                assert backend_name not in available, f"{backend_name} should not be available in base installation"
        else:
            # We're in a full environment (e.g., running with nox -e test)
            # Skip this test
            pytest.skip("Skipping base-only test in full environment (use 'nox -e base' to run)")

    def test_get_matplotlib_backend(self) -> None:
        """Test that matplotlib backend can be retrieved."""
        backend = get_backend("matplotlib")
        assert backend is not None
        assert type(backend).__name__ == "MatplotlibBackend"

    def test_get_default_backend(self) -> None:
        """Test that default backend (matplotlib) works."""
        backend = get_backend()  # Should get matplotlib as default
        assert backend is not None
        assert type(backend).__name__ == "MatplotlibBackend"

    def test_set_default_backend_matplotlib(self) -> None:
        """Test setting default backend to matplotlib."""
        set_default_backend("matplotlib")
        backend = get_backend()
        assert type(backend).__name__ == "MatplotlibBackend"

    def test_optional_backend_import_error(self, mock_balance: object) -> None:
        """Test that using unavailable backend raises ImportError with helpful message."""
        available = get_available_backends()

        # Only test if we're in a base-only environment
        has_seaborn = False
        try:
            __import__("seaborn")
            has_seaborn = True
        except ImportError:
            pass

        if has_seaborn:
            pytest.skip("Skipping base-only test in full environment (use 'nox -e base' to run)")

        # seaborn should not be available in base installation
        assert "seaborn" not in available

        # Getting the backend should work (lazy loading)
        backend = get_backend("seaborn")
        assert backend is not None

        # But actually using it should raise ImportError (might be seaborn or one of its dependencies)
        with pytest.raises(ImportError) as exc_info:
            backend.plot(mock_balance)  # type: ignore[arg-type]

        error_msg = str(exc_info.value)
        # The error might be about seaborn or its dependencies (pandas, etc)
        assert "module" in error_msg.lower(), f"Expected ImportError about missing module, got: {error_msg}"

    def test_unknown_backend_error(self) -> None:
        """Test that requesting unknown backend raises ValueError with helpful message."""
        with pytest.raises(ValueError) as exc_info:
            get_backend("nonexistent_backend")

        error_msg = str(exc_info.value)
        assert "Unknown backend" in error_msg
        assert "nonexistent_backend" in error_msg
        assert "Known backends:" in error_msg
        assert "Available backends:" in error_msg

    def test_plot_with_matplotlib(self, mock_coverage: object) -> None:
        """Test that plot() works with matplotlib backend."""
        import numpy as np

        # Create dummy images for the coverage plot
        images = np.random.rand(25, 28, 28)  # 25 grayscale images

        # This should work in base installation
        fig = plot(mock_coverage, backend="matplotlib", images=images, top_k=6)  # type: ignore[arg-type]
        assert fig is not None

    def test_plot_with_default_backend(self, mock_coverage: object) -> None:
        """Test that plot() works with default backend (matplotlib)."""
        import numpy as np

        # Create dummy images for the coverage plot
        images = np.random.rand(25, 28, 28)  # 25 grayscale images

        # This should work in base installation
        fig = plot(mock_coverage, images=images, top_k=6)  # type: ignore[arg-type]
        assert fig is not None

    def test_available_backends_is_set(self) -> None:
        """Test that get_available_backends returns a set."""
        available = get_available_backends()
        assert isinstance(available, set)
        assert len(available) >= 1  # At least matplotlib should be available

    def test_available_backends_cached(self) -> None:
        """Test that get_available_backends is cached and returns a copy."""
        available1 = get_available_backends()
        available2 = get_available_backends()

        # Should be equal
        assert available1 == available2

        # But should be different objects (copies)
        assert available1 is not available2

        # Modifying one should not affect the other
        available1.add("fake_backend")
        assert "fake_backend" not in available2

    def test_matplotlib_in_sys_modules_after_import(self) -> None:
        """Test that matplotlib is in sys.modules after getting the backend."""
        # Get the backend (will lazy import matplotlib)
        get_backend("matplotlib")

        # matplotlib should now be in sys.modules
        assert "matplotlib" in sys.modules
