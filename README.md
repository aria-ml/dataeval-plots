# DataEval Plots

Multi-backend plotting utilities for DataEval outputs.

## Installation

```bash
# Minimal - no plotting backend included
pip install dataeval-plots

# With matplotlib plotting (recommended)
pip install dataeval-plots[matplotlib]

# With multiple backends
pip install dataeval-plots[matplotlib,plotly]

# Everything
pip install dataeval-plots[all]
```

For development:
```bash
pip install -e dataeval-plots[all]
```

## Available Backends

| Backend | Status | Install With | Description |
|---------|--------|--------------|-------------|
| matplotlib | ✅ Default | `[matplotlib]` | Standard publication-quality plots |
| seaborn | ✅ Available | `[seaborn]` | Statistical data visualization |
| plotly | ✅ Available | `[plotly]` | Interactive web-based plots |
| altair | ✅ Available | `[altair]` | Declarative visualization grammar |

## Usage

### Option 1: Import from dataeval-plots directly

```python
from dataeval_plots import plot
from dataeval.metrics.bias import coverage

result = coverage(embeddings)
fig = plot(result, images=dataset, top_k=6)
fig.savefig("coverage.png")
```

### Option 2: Import from dataeval core (convenience)

```python
from dataeval import plotting
from dataeval.metrics.bias import coverage

result = coverage(embeddings)
fig = plotting.plot(result, images=dataset)
```

### Option 3: Set default backend

```python
from dataeval_plots import plot, set_default_backend

# Set seaborn as default
set_default_backend("seaborn")
fig = plot(result, images=dataset)  # Uses seaborn

# Override for a specific plot
fig = plot(result, backend="matplotlib", images=dataset)
```

## Features

- **Multi-backend architecture**: Support for matplotlib (default), seaborn, plotly, and altair
- **Optional dependencies**: Install only the backends you need
- **Clean separation**: Core dataeval has zero plotting dependencies
- **Protocol-based design**: Loose coupling via structural typing (`Plottable` protocol)
- **Extensible**: Easy to add new backends via `BasePlottingBackend` or custom outputs via `Plottable`
- **Lazy loading**: Backends are only imported when first used
- **Type safe**: Static type checking with mypy/pyright via `@runtime_checkable` protocols
- **DRY architecture**: Centralized routing logic in `BasePlottingBackend`

## Architecture

The package uses a **protocol-based architecture** for loose coupling between dataeval and dataeval-plots:

```
dataeval/                           # Core package
    outputs/
        _bias.py                    # CoverageOutput, BalanceOutput, DiversityOutput
        _stats.py                   # BaseStatsOutput
        _workflows.py               # SufficiencyOutput
        _drift.py                   # DriftMVDCOutput
    plotting.py                     # Convenience hook to dataeval-plots

dataeval-plots/                     # Separate plotting package
    src/dataeval_plots/
        __init__.py                 # Main plot() function
        _registry.py                # Backend registry with lazy loading
        protocols.py                # Protocol definitions (Plottable hierarchy)
        backends/
            _base.py                # BasePlottingBackend (abstract routing)
            _matplotlib.py          # MatplotlibBackend (default)
            _seaborn.py             # SeabornBackend
            _plotly.py              # PlotlyBackend
            _altair.py              # AltairBackend
```

### Protocol-Based Design

All DataEval output classes implement the `Plottable` protocol, which requires:

- `plot_type()`: Returns a string identifying the plot type (e.g., "coverage", "balance")
- `meta()`: Returns execution metadata

This enables:

- **Loose coupling**: dataeval-plots doesn't import concrete classes from dataeval
- **Type safety**: Static and runtime type checking via `@runtime_checkable` protocols
- **Extensibility**: Anyone can create custom outputs implementing `Plottable`
- **Zero dependencies**: Core dataeval has no plotting dependencies

## Supported Output Types

| Output Type | Plot Type | Description | Source |
|-------------|-----------|-------------|--------|
| `CoverageOutput` | "coverage" | Image grid showing uncovered samples | [dataeval/_bias.py](../dataeval/src/dataeval/outputs/_bias.py) |
| `BalanceOutput` | "balance" | Heatmap of class balance metrics | [dataeval/_bias.py](../dataeval/src/dataeval/outputs/_bias.py) |
| `DiversityOutput` | "diversity" | Visualization of diversity indices | [dataeval/_bias.py](../dataeval/src/dataeval/outputs/_bias.py) |
| `SufficiencyOutput` | "sufficiency" | Learning curves with extrapolation | [dataeval/_workflows.py](../dataeval/src/dataeval/outputs/_workflows.py) |
| `BaseStatsOutput` | "base_stats" | Statistical histograms and distributions | [dataeval/_stats.py](../dataeval/src/dataeval/outputs/_stats.py) |
| `DriftMVDCOutput` | "drift_mvdc" | Drift detection plots (MVDC analysis) | [dataeval/_drift.py](../dataeval/src/dataeval/outputs/_drift.py) |

Each output type implements the `Plottable` protocol and can be plotted using any registered backend.

## Extending the Package

### Creating Custom Outputs

You can create custom output classes that work with the plotting system by implementing the `Plottable` protocol:

```python
from dataclasses import dataclass
from numpy.typing import NDArray
from dataeval_plots.protocols import Plottable, ExecutionMetadata

@dataclass
class MyCustomOutput:
    """Custom output that reuses existing plot type."""
    uncovered_indices: NDArray

    def plot_type(self) -> str:
        return "coverage"  # Reuse existing coverage plotting

    def meta(self) -> ExecutionMetadata:
        return ExecutionMetadata.empty()

# Works seamlessly with existing backends
result = MyCustomOutput(uncovered_indices=my_data)
fig = plot(result, images=my_images)
```

### Creating Custom Backends

Extend `BasePlottingBackend` to create a new plotting backend:

```python
from dataeval_plots.backends._base import BasePlottingBackend
from dataeval_plots.protocols import PlottableCoverage, PlottableBalance
from dataeval_plots import register_backend

class CustomBackend(BasePlottingBackend):
    """Custom plotting backend using your preferred library."""

    def _plot_coverage(self, output: PlottableCoverage, **kwargs):
        # Implement coverage plotting
        # Access output.uncovered_indices, etc.
        return my_figure

    def _plot_balance(self, output: PlottableBalance, **kwargs):
        # Implement balance plotting
        return my_figure

    # Implement other _plot_* methods...

# Register and use
register_backend("custom", CustomBackend())
fig = plot(result, backend="custom")
```

The `BasePlottingBackend` class handles all routing logic automatically - you just implement the plot-type-specific methods (`_plot_coverage`, `_plot_balance`, etc.).
