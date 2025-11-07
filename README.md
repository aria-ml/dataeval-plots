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

- **Multi-backend architecture**: Support for matplotlib, seaborn, plotly, and altair
- **Optional dependencies**: Install only the backends you need
- **Clean separation**: Core DataEval has zero plotting dependencies
- **Extensible**: Easy to add new backends via the PlottingBackend protocol
- **Lazy loading**: Backends are only imported when first used

## Architecture

```
dataeval/                           # Core (zero plotting code)
    outputs/
        _bias.py                    # No .plot() methods
    plotting.py                     # Convenience hook

dataeval-plots/                     # Separate package
    src/dataeval_plots/
        __init__.py                 # Main plot() function
        _registry.py                # Backend registry
        backends/
            _base.py                # Protocol definitions
            _matplotlib.py          # Matplotlib backend
            _seaborn.py             # Seaborn backend (future)
            _plotly.py              # Plotly backend (future)
            _altair.py              # Altair backend (future)
```

## Supported Output Types

- `CoverageOutput` - Image grid visualization
- `BalanceOutput` - Heatmap of balance metrics
- `DiversityOutput` - Diversity index visualization
- `SufficiencyOutput` - Learning curves with extrapolation
- `BaseStatsOutput` - Statistical histograms
- `DriftMVDCOutput` - Drift detection plots
