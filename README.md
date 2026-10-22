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

- `plot_type()`: Returns a string identifying the plot type (e.g., "balance")
- `meta()`: Returns execution metadata

This enables:

- **Loose coupling**: dataeval-plots doesn't import concrete classes from dataeval
- **Type safety**: Static and runtime type checking via `@runtime_checkable` protocols
- **Extensibility**: Anyone can create custom outputs implementing `Plottable`
- **Zero dependencies**: Core dataeval has no plotting dependencies

## Supported Output Types

| Output Type | Plot Type | Description | Source |
|-------------|-----------|-------------|--------|
| `BalanceOutput` | "balance" | Heatmap of class balance metrics | [dataeval/_bias.py](../dataeval/src/dataeval/outputs/_bias.py) |
| `DiversityOutput` | "diversity" | Visualization of diversity indices | [dataeval/_bias.py](../dataeval/src/dataeval/outputs/_bias.py) |
| `SufficiencyOutput` | "sufficiency" | Learning curves with extrapolation | [dataeval/_workflows.py](../dataeval/src/dataeval/outputs/_workflows.py) |
| `BaseStatsOutput` | "stats" | Statistical histograms and distributions | [dataeval/_stats.py](../dataeval/src/dataeval/outputs/_stats.py) |
| `DriftMVDCOutput` | "drift_mvdc" | Drift detection plots (MVDC analysis) | [dataeval/_drift.py](../dataeval/src/dataeval/outputs/_drift.py) |
| `FeasibilityOutput` | "feasibility" | Which accuracy requirements the evidence demonstrates or rules out | [dataeval/_feasibility.py](../dataeval/src/dataeval/performance/_feasibility.py) |

Each output type implements the `Plottable` protocol and can be plotted using any registered backend.

## Reading the Feasibility Plot

`plot()` draws the result of DataEval's experimental `Feasibility().evaluate(...)`. It answers one question per
quantity you asked about: for each accuracy or AP requirement, does the evidence demonstrate it, rule it out, or leave
it unknown?

```python
from dataeval.performance import Feasibility
from dataeval_plots import plot

result = Feasibility().evaluate(dataset, target={"accuracy": 0.95}, embeddings=embeddings)

fig = plot(result)  # one band per quantity
fig = plot(result, plot_evidence=True)  # one row per evidence source, in place of the bands
```

### What the plot shows

Each quantity gets its own band, with a "Required ..." axis (for example "Required accuracy" or "Required AP") and a
subtitle that says which models the band judges:

| Subtitle | Judged models |
|----------|---------------|
| Models may use the whole image | Models that may use the whole input |
| Models may use only the embeddings | Models restricted to the embeddings |
| Models give one answer per group | Models that give one answer per group |

The axis is split into three zones. Each zone classifies the requirements that fall inside it:

- **Demonstrated** runs from the axis minimum up to the lower confidence bound of the best tested model. The band
  reads "Requirements through *x*% are demonstrated".
- **Unknown** lies between the other two zones. The evidence neither demonstrates nor rules out these requirements.
- **Ruled out** runs from the upper confidence bound of the lowest applicable ceiling to the top of the axis. It holds
  only under the recorded assumptions, and the zone reads "ruled out under the recorded assumptions" wherever the
  label fits.

Each of your requirements is marked on the axis, and the marker states the determination in words:

| Marker | Meaning |
|--------|---------|
| "95% requirement: demonstrated ✓" | The requirement is at or below the lower confidence bound |
| "not enough evidence ?" | The requirement falls in the unknown zone, and the evidence cannot decide it |
| "ruled out under the recorded assumptions ✗" | The requirement is above the upper confidence bound of a ceiling |
| "not ruled out ?" | The requirement also falls in the unknown zone: it is not above any ceiling's upper confidence bound, and no tested model reaches it |

The caption under the band reports the observed score next to the lower confidence bound it supports, for example
"Your model scored 91.0% on the 2,000 evaluation images. Allowing for sampling uncertainty gives a lower
confidence bound of 88.5%." The sentence about the observed score appears only when an observed score exists. The
caption then states that the bounds concern new images drawn from the same population, and that confidence holds
jointly across the whole evaluation (for example "Confidence is 95% jointly across this evaluation").

Numbers read as DataEval prints them. Bounds are given to 0.1%, a lower bound rounded down and an upper bound rounded
up, so rounding never overstates what the evidence shows. Requirements and confidence keep their own decimals, such as
"95.5% requirement" or "Confidence is 97.5%".

When a ceiling bounds the band, the caption adds "Under the recorded assumptions, requirements above *x*% are ruled out
at the stated confidence." A group-based ceiling carries the extra note that its validity argument awaits lead-scientist
sign-off.

A ceiling that applies only to narrower models than the band judges never draws a zone. It appears as a note instead,
for example that the group-based ceiling does not limit models that distinguish individual images.

### Other outcomes

- **Ruled out.** A requirement above the lowest applicable ceiling's upper confidence bound is marked "ruled out
  under the recorded assumptions ✗". The ruled-out zone covers everything above that bound, and the caption reads
  "Under the recorded assumptions, requirements above *x*% are ruled out at the stated confidence."
- **Evidence conflict.** When bounds from different sources contradict each other, the band draws no zones. A dashed
  warning outline reads "Evidence conflicts: check assumptions", both clashing bounds are marked, and the caption
  says that the bounds disagree and to check the assumptions, including whether evaluation images entered training.
  Even when the assumptions hold, the probability that any bound fails is at most the stated alpha (for example 5% at
  95% confidence).
- **Bounds at 0% or 100%.** A lower confidence bound of 0% demonstrates nothing and an upper confidence bound of 100%
  rules nothing out, so neither draws a zone.

### Showing the evidence

`plot_evidence=True` replaces each band with one row per evidence source. Each row shows the observed score as a dot
and the confidence bound as a tick, with a legend line explaining both: the dot is the score observed on the
evaluation images, and the tick is the lower (or upper) confidence bound for new images from the same population.
Evidence that does not apply to the judged models is drawn faint and labeled with the models it applies to. A source
that produced no bound for the quantity is listed faint as "no bound available", and when no source has a bound the
view says "No bound is available yet." instead of showing empty axes.

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
        return "balance"  # Reuse existing balance plotting

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
from dataeval_plots.protocols import PlottableBalance
from dataeval_plots import register_backend

class CustomBackend(BasePlottingBackend):
    """Custom plotting backend using your preferred library."""

    def _plot_balance(self, output: PlottableBalance, **kwargs):
        # Implement balance plotting
        return my_figure

    # Implement other _plot_* methods...

# Register and use
register_backend("custom", CustomBackend())
fig = plot(result, backend="custom")
```

The `BasePlottingBackend` class handles all routing logic automatically - you just implement the plot-type-specific methods (`_plot_coverage`, `_plot_balance`, etc.).
