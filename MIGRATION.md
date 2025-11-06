# Migration Guide: dataeval to dataeval.plots

This guide helps you migrate from the old plotting methods in DataEval output classes to the new `dataeval.plots` namespace package.

## Overview

All plotting functionality has been extracted from the core DataEval package into a separate `dataeval-plots` package. This allows:

1. **Smaller core package**: DataEval no longer requires matplotlib as a dependency
2. **Cleaner separation**: Plotting code is now separate from data analysis logic
3. **Easier maintenance**: Plotting features can evolve independently
4. **Optional installation**: Users who don't need plotting don't have to install matplotlib

## Installation

### Old Way
```bash
pip install dataeval[matplotlib]
```

### New Way
```bash
# Install core package
pip install dataeval

# Install plotting package (automatically installs matplotlib)
pip install dataeval-plots
```

## API Changes

The new API uses a **registration-based pattern** with `functools.singledispatch`, providing a clean and extensible interface.

### Pattern: `output.plot()` → `plot(output)`

#### Balance Output

**Old:**
```python
from dataeval.metrics import balance

output = balance(labels, metadata, factor_names)
fig = output.plot(plot_classwise=True)
```

**New:**
```python
from dataeval.metrics import balance
from dataeval.plots import plot

output = balance(labels, metadata, factor_names)
fig = plot(output, plot_classwise=True)
```

#### Diversity Output

**Old:**
```python
from dataeval.metrics import diversity

output = diversity(labels, metadata, factor_names)
fig = output.plot(plot_classwise=False)
```

**New:**
```python
from dataeval.metrics import diversity
from dataeval.plots import plot

output = diversity(labels, metadata, factor_names)
fig = plot(output, plot_classwise=False)
```

#### Coverage Output

**Old:**
```python
from dataeval.metrics import coverage

output = coverage(embeddings, images)
fig = output.plot(images, top_k=6)
```

**New:**
```python
from dataeval.metrics import coverage
from dataeval.plots import plot

output = coverage(embeddings, images)
fig = plot(output, images, top_k=6)
```

#### Drift MVDC Output

**Old:**
```python
from dataeval.detectors.drift import DriftMVDC

detector = DriftMVDC()
output = detector.fit_detect(reference, test)
fig = output.plot()
```

**New:**
```python
from dataeval.detectors.drift import DriftMVDC
from dataeval.plots import plot

detector = DriftMVDC()
output = detector.fit_detect(reference, test)
fig = plot(output)
```

#### Stats Outputs (PixelStats, ImageStats, etc.)

**Old:**
```python
from dataeval.metrics import pixelstats

output = pixelstats(images)
fig = output.plot(log=True, channel_limit=3)
```

**New:**
```python
from dataeval.metrics import pixelstats
from dataeval.plots import plot

output = pixelstats(images)
fig = plot(output, log=True, channel_limit=3)
```

#### Sufficiency Output

**Old:**
```python
from dataeval.workflows import Sufficiency

workflow = Sufficiency(...)
output = workflow.evaluate(...)
figs = output.plot(
    class_names=["cat", "dog"],
    show_error_bars=True,
    show_asymptote=True,
)
```

**New:**
```python
from dataeval.workflows import Sufficiency
from dataeval.plots import plot

workflow = Sufficiency(...)
output = workflow.evaluate(...)
figs = plot(
    output,
    class_names=["cat", "dog"],
    show_error_bars=True,
    show_asymptote=True,
)
```

## Benefits of the New API

### 1. Consistent Interface
All plotting now uses the same `plot(output, **kwargs)` pattern, making it easier to remember and use.

### 2. Type-Safe Dispatching
The singledispatch pattern provides type-safe routing to the correct plotting function:

```python
from dataeval.plots import plot

# Automatically routes to the correct implementation
balance_fig = plot(balance_output)
drift_fig = plot(drift_output)
stats_fig = plot(stats_output)
```

### 3. Extensibility
You can register custom plotting functions for your own output types:

```python
from dataeval.plots import plot
from dataeval.outputs import Output

@plot.register
def plot_custom_output(output: MyCustomOutput, **kwargs):
    # Your custom plotting logic
    ...
    return fig
```

### 4. Clear Dependencies
The separation makes it explicit when plotting functionality is needed:

```python
# Fails clearly if dataeval-plots not installed
from dataeval.plots import plot
```

## Compatibility Notes

### Breaking Changes
- All `.plot()` methods have been removed from output classes
- The `matplotlib` optional dependency in core dataeval is **deprecated** (but still works for now)
- Import paths have changed: `from dataeval.utils._plot import heatmap` → `from dataeval.plots.utils import heatmap`

### Deprecation Timeline
1. **Current release**: Both old and new APIs work (old methods removed, users must migrate)
2. **Next minor release**: Documentation updated to reflect new API only
3. **Future major release**: Remove matplotlib from dataeval's optional dependencies entirely

## Testing Your Migration

After migrating, ensure your plotting code works:

```python
import dataeval
from dataeval.plots import plot

# Run your analysis
output = some_detector.fit_detect(data)

# Plot results
fig = plot(output)
fig.savefig("output.png")

# Verify the figure was created correctly
assert fig is not None
```

## Getting Help

If you encounter issues during migration:

1. Check the [dataeval-plots documentation](https://dataeval.readthedocs.io/)
2. Review the [examples in the repository](https://github.com/aria-ml/dataeval/tree/main/examples)
3. Open an issue on [GitHub](https://github.com/aria-ml/dataeval/issues)

## Summary

The migration is straightforward:

1. Install `dataeval-plots` package
2. Change `output.plot()` to `plot(output)`
3. Import `plot` from `dataeval.plots`
4. Pass the same keyword arguments as before

The new API provides better separation of concerns, clearer dependencies, and a more extensible design for future enhancements.
