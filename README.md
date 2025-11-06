# DataEval Plots

Matplotlib plotting utilities for DataEval outputs using PEP 420 namespace packages.

## Installation

```bash
pip install dataeval-plots
```

For development:
```bash
pip install -e dataeval-plots
```

## Usage

```python
from dataeval.plots import plot

# Use singledispatch-based plotting
output = detector.fit_detect(data)
fig = plot(output)
fig.savefig("output.png")
```

## Features

- Registration-based plotting using `functools.singledispatch`
- Supports all DataEval output types
- Clean namespace package integration (`dataeval.plots`)
- Separate installation from core DataEval package
