# Changelog

All notable changes to this project will be documented in this file.

## [0.0.2] - 2025-01-07

- Install matplotlib by default - removed `matplotlib` extra
- Enable auto-registration of backends on initial load

## [0.0.1] - 2025-01-07

### Added

#### Core Features

- Multi-backend plotting architecture supporting matplotlib, seaborn, plotly, and altair
- Main `plot()` function with automatic backend routing based on output type
- Protocol-based design using `Plottable` protocol for loose coupling with dataeval core
- Backend registry system with lazy loading for optimal dependency management

#### Plotting Backends

- **matplotlib**: Default backend with publication-quality static plots
- **seaborn**: Statistical visualization backend
- **plotly**: Interactive web-based plotting backend
- **altair**: Declarative visualization grammar backend

#### Supported Plot Types

- Coverage plots: Image grids showing uncovered samples (`CoverageOutput`)
- Balance plots: Heatmaps of class balance metrics (`BalanceOutput`)
- Diversity plots: Visualization of diversity indices (`DiversityOutput`)
- Sufficiency plots: Learning curves with extrapolation (`SufficiencyOutput`)
- Base statistics plots: Histograms and distributions (`BaseStatsOutput`)
- Drift detection plots: MVDC analysis visualization (`DriftMVDCOutput`)
