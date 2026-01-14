# Changelog

All notable changes to this project will be documented in this file.

## [0.0.5] - 2026-01-14

- [feat] Update for new DataEval output classes
- [deps] Remove upper pin for numpy>2.3

## [0.0.4] - 2025-11-10

- [impr] Refactor duplicate code and improve bounding boxes
- [housekeeping] Rename PlottableBaseStats
- [feat] Add bounding box rendering
- [fix] Switch to typing_extensions for Required/NotRequired
- [fix] Remove mock_coverage test
- [depr] Remove Coverage plotting and enhance image plotting
- [feat] Add image grid plotting functionality for datasets
- [fix] Update protocols for better compatibility
- [misc] Add plot overloads for parameter hinting and minor visual tweaks
- [misc] Refactor image_to_hwc helper

## [0.0.3] - 2025-11-08

- [impr] Refactor shared code to shared module
- [fix] Address rendering issues for multiple backends
- [fix] Fix the ExecutionMetadata protocol

## [0.0.2] - 2025-11-07

- [feat] Update backend detection and install matplotlib by default

## [0.0.1] - 2025-11-07

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
