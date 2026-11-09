"""Tests for plotting a Feasibility output in the determinations and bounds shape."""

import json
from dataclasses import dataclass
from typing import Any, Literal

import conftest
import polars as pl
import pytest
from conftest import MockExecutionMetadata, MockPlottableFeasibility, _feasibility, _source, _target

from dataeval_plots import plot
from dataeval_plots.backends._shared import feasibility_zones, prepare_feasibility_data
from dataeval_plots.protocols import PlottableFeasibilityBounds, PlottableFeasibilityDeterminations

_METRIC = pl.Enum(["accuracy", "ap", "recall", "recall_at_fp", "mota"])
_QUANTITY = pl.Enum([*_METRIC.categories, "false_positive_rate"])
_SCOPE = pl.Enum(["full_input", "fixed_representation", "partition_restricted"])
_DETERMINATION = pl.Enum(
    ["demonstrated_feasible", "infeasible_under_recorded_assumptions", "not_ruled_out", "unresolved"]
)
_LIMITATION = pl.Enum(
    ["sampling_width", "no_replicas", "scope_restricted", "no_witness", "missing_bound", "conflict", "no_envelope"]
)
_AGGREGATION = pl.Enum(["frame_pooled", "sequence_mean"])
_AP_FUNCTIONAL = pl.Enum(["population_ap_integral", "population_ap_101"])
_ROLE = pl.Enum(["envelope", "witness"])
_BOUND_TYPE = pl.Enum(["lower", "upper"])
_STATE = pl.Enum(["unavailable", "uninformative", "informative"])
_UNIT = pl.Enum(["image", "sequence"])
_SOURCES = ("local_panels", "replicas", "knn", "linear_probe", "user_model")
# DataEval 1.2.0's evidence ids, as the new shape names the sources; any other id keeps its text
_NEW_SOURCE = {
    "user_witness/v1": "user_model",
    "builtin_witness/v1:knn": "knn",
    "builtin_witness/v1:probe": "linear_probe",
    "replica_ceiling/v1": "replicas",
}


@dataclass
class MockPlottableFeasibilityBounds:
    """Mock feasibility output in the determinations and bounds shape."""

    determinations: pl.DataFrame
    bounds: pl.DataFrame
    alpha: float = 0.05

    @property
    def plot_type(self) -> Literal["feasibility"]:
        return "feasibility"

    def meta(self) -> MockExecutionMetadata:
        return MockExecutionMetadata(arguments={}, state={"alpha": self.alpha})


def _new_source(evidence_id: str, kind: str = "") -> str:
    if kind.startswith("local_panel_ceiling/v1"):
        return "local_panels"
    return _NEW_SOURCE.get(evidence_id, evidence_id)


def as_bounds(old: MockPlottableFeasibility) -> MockPlottableFeasibilityBounds:
    """Re-express a determinations and evidence mock in the determinations and bounds shape."""
    determinations: list[dict[str, Any]] = []
    bounds: list[dict[str, Any]] = []
    for i, row in enumerate(old.determinations.iter_rows(named=True)):
        spec_key = row["target_spec"]
        spec = json.loads(spec_key)
        conflict = row["determination"] == "evidence_conflict"
        witness_source = row["witness_source"]
        determinations.append(
            {
                "requirement": i,
                "metric": row["metric"],
                "threshold": row["threshold"],
                "judged_scope": row["judged_scope"],
                "class_name": spec["class_name"],
                "aggregation": spec["aggregation"],
                "ap_functional": spec["ap_functional"],
                "max_false_positive_rate": spec["max_false_positive_rate"],
                "determination": "unresolved" if conflict else row["determination"],
                "limitation": "conflict" if conflict else None,
                "envelope": row["ceiling"],
                "envelope_scope": row["ceiling_scope"],
                "witness": row["witness"],
                "witness_source": None
                if witness_source is None
                else _new_source(witness_source, getattr(old.evidence.get(witness_source), "kind", "")),
            }
        )
        for eid, ev in old.evidence.items():
            for e in ev.endpoints:
                if e.target_spec != spec_key:
                    continue
                fp = e.quantity == "false_positive_rate"
                stats = ev.details["endpoints"].get(f"{spec_key}#false_positive_rate" if fp else spec_key) or {}
                bounds.append(
                    {
                        "requirement": i,
                        "source": _new_source(eid, ev.kind),
                        "role": "envelope" if e.direction == "upper" and not fp else "witness",
                        "quantity": "false_positive_rate" if fp else row["metric"],
                        "bound_type": e.direction,
                        "information_scope": ev.information_scope,
                        "unit": "sequence" if ev.unit == "group" else "image",
                        "state": e.state,
                        "bound": e.value,
                        "observed": stats.get("mean") if e.direction == "lower" or fp else None,
                        "n": stats.get("n"),
                    }
                )
    unknown = sorted({b["source"] for b in bounds} - set(_SOURCES))
    source = pl.Enum([*_SOURCES, *unknown])
    return MockPlottableFeasibilityBounds(
        determinations=pl.DataFrame(
            determinations,
            schema={
                "requirement": pl.Int64,
                "metric": _METRIC,
                "threshold": pl.Float64,
                "judged_scope": _SCOPE,
                "class_name": pl.String,
                "aggregation": _AGGREGATION,
                "ap_functional": _AP_FUNCTIONAL,
                "max_false_positive_rate": pl.Float64,
                "determination": _DETERMINATION,
                "limitation": _LIMITATION,
                "envelope": pl.Float64,
                "envelope_scope": _SCOPE,
                "witness": pl.Float64,
                "witness_source": source,
            },
        ),
        bounds=pl.DataFrame(
            bounds,
            schema={
                "requirement": pl.Int64,
                "source": source,
                "role": _ROLE,
                "quantity": _QUANTITY,
                "bound_type": _BOUND_TYPE,
                "information_scope": _SCOPE,
                "unit": _UNIT,
                "state": _STATE,
                "bound": pl.Float64,
                "observed": pl.Float64,
                "n": pl.Int64,
            },
        ),
        alpha=old.alpha,
    )


def test_each_mock_satisfies_only_its_own_protocol(mock_feasibility_how_to):
    new = as_bounds(mock_feasibility_how_to)
    assert isinstance(new, PlottableFeasibilityBounds)
    assert not isinstance(new, PlottableFeasibilityDeterminations)
    assert isinstance(mock_feasibility_how_to, PlottableFeasibilityDeterminations)
    assert not isinstance(mock_feasibility_how_to, PlottableFeasibilityBounds)


def test_determinations_columns_use_enum_dtypes(mock_feasibility_how_to):
    new = as_bounds(mock_feasibility_how_to)
    assert isinstance(new.determinations.schema["determination"], pl.Enum)
    assert isinstance(new.bounds.schema["source"], pl.Enum)
    assert new.determinations["determination"].to_list() == ["demonstrated_feasible", "unresolved"]


FIXTURES = [n for n in vars(conftest) if n.startswith("mock_feasibility_")]  # ten today


def _set_limitation(output: MockPlottableFeasibilityBounds, value: str) -> MockPlottableFeasibilityBounds:
    """Give every unresolved row the limitation ``value``."""
    limitation = pl.when(pl.col("determination") == "unresolved").then(pl.lit(value)).otherwise(None).cast(_LIMITATION)
    return MockPlottableFeasibilityBounds(
        output.determinations.with_columns(limitation.alias("limitation")), output.bounds, output.alpha
    )


def test_every_feasibility_fixture_is_covered():
    assert len(FIXTURES) >= 10


@pytest.mark.parametrize("name", FIXTURES)
def test_bounds_shape_gives_the_same_bands(name, request):
    old = request.getfixturevalue(name)
    assert prepare_feasibility_data(as_bounds(old)) == prepare_feasibility_data(old)


def test_conflict_is_unresolved_with_limitation_conflict(mock_feasibility_conflict):
    (band,) = prepare_feasibility_data(as_bounds(mock_feasibility_conflict))
    assert band.conflict and feasibility_zones(band) == []


def test_another_limitation_is_not_a_conflict(mock_feasibility_how_to):
    new = as_bounds(mock_feasibility_how_to)
    (band,) = prepare_feasibility_data(_set_limitation(new, "no_witness"))
    assert not band.conflict


def test_unknown_source_shows_its_text():
    old = _feasibility([_target(0.9, "unresolved")], [_source("custom_witness/v1:mine", 0.85)])
    (band,) = prepare_feasibility_data(as_bounds(old))
    assert band.evidence[0].source == "custom_witness/v1:mine"


def test_bounds_for_an_unknown_requirement_are_ignored(mock_feasibility_ruled_out):
    new = as_bounds(mock_feasibility_ruled_out)
    stray = new.bounds.head(1).with_columns(requirement=pl.lit(99, dtype=pl.Int64))
    extra = MockPlottableFeasibilityBounds(new.determinations, pl.concat([new.bounds, stray]), new.alpha)
    assert prepare_feasibility_data(extra) == prepare_feasibility_data(new)


@pytest.mark.parametrize("backend", ["matplotlib", "seaborn", "plotly", "altair"])
@pytest.mark.parametrize("plot_evidence", [False, True])
def test_plot_draws_the_bounds_shape(backend, plot_evidence, mock_feasibility_recall):
    pytest.importorskip(backend)
    assert plot(as_bounds(mock_feasibility_recall), backend=backend, plot_evidence=plot_evidence) is not None
