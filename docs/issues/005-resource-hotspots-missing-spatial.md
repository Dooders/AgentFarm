# [Bug] `compute_resource_hotspots()` ignores spatial data

**Labels:** `bug`, `analysis`

## Summary

`farm/analysis/resources/compute.py::compute_resource_hotspots()` contains the comment *"In a full implementation, this would analyze spatial hotspot data"* but currently only computes simple concentration metrics from the resource timeseries — with no spatial awareness.

## Location

`farm/analysis/resources/compute.py` lines 151–178

## Current Behavior

Returns global concentration metrics (max, mean, ratio) that are derived purely from `total_resources` over time — no grid coordinates used.

## Expected Behavior

Hotspot analysis should:

1. Ingest per-cell resource grid data (available from the spatial module's data layer)
2. Identify grid cells with resource concentration above a threshold (e.g. mean + 2σ)
3. Track hotspot movement, growth, and decay over time
4. Return spatial coordinates of hotspots, not just scalar aggregates

## Suggested Approach

- Pull per-step spatial resource data from `farm/analysis/spatial/data.py`
- Use kernel density estimation or simple thresholding on the grid
- Integrate with `farm/analysis/spatial/location.py` for coordinate handling
