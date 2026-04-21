# GitHub Actions

`deep-snow` includes GitHub Actions workflows for running tiled predictions and time-series jobs from the GitHub web interface.

If you want the scientific background behind these workflows, see [scientific-context.md](scientific-context.md) and the [preprint](https://doi.org/10.2139/ssrn.6557436).

## Before you begin

1. Fork the repository to your own GitHub account.
2. Open your fork.
3. Go to the `Actions` tab.

## Available workflows

### `batch_predict_sd`

Use this workflow when you want snow-depth output for one target date over one area of interest.

What it does:

- splits the AOI into tiles when needed
- runs a prediction workflow for each tile
- uploads the results as GitHub artifacts

### `batch_sd_timeseries`

Use this workflow when you want outputs for a range of target dates.

What it does:

- generates target dates across the requested range
- derives a snow-off date for each target date from the supplied `snow_off_day`
- calls the batch prediction workflow for each date

## Required inputs

### For `batch_predict_sd`

- `target_date`: date to predict in `YYYYmmdd` format
- `snow_off_date`: snow-free reference date in `YYYYmmdd` format
- `aoi`: bounding box as `minlon minlat maxlon maxlat`
- `cloud_cover`: maximum allowed Sentinel-2 cloud-cover percentage

Optional but important:

- `selection_strategy`: `composite` or `nearest_usable`
- `s1_orbit_selection`: `descending` or `all`
- `use_ensemble`: `True` or `False`
- `predict_swe`: `True` or `False`

### For `batch_sd_timeseries`

- `begin_date`: first target date in `YYYYmmdd` format
- `end_date`: last target date in `YYYYmmdd` format
- `snow_off_day`: month and day in `mmdd` format, usually a late-summer date
- `aoi`: bounding box as `minlon minlat maxlon maxlat`
- `cloud_cover`: maximum allowed Sentinel-2 cloud-cover percentage

Optional but important:

- `selection_strategy`: `composite` or `nearest_usable`
- `s1_orbit_selection`: `descending` or `all`
- `use_ensemble`: `True` or `False`
- `predict_swe`: `True` or `False`

## How the key options change behavior

### `selection_strategy`

`selection_strategy` controls how the package turns the available satellite scenes in the search window into a single input mosaic.

`composite`:

- computes a median composite across all usable acquisitions in the window
- usually produces the most spatially complete and stable mosaic
- is the recommended default for most users

`nearest_usable`:

- prioritizes the acquisition closest in time to the requested date
- fills missing areas with additional acquisitions only when needed
- is useful when temporal proximity matters more than a full-window composite

In practice, `composite` is often the safer default for large AOIs or patchy cloud conditions, while `nearest_usable` is a better choice when you want the inputs to stay as close as possible to the requested date.

### `s1_orbit_selection`

`descending` is the default and generally recommended option. Limiting Sentinel-1 to descending passes helps reduce the influence of wet-snow conditions that are more likely later in the day.

`all` uses all available Sentinel-1 passes.

### `use_ensemble`

`False` uses the packaged default single model.

`True` uses the packaged ensemble of five models. The preprint reports the strongest performance from the ensemble path, so this is a reasonable option when you want a more conservative operational result.

### `predict_swe`

`False` keeps the original depth-only behavior.

`True` also applies the Hill et al. depth-to-SWE model after depth prediction and uploads `*_swe.tif` and `*_density.tif` artifacts for each tile.

## Search windows and retries

The Actions workflows use the package's internal acquisition-search behavior. In the current implementation, predictions begin with a default search window around the requested dates, and the underlying local prediction workflow will expand that window if it cannot find usable acquisitions.

This is helpful operationally, but it also means the final inputs may come from a broader temporal window than you initially expected. The prediction summary and provenance metadata are therefore important for interpretation.

## Artifacts

The tile prediction workflow uploads these outputs when present:

- `*_sd.tif`: predicted snow-depth raster
- `*_swe.tif`: predicted snow water equivalent raster
- `*_density.tif`: predicted bulk snow density raster
- `*_input_gaps.tif`: combined raster showing where required inputs were missing
- `*_input_gaps.nc`: detailed gap information

For large AOIs or time-series jobs, expect multiple artifacts because work is split across tiles and dates.


## Tips

- Keep the AOI small for first-time runs.
- Use a realistic late-summer `snow_off_date` or `snow_off_day`.
- Start with the default `selection_strategy` unless temporal proximity is more important than a full composite.
- Keep `s1_orbit_selection=descending` unless you have a specific reason to broaden the Sentinel-1 inputs.
- Consider `use_ensemble=True` for more conservative science-facing runs.

## When to switch to local runs

Consider moving to local execution if:

- you want to inspect intermediate files in detail
- you want notebook-based exploration
- you need to debug repeated failures
- you want to build a custom workflow around the package

See [local-prediction.md](local-prediction.md).
