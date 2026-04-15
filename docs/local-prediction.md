# Local Prediction

This page covers the local interfaces for `deep-snow`: installation, CLI usage, and Python usage.

For the scientific interpretation of these options, see [scientific-context.md](scientific-context.md).

## Installation

The repository currently uses a Conda environment plus a local package install.

```bash
git clone https://github.com/geo-smart/deep-snow.git
cd deep-snow
conda install mamba -n base -c conda-forge
mamba env create -f environment.yml
conda activate deep-snow
pip install -e .
```

The package installs a command-line entry point named `deep-snow`.

## CLI usage

The main local commands are:

- `predict-sd`: predict snow depth for one AOI/date
- `predict-sd-timeseries`: predict a time series for one AOI across a date range

Large AOIs are tiled and mosaiced automatically during `predict-sd`.

```bash
deep-snow predict-sd 20240320 20230910 "-108.20 37.55 -108.00 37.75" 25
```

Positional arguments:

- `target_date`: target prediction date in `YYYYmmdd`
- `snow_off_date`: reference snow-off date in `YYYYmmdd`
- `aoi`: bounding box as `minlon minlat maxlon maxlat`
- `cloud_cover`: maximum allowed Sentinel-2 cloud-cover percentage

Useful options for `predict-sd`:

- `--out-dir`: output directory, default `data`
- `--out-crs`: output CRS, default `wgs84`
- `--model-path`: custom model checkpoint
- `--use-ensemble`: use the configured ensemble instead of the packaged single-model default
- `--buffer-period`: search window in days around the target date
- `--fcf-path`: cached forest-cover fraction raster
- `--tile-large-aoi`: automatically tile large AOIs locally
- `--tile-size-degrees`: override the local tile size
- `--delete-inputs`: remove `model_inputs.nc` after prediction
- `--write-tif`: write the predicted GeoTIFF
- `--predict-swe`: also derive density and SWE from predicted depth using the Hill et al. model
- `--gpu`: run inference on GPU when available

Two options deserve special attention:

- `--use-ensemble`: uses the packaged ensemble of five models instead of the default single model
- `--buffer-period`: controls the search window around the requested date for satellite acquisitions

For most users, stopping here is enough. There is also a lower-level `predict-tile` command for advanced/debug use when you explicitly want to force a single tile without automatic large-AOI tiling.

Example:

```bash
deep-snow predict-sd-timeseries 20240101 20240320 0901 "-108.20 37.55 -108.00 37.75" 25
```

If you also want density and SWE for every output date, add `--predict-swe True`.

Other CLI utilities:

- `deep-snow prep-tiles ...`: generate tile jobs for a bounding box
- `deep-snow prep-time-series ...`: generate target and snow-off date pairs for a time series

## Python API

The main Python entry points are `predict_sd` and `predict_sd_timeseries`.

```python
from deep_snow import predict_sd

aoi = {
    "minlon": -108.20,
    "minlat": 37.55,
    "maxlon": -108.00,
    "maxlat": 37.75,
}

ds = predict_sd(
    aoi=aoi,
    target_date="20240320",
    snowoff_date="20230910",
    out_dir="data/application",
    out_crs="utm",
    cloud_cover=25,
)
```

Common arguments:

- `aoi`: dictionary with `minlon`, `minlat`, `maxlon`, `maxlat`
- `target_date`: date to predict
- `snowoff_date`: reference snow-off date
- `out_dir`: working and output directory
- `out_crs`: output coordinate reference system
- `cloud_cover`: maximum allowed Sentinel-2 cloud-cover percentage
- `buffer_period`: initial search window for acquisitions
- `gpu`: use GPU if available
- `use_ensemble`: use ensemble model defaults
- `predict_swe`: when `True`, also compute Hill-model SWE and bulk density from predicted depth
- `selection_strategy`: `composite` or `nearest_usable`
- `sentinel1_orbit_selection`: `descending` or `all`

For time series, use `predict_sd_timeseries`.

Advanced lower-level helpers such as `predict_tile` and `predict_batch` still exist internally, but most local users should not need them. The older alias `predict_sd_ts` is also still available for compatibility.

## What the selection options mean

### `selection_strategy="composite"`

This is the default. The workflow forms a median mosaic from all usable acquisitions in the search window. It usually gives the most stable and spatially complete input dataset, especially for large AOIs or when no single scene is clean and complete.

### `selection_strategy="nearest_usable"`

This option prioritizes acquisitions closest in time to the requested date and only adds additional scenes to fill missing pixels. It is often a good choice when you want the inputs to remain tied as closely as possible to one target date.

### `sentinel1_orbit_selection="descending"`

This is the recommended default. The model and manuscript were built around the idea that descending-pass Sentinel-1 acquisitions are generally more favorable for dry-snow retrieval than later-day passes.

### `sentinel1_orbit_selection="all"`

This broadens the available Sentinel-1 pool, which can help in sparse cases, but it relaxes that scientific constraint.

## Automatic retries and expanded windows

If a run cannot find usable acquisitions, the local prediction workflow can automatically expand the acquisition search window and retry. That is helpful operationally, but it also means the final inputs may be farther from the requested date than the initial `buffer_period` suggests.

For science use, it is worth checking:

- the printed prediction summary
- the acquisition provenance stored in metadata
- any gap outputs written alongside the prediction

## Output behavior

Local runs can produce:

- `model_inputs.nc`
- predicted snow-depth GeoTIFFs
- optional `*_swe.tif` and `*_density.tif` rasters when `predict_swe=True`
- gap rasters and gap NetCDFs when missing-data diagnostics are written

The returned dataset also carries useful metadata about the run, including model settings and acquisition provenance when available.
