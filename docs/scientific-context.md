# Scientific Context

This page summarizes the scientific context behind `deep-snow` and points to the manuscript for full detail.

Primary reference:

- [Brencher et al., 2026 preprint](https://doi.org/10.2139/ssrn.6557436)

## What the model is designed to do

`deep-snow` is designed to estimate snow depth at 50 m resolution across the Western U.S. by combining a coarse physically based snow estimate with higher-resolution remote sensing and terrain data in a convolutional neural network.

The current packaged workflow is intended for inference, not training. It exposes the operational prediction pipeline built around the trained CNN models.

## Training and evaluation data

The CNN was trained and evaluated using airborne lidar snow-depth products acquired from 2016 to 2023, including the ASO archive and SnowEx QSI products. After quality control and filtering, the working dataset contained 233 airborne lidar snow-depth products.

Important consequences for users:

- the model is best supported in the Western U.S.
- the model is strongest during the parts of the snow season represented in the training data (January-July)
- predictions outside the training domain or season may still run, but should be treated more cautiously

## Main input data sources

The operational workflow combines:

- Sentinel-1 RTC backscatter for snow-on and snow-off conditions
- Sentinel-2 Level-2A optical imagery for snow-on conditions
- SNODAS snow depth for the target date
- fractional forest cover
- COP30 elevation data

The preprint argues that SNODAS provides a strong coarse-resolution prior, while remote sensing and terrain help refine the spatial pattern to 50 m resolution.

## Why these inputs matter

- Sentinel-1 provides information related to dry-snow volume scattering and backscatter change between snow-on and snow-off conditions.
- Sentinel-2 provides contextual information on snow cover, reflectance, vegetation, and surface water conditions.
- SNODAS provides a daily physically based estimate of snow depth at coarse resolution.
- Forest cover and terrain variables help the model account for strong topographic and canopy controls on snow distribution.

## Acquisition windows and selection

The code and manuscript together support the following interpretation:

- satellite data are searched within a time window around the target date or snow-off reference date
- the window is controlled by `buffer_period`
- the software can automatically expand the search window if it cannot find usable acquisitions

Two acquisition-selection strategies are currently exposed:

### `composite`

`composite` takes the median across all acquisitions in the search window after masking unusable pixels. This usually gives the most spatially complete and robust input mosaic, especially when no single satellite scene covers the AOI cleanly.

Use `composite` when:

- you want the default operational behavior
- your AOI is large
- you expect patchy cloud cover or incomplete single-scene coverage
- you prefer a more stable mosaic over strict temporal proximity

Tradeoff:

- the resulting input may combine observations from multiple acquisition times, so it is less tied to one exact overpass date

### `nearest_usable`

`nearest_usable` sorts acquisitions by temporal distance from the requested date, then fills the mosaic starting from the closest usable acquisition and adds additional scenes only as needed to cover missing pixels.

Use `nearest_usable` when:

- temporal closeness to the requested date matters more than using all available scenes
- you want a mosaic dominated by the closest acquisition rather than the median of many scenes

Tradeoff:

- this can be more sensitive to artifacts or limited spatial coverage in the nearest scenes

## Sentinel-1 orbit selection

The default `s1_orbit_selection` is `descending`.

That default is scientifically motivated. Restricting Sentinel-1 to the descending pass to reduce the influence of wet-snow conditions that are more likely later in the day. In the current code, `descending` filters acquisitions by acquisition hour, while `all` keeps all available Sentinel-1 acquisitions.

Use `all` only if you have a specific reason to do so and are comfortable with the possibility of mixing in less favorable radar conditions.

## Resolution and effective temporal detail

Outputs are written at 50 m resolution, but the effective temporal detail of the satellite-driven part of the prediction is limited by satellite revisit frequency and data availability.

That means:

- you can request daily predictions
- nearby dates may still rely on overlapping satellite observations
- temporal differences between day-to-day outputs should be interpreted with that acquisition reality in mind

## Single model versus ensemble

The package ships with:

- one default single model
- an ensemble of five packaged models

The best performance was obtained using the median prediction from an ensemble of five CNNs. In this package, the ensemble option is exposed as `use_ensemble=True`.

Practical interpretation:

- use the default single model when you want a simpler, faster baseline
- use the ensemble when you want the package's more conservative multi-model prediction path

## Known limitations

There are currently several limitations that users should keep in mind:

- underprediction of very deep snow
- residual aspect-related bias
- residual seasonal bias
- reduced precision in steep, heterogeneous, sparsely forested terrain
- lower confidence outside the Western U.S. training domain
- lower confidence in parts of the season that are poorly represented in the training data

Local validation against in situ or airborne observations is valuable whenever available.

