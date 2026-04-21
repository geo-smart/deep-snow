# Scientific Context

This page summarizes the scientific context behind `deep-snow` and points to the manuscript for full detail.

Primary reference:

- [Brencher et al., 2026 preprint](https://doi.org/10.2139/ssrn.6557436)

## What the package is designed to do

`deep-snow` uses a convolutional neural network (CNN) to estimate snow depth at 50 m resolution across the Western U.S. based on remote sensing data, coarse predictions from a physical model, and vegetation and terrain data.

The current packaged workflow is intended for inference, not training. 

## Training and evaluation data

The CNN was trained and evaluated using 233 airborne lidar snow-depth products acquired from 2016 to 2023, including the ASO archive and SnowEx QSI products.

Important consequences for users:

- the model was trained on data from the Western U.S.
- the model performs best during the parts of the snow season represented in the training data (January-July)
- predictions outside the training domain or season may still run, but should be treated more cautiously

## Main input data sources

The operational workflow combines:

- Sentinel-1 RTC backscatter for snow-on and snow-off conditions
- Sentinel-2 Level-2A optical imagery for snow-on conditions
- SNODAS snow depth for the target date
- fractional forest cover
- COP30 elevation data

## Why these inputs matter

- Sentinel-1 provides information related to dry-snow volume scattering and backscatter change between snow-on and snow-off conditions.
- Sentinel-2 provides contextual information on snow cover, reflectance, vegetation, and surface water conditions.
- SNODAS provides a daily physically based estimate of snow depth at coarse resolution.
- Forest cover and terrain variables help the model account for topographic and canopy controls on snow distribution.

## Acquisition windows and selection

- satellite data are searched for within a time window around the target date or snow-off reference date
- the time window is controlled by `buffer_period`
- the software will automatically expand the search window if it cannot find usable acquisitions

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

- temporal closeness to the requested date matters more than using all available acquisitions
- you want a mosaic dominated by the closest acquisition rather than the median of many acquisitions

hint: This strategy probably works better in sunny areas

Tradeoff:

- the nearest acquisitions might not provide useful information over the entire AOI

## Sentinel-1 orbit selection

The default `s1_orbit_selection` is `descending`.

That default is scientifically motivated. Descending Sentinel-1 passes occur in the morning (local time), and ascending passes occur in the early evening. Wet snow, which is more likely in early evening acquisitions, strongly attenuates the radar signal, reducing the amount of useful information from the acquisition. In the current code, `descending` keeps only morning acquisitions, while `all` keeps all available Sentinel-1 acquisitions.

## Spatial and temporal resolution

While the input data sources have a variety of spatial resolutions, outputs are produced at 50 m spatial resolution. 

Unique SNODAS predictions are available for each day, so the temporal resolution of predictions is one day. However, Sentinel-1 and 2 have varying revisit times, depending on the AOI. In practice, target dates one day away from one another may reuse Sentinel-1 and 2 acquisitions. To ensure a set of unique remote sensing inputs, 12 days are needed between target dates. 

That means:
- you can request daily predictions and get unique predictions for each day
- nearby dates may still rely on overlapping satellite observations
- temporal differences between day-to-day outputs should be interpreted with that acquisition reality in mind

## Single model versus ensemble

The package ships with:

- one default single model
- an ensemble of five models

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

