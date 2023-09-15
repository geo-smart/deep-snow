# crunchy-snow
## Neural networks for Sentinel-1 SAR backscatter snow depth retrieval

### Collaborators
* Eric Gagliano, egagli@uw.edu
* Quinn Brencher, gbrench@uw.edu

### The problem
Seasonal snow provides drinking water for billions, but current global measurements of snow depth lack adequate spatial and temporal resolution for effective resource management–especially in mountainous terrain. Recent work has demonstrated the potential to retrieve snow-depth measurements from Sentinel-1 synthetic aperture radar (SAR) backscatter data. However, comparisons with airborne lidar data suggest that existing algorithms fail to capture the full complexity of relationships between snow depth, terrain, vegetation, and SAR backscatter, the physics of which are poorly understood. We suggest that a neural network may be able to effectively learn these relationships and retrieve snow depth from SAR backscatter with improved accuracy. 

During the GeoSMART Hack Week, we hope to evaluate:
* Two neural network architectures:
  * one that predicts snow depth for a single Sentinel-1 backscatter acquisition
  * one that predicts snow depth for a Sentinel-1 backscatter time series
* Three sources of target data:
  * Snow depths from Airborne Snow Observatory (ASO) lidar data
  * Snow depths retrieved with an existing algorithm implemented by spicy-snow
  * Snow depths from the physically-based modeling
* The impact of additional input data, including:
  * Various fractional forest cover datasets
  * Incidence angle, layover, and radar shadow maps
  * Digital elevation models
  * Various snow extent products
  * Optical remote sensing data

### Project Goals
We see two potential end products:
* A tool that takes a date range and a bounding box and produces a snow depth time series using our neural network
* The beginning of a paper where we present and validate our results

### Data
We will make use of scripts developed by the [spicy-snow](https://github.com/SnowEx/spicy-snow) team to automatically pull in 1) radiometrically terrain corrected (RTC) Sentinel-1 backscatter data (and associated DEMs, incidence angle maps, etc) using the Alaska Satellite Facility HyP3 on-demand processing service, 2) fractional forest cover maps, 3) snow extent data, and 4) snow depths from an existing algorithm (Lievens et al., 2022). These products are automatically delivered as xarray-compatible NetCDFs by the spicy-snow pipeline. Additional work will be needed to bring in harmonized Sentinel-2 Landsat imagery. We have access to ASO lidar snow depth data in geotiff format from numerous sites in the Western US. We (may) have access to modeled snow depth data from the Upper East River Basin in Colorado. 

### Tasks
1. Prepare training, validation, and testing dataset from large rasters (hopefully to be done by project leads before hackweek)
  * Delineate tiles, randomly subset rasters
  * Preprocess data: normalization, gap filling
  * Augment data
3. Implement two neural network architectures (team one and two during hackweek)
  * Prepare models in pytorch to accept appropriate inputs
  * Decide on initial loss functions, optimizers, hyperparameters
4. Implement training metrics (team three during hackweek)
  * Decide how to evaluate model performance. MSE? SSIM?
  * Functions to generate plots that tell us about training results at-a-glance
5. Initial training runs (all group members)
  * Overtrain small dataset, examine outputs, troubleshoot issues
6. Train neural network (all group members)
  * Train network until validation loss is minimized (or other metrics are optimized)
  * Examine impact of different input data
  * Examine impact of different target data
7. Optimize hyperparameters (teams one and two)
8. Evaluate network performance with test data (team three)
9. Application 
  * Apply model to out-of-region application area, compare results to SNOTEL or other data sources
10. Build tool
  * Create a clean, pip-installable tool that applies our neural network, using the spicy-snow framework
11. Create figures
  * Map showing aois
  * Methods schematic, network architectures
  * example inputs and outputs
  * training results
  * testing results
  * application results
12. Write methods 
13. Write results

### Additional resources or background reading
- spicy-snow background: https://github.com/SnowEx/spicy-snow/blob/main/contrib/brencher/tutorial/01background.ipynb
- Lievens et al. (2022) paper https://tc.copernicus.org/articles/16/159/2022/
- Lievens et al. (2019) paper https://www.nature.com/articles/s41467-019-12566-y
- SAR basics https://asf.alaska.edu/information/sar-information/what-is-sar/
- More SAR basics https://www.earthdata.nasa.gov/learn/backgrounders/what-is-sar
- Sentinel-1 SAR https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-1-sar
- More on ASF HyP3 RTC: https://hyp3-docs.asf.alaska.edu/guides/rtc_product_guide/
- SAR theory from 2022 UNAVCO InSAR class (more advanced) https://nbviewer.org/github/parosen/Geo-SInC/blob/main/UNAVCO2022/0.8_SAR_Theory_Phenomenology/SAR.ipynb, https://nbviewer.org/github/parosen/Geo-SInC/blob/main/UNAVCO2022/0.9_SAR_Imaging_Theory/SAR_Processor.ipynb

### Citations
Lievens, H., Brangers, I., Marshall, H. P., Jonas, T., Olefs, M., & De Lannoy, G. (2022). Sentinel-1 snow depth retrieval at sub-kilometer resolution over the European Alps. The Cryosphere, 16(1), 159-177.
