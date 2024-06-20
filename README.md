# Eyes on Australian Forests

Author:
- Calvin Pang (calvin.pang@curtin.edu.au)
- Leigh Tyers (leigh.tyers@curtin.edu.au)

# Overview

This repository contains a set of Jupyter Notebooks and accompanying Python scripts to allow a user to interactively query, download and process satellite data into vegetation health metrics.

Metrics include:
- Normalised Difference Vegetation Index (NDVI)
    
    $NDVI = \frac{NIR - Red}{NIR + Red}$

- Normalised Difference Moisture Index (NDMI)
    
    $NDMI = \frac{NIR - SWIR1}{NIR + SWIR1}$

- Normalised Burn Ratio (NBR)

    $NBR = \frac{NIR - SWIR2}{NIR + SWIR2}$

Available satellite imagery sources currently include:
- Sentinel-2
- Landsat-8
- VIIRS

Due to the various satellite imagery sources and processing requirements, each has a dedicated Jupyter Notebook for downloading, processing and visualisation.

## Demo
You can view a static visualisation of NDVI, NDMI and NBR as captured by Sentinel-2 on 2024-01-06 for the Perth region below.

![](sample/demo_sentinel_2.png)

For an interactive visualisation, you can download the *sample/demo_sentinel_2.html* file in this repository and view via your web browser.

## Wiki
An in-depth guide on the Eyes on Australian Forests repository can be found [here](https://github.com/AustralianSDAF/EoAF/wiki).

# Quick Start
## Installation
To use this repository, you will need to have conda setup on your machine. A recommended version of conda is [Miniforge](https://github.com/conda-forge/miniforge). Please follow the latest installation instructions found on the link provided.

Once you have installed Miniforge, please download this repository (top-right code button) and follow the below steps in a terminal to install the environment (Mac & Linux).
```
% cd <extracted folder, e.g. ~/Downloads/EoAF>
% conda env create -f environment.yml
% conda activate EoAF
% poetry install --no-root
```

On Windows, you will need to run these commands inside "Miniforge Prompt" (that you just installed from the link above) available from the start menu

## Configuration
Before proceeding, please confirm the download and processed directory paths for each satellite provider within `config.yaml` and update as required.
By default, the data will be saved in the *data* directory located within this repository's working directory.

## Usage (Mac and Linux)
### WebUI (Basic, Mac and Linux)
A Gradio web interface is available for each satellite source for the basic downloading and processing of satellite products.
Visualisations of the data can be generated and downloaded as HTML for viewing via a web browser.

The webUI can be launched as follows.

**Sentinel-2 Example**
```
python webui/sentinel_ui.py
```


### WebUI (Basic, Windows)
If you are on windows, once you have followed the installation steps above, to use the webUI  please instead run the following in the miniforge prompt:

```
poetry run python webui\sentinel_ui.py
```

### Jupyter Notebooks (Intermediate)
For additional interactivity and customisations, a set of Jupyter Notebooks have been provided for each satellite source.

To access these notebooks, you can open then via VSCode or by launching Jupyter Lab via Terminal to open your required notebook.
```
jupyter lab
```
Please run each cell sequentially and use the Jupyter Widgets to interact with the notebook.


### Command Line Interface (Advanced)
A Command Line Interface has also been provided for automatically download and process products based on a search criteria. This enables the creation of automated processing pipelines and usage of output products in applications such as QGIS.

**Sentinel-2 Example with Bounding Box**
```
python src/sentinel.py --session-id Sentinel_2_Example --start-date 2024-01-01 --end-date 2024-01-08 --min-lon 116.01897 --max-lon 116.20093 --min-lat -32.30959 --max-lat -31.98176 --collection ga_s2am_ard_3 --crop --no-merge --download-from-thredds --raw-dir data/raw_data/sentinel_2 --processed-dir data/processed_data/sentinel_2
```

**Landsat 8 & 9 Example with Shapefile**
```
python src/landsat.py --session-id Landsat_8_Example --start-date 2024-01-01 --end-date 2024-01-08 --shapefile shapefiles/large.shp --collection ga_ls8c_ard_3 --crop --merge --download-from-thredds --raw-dir data/raw_data/landsat_8_9 --processed-dir data/processed_data/landsat_8_9
```

Help documentation for the CLI can be accessed as follows.
```
python src/viirs.py --help
```

**Note: All three interfaces will store the downloaded and processed products in the previously configured directories. You may export these for use in other applications such as QGIS or ArcGIS.**

# Product naming conventions
Once products have been downloaded, it is advised to keep the given naming conventions to maintain [data provenance](https://ardc.edu.au/resource/data-provenance/).  
To assist in this:  
 - Sentinel 2 and Landsat naming conventions can be found [here](https://knowledge.dea.ga.gov.au/guides/reference/collection_3_naming/).  
 - VIIRS naming conventions can be found [here](https://lpdaac.usgs.gov/data/get-started-data/collection-overview/missions/s-npp-nasa-viirs-overview/#viirs-naming-conventions).  


# Additional Notes
1. Sentinel Australasia Region Access (SARA) is undertaking major upgrades to migrate to the new Copernicus Data Space Ecosystem (CDSE) platform from October 2023 to Mid 2024. Data access may be disrupted, with changes on how to access data during this period. It is hoped that Digital Earth Australia will account for this major change and their STAC API will remain accessible.
2. As of 28th February 2024, AρρEEARS are currently experiencing a technical issue preventing AρρEEARS from correctly processing requests containing VIIRS version 2 data. As such, they have made the impacted products unavailable for selection and do not have an estimated time for when the issue will be fixed.
