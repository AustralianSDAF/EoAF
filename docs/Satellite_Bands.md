# Landsat-8 and Landsat-9
| Band | Description | Wavelength | Resolution |
| ---- | ---- | ---- | ---- |
| 1 | Coastal Aerosol | 0.43 - 0.45 µm | 30 m |
| 2 | Blue | 0.450 - 0.51 µm | 30 m |
| 3 | Green | 0.53 - 0.59 µm | 30 m |
| 4 | Red | 0.64 - 0.67 µm | 30 m |
| 5 | NIR | 0.85 - 0.88 µm | 30 m |
| 6 | SWIR 1 | 1.57 - 1.65 µm | 30 m |
| 7 | SWIR 2 | 2.11 - 2.29 µm | 30 m |
| 8 | Panochromatic (PAN) | 0.50 - 0.68 µm | 15 m |
| 9 | Cirrus | 1.36 - 1.38 µm | 30 m |
| 10 | TIRS 1 | 10.6 - 11.19 µm | 100 m |
| 11 | TIRS 2 | 11.5 - 12.51 µm | 100 m |

Source: https://www.usgs.gov/landsat-missions/landsat-8, 

# Sentinel-2


| Band | Description | Wavelength | Resolution |
| ---- | ---- | ---- | ---- |
| 1 | Coastal Aerosol | 421.7 - 463.7 nm | 60m |
| 2 | Blue | 426.4 - 558.4 nm | 10m |
| 3 | Green | 523.8 - 595.8 nm | 10m |
| 4 | Red | 633.6 - 695.6 nm | 10m |
| 5 | Vegetation Red Edge | 689.1 - 719.1 nm | 20m |
| 6 | Vegetation Red Edge | 725.5 - 755.5 nm | 20m |
| 7 | Vegetation Red Edge | 762.8 - 802.8 nm | 20m |
| 8 | NIR | 726.8 - 938.8 nm | 10m |
| 8A | Narrow NIR | 843.7 - 885.7 nm | 20m |
| 9 | Water Vapour | 925.1 - 965.1 nm | 60m |
| 10 | SWIR - Cirrus | 1342.5 - 1404.5 nm | 60m |
| 11 | SWIR | 1522.7 - 1704.7 nm | 20m |
| 12 | SWIR | 2027.4 - 2377.4 nm | 20m |

Source: https://gisgeography.com/sentinel-2-bands-combinations/

# VIIRS

| Band | Description | Wavelength | Resolution |
| ---- | ---- | ---- | ---- |
| I1 | Imagery band | 0.600-0.680 µm | 375 m |
| I2 | NDVI | 0.846-0.885 µm | 375 m |
| I3 | Binary Snow Map | 1.580-1.640 µm | 375 m |
| I4 | Imagery band Clouds | 3.550-3.930 µm | 375 m |
| I5 | Imagery band Clouds | 10.500-12.400 µm | 375 m |
| M1 | Ocean Color Aerosol | 0.402-0.422 µm | 750 m |
| M2 | Ocean Color Aerosol | 0.436-0.454 µm | 750 m |
| M3 | Ocean Color Aerosol | 0.478-0.498 µm | 750 m |
| M4 | Ocean Color Aerosol | 0.545-0.565 µm | 750 m |
| M5 | Ocean Color Aerosol | 0.662-0.682 µm | 750 m |
| M6 | Atmospheric Correction | 0.739-0.754 µm | 750 m |
| M7 | Ocean Color Aerosol | 0.846-0.885 µm | 750 m |
| M8 | Cloud Particle Size | 1.230-1.25 µm | 750 m |
| M9 | Cirrus Cloud Cover | 1.371-1.386 µm | 750 m |
| M10 | Snow Fraction | 1.580-1.640 µm | 750 m |
| M11 | Clouds | 2.225-2.275 µm | 750 m |
| M12 | Sea Surface Temperature | 3.660-3.840 µm | 750 m |
| M13 | Sea Surface Temperature/Fires | 3.973-4.128 µm | 750 m |
| M14 | Cloud Top Properties | 8.400-8.700 µm | 750 m |
| M15 | Sea Surface Temperature | 10.263-11.263 µm | 750 m |
| M16 | Sea Surface Temperature | 11.538-12.488 µm | 750 m |
| DNB | Day/Night Band | 0.500-0.900 µm | 750 m |

Source: https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/viirs/

# Band Comparison

|  | **Landsat** | **Sentinel-2** | **VIIRS** |
| ---- | ---- | ---- | ---- |
| **Red** | 2 | 4 | M5 |
| **NIR** | 5 | 8 | - |
| **Narrow NIR** | - | 8A | M7 |
| **SWIR_1** | 6 | 11 | M10 |
| **SWIR_2** | 7 | 12 | M11 |
| **NDVI** | - | - | I2 |
