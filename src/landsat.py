"""
Module for downloading Landsat 8 and/or 9 data from DEA or NCI THREDDS and processing NDVI, NDMI and NBR.

This module provides functions for downloading Landsat 8 and/or 9 data from DEA or NCI THREDDS using the STAC API.
It also includes functions for processing the downloaded data by calculating NDVI, NDMI, and NBR.
The processed data is saved as GeoTIFFs.

Author: Calvin Pang
Date: 2024-02-28
"""

import json
import sys
import urllib
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd
import pystac_client
import rioxarray as rio
import typer
from tqdm.auto import tqdm

# Add the parent directory to the path so that the module can be run from the command line
try:
    from utils import (
        DEADownloaderTool,
        TqdmUpTo,
        group_by_date,
        merge_products,
        unique_session_id,
        xa_int16_to_float32,
    )
except ModuleNotFoundError:
    sys.path.append("..")
    from src.utils import (
        DEADownloaderTool,
        TqdmUpTo,
        group_by_date,
        merge_products,
        unique_session_id,
        xa_int16_to_float32,
    )


#####################################################################################################
#                                                                                                   #
#                                   Download Functions                                              #
#                                                                                                   #
#####################################################################################################


class LandsatDownloaderTool(DEADownloaderTool):
    """
    Subclass for downloading Landsat 8 or 9 data from DEA or NCI THREDDS.
    """

    THREDDS_URL_HEADER = "https://dapds00.nci.org.au/thredds/fileServer/xu18"

    def get_thredds_url_header(self) -> str:
        """
        Returns the NCI THREDDS URL header for Landsat data.

        Args:
            None

        Returns:
            str: The NCI THREDDS URL header for Landsat data.
        """
        return self.THREDDS_URL_HEADER

    def check_collection(self, collection: str) -> bool:
        """
        Checks if the collection is valid.

        Args:
            collection (str): The collection to check.

        Returns:
            bool: True if the collection is valid, False otherwise.
        """
        if collection in {"ga_ls8c_ard_3", "ga_ls9c_ard_3"}:
            return True
        else:
            raise ValueError(
                "Invalid Collection. Please select 'ga_ls8c_ard_3' or 'ga_ls9c_ard_3'."
            )

    def get_next_aquisition_date(self, search_criteria: dict):
        """
        Returns the next aquisition date for Landsat-8/9 data.

        Args:
            search_criteria (dict): The search criteria.

        Returns:
            None
        """
        # Search for all products based on search criteria within the last 14 days
        start_date = datetime.now().date() - timedelta(days=14)
        end_date = datetime.now().date() - timedelta(days=1)

        search_criteria["datetime"] = (
            f"{start_date.strftime('%Y-%m-%d')}T00:00:00Z/{end_date.strftime('%Y-%m-%d')}T23:59:59Z"
        )

        sorted_results = sorted(
            self.search(search_criteria, False).item_collection(),
            key=lambda item: item.datetime,
        )

        # Print the latest aquisition date and the next aquisition date (approximately 8 days after the latest aquisition date)
        print(
            f"Latest Landsat-8 or 9 product for the selected area of interest is {sorted_results[-1].properties['title']} on {sorted_results[-1].datetime.date()}."
        )
        print(
            f"Next acquisition date for this product is approximately {sorted_results[-1].datetime.date() + timedelta(days=8)}.\n"
        )

    def download(
        self,
        search_results: pystac_client.item_search.ItemSearch,
        download_dir: Path,
        download_from_thredds: bool,
    ):
        """
        Searches for and downloads data from the DEA STAC API or NCI THREDDS.

        Args:
            search_results (pystac.ItemCollection): The search results from the DEA STAC API.
            download_dir (Path): The directory to save the downloaded data to.
            download_from_thredds (bool): True if the data should be downloaded from NCI THREDDS, False if it should be downloaded from DEA STAC API.

        Returns:
            None
        """
        for item in search_results.items():
            # Creating a directory to save the product
            save_dir = Path(download_dir, item.properties["title"])
            save_dir.mkdir(parents=True, exist_ok=True)

            # Saving Product Metadata
            metadata_path = Path(save_dir, "metadata.json")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(item.properties, f)

            print(
                f"\nDownloading {item.properties['title']} from {'NCI THREDDS' if download_from_thredds else 'DEA'}"
            )

            # Downloading Assets in product
            # item_assets = list(item.assets.keys())
            item_assets = ["nbart_red", "nbart_nir", "nbart_swir_1", "nbart_swir_2"]

            for asset in item_assets:
                asset_url = item.assets[asset].get_absolute_href()

                # Downloading from DEA
                if not download_from_thredds:
                    asset_url = asset_url.replace(
                        "s3://dea-public-data", self.DEA_URL_HEADER
                    )
                else:
                    asset_url = asset_url.replace(
                        "s3://dea-public-data/baseline", self.get_thredds_url_header()
                    )

                # Specifying the save path of the asset
                asset_filename = Path(asset_url).name
                save_path = Path(save_dir, asset_filename)

                # Check if file already exists, else download
                if save_path.exists():
                    print(f"File {save_path} already exists, skipping...")
                    continue
                else:
                    with TqdmUpTo(
                        unit="B",
                        unit_scale=True,
                        miniters=1,
                        desc=asset_url.split("/")[-1],
                    ) as t:
                        urllib.request.urlretrieve(
                            asset_url, save_path, reporthook=t.update_to
                        )


#####################################################################################################
#                                                                                                   #
#                                   Processing Functions                                            #
#                                                                                                   #
#####################################################################################################


def process_product(
    input_dir: Path, output_dir: Path, crop: bool, bbox: tuple, shapefile: Path
):
    """
    Processes a Landsat 8 or 9 product, calculating the NDVI, NDMI and NBR.
    Then saves the results as GeoTIFFs in the output directory.

    Args:
        input_dir (Path): Directory containing the raw Landsat 8/9 product.
        output_dir (Path): Directory to save the processed Landsat 8/9 data.
        crop (bool): Whether to crop the products to the area of interest.
        bbox (tuple): A tuple of the bounding box coordinates.
        shapefile (Path): Path to the shapefile.

    Returns:
        None
    """
    # Loading the required bands
    bands = {
        "red": rio.open_rasterio(list(input_dir.glob("*band04.tif"))[0]),
        "nir": rio.open_rasterio(list(input_dir.glob("*band05.tif"))[0]),
        "swir_1": rio.open_rasterio(list(input_dir.glob("*band06.tif"))[0]),
        "swir_2": rio.open_rasterio(list(input_dir.glob("*band07.tif"))[0]),
    }

    for band_name in ["red", "nir", "swir_1", "swir_2"]:
        # Reprojecting the bands to EPSG:4326
        bands[band_name] = bands[band_name].rio.reproject("EPSG:4326")

        # Convert from int16 to float32
        bands[band_name] = xa_int16_to_float32(
            bands[band_name], bands[band_name].attrs["_FillValue"]
        )

        # Update the CRS to EPSG:4326
        bands[band_name].rio.write_crs(4326, inplace=True)
        bands[band_name].attrs["crs"] = "EPSG:4326"

    # Calculating NDVI, NDMI and NBR
    ndvi = (bands["nir"] - bands["red"]) / (bands["nir"] + bands["red"])
    ndmi = (bands["nir"] - bands["swir_1"]) / (bands["nir"] + bands["swir_1"])
    nbr = (bands["nir"] - bands["swir_2"]) / (bands["nir"] + bands["swir_2"])

    # Opening metadata file
    with open(Path(input_dir, "metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)
    keys = list(metadata.keys())

    # Writing Product Name to the Attributes
    ndvi.attrs["name"] = f"{output_dir.name}_ndvi"
    ndmi.attrs["name"] = f"{output_dir.name}_ndmi"
    nbr.attrs["name"] = f"{output_dir.name}_nbr"

    for band in [ndvi, ndmi, nbr]:
        # Copying original metadata to the attributes
        for key in keys:
            band.attrs[key] = metadata[key]

        # Updating CRS to EPSG:4326
        band.rio.write_crs(4326, inplace=True)
        band.attrs["crs"] = "EPSG:4326"
        band.attrs["proj:epsg"] = 4326

        # Clipping the bands to the bounding box or shapefile
        if crop:
            try:
                if shapefile:
                    aoi_polygon = gpd.read_file(shapefile)
                    band = band.rio.clip(aoi_polygon.geometry)
                else:
                    band = band.rio.clip_box(*bbox)
            except Exception:
                print(
                    f"No data in bounds for {band.attrs['name']} in product {output_dir.name} - Skipping..."
                )
                continue

        # Saving the calculated band as a GeoTIFF
        band.rio.to_raster(Path(output_dir, f"{band.attrs['name']}.tif"))


def process_products(
    product_paths: tuple,
    landsat_dirs: list,
    landsat_processed_dir: Path,
    all_products: bool,
    crop: bool,
    bbox=tuple,
    shapefile=Path,
):
    """
    Processes a list of Landsat-8 and/or Landsat-9 products, calculating the NDVI, NDMI and NBR.
    Then saves the results as GeoTIFFs in the output directory.

    Args:
        product_paths (tuple): Paths to the products to process.
        landsat_dirs (list): List of directories containing raw Landsat 8/9 products.
        landsat_processed_dir (Path): Directory to save processed Landsat 8/9 products.
        all_products (bool): Whether to process all products in the directory.
        crop (bool): Whether to crop the products to the area of interest.
        aoi_bbox (tuple): A tuple of the bounding box coordinates.
        shapefile (Path): Path to the shapefile.

    Returns:
        None
    """

    # Process all Landsat products in the directory, if all_products is True
    products = landsat_dirs if all_products else product_paths

    # Process each product in the list
    for product in tqdm(products):
        # Create the output directory
        output_dir = Path(landsat_processed_dir, product.name)
        if crop:
            output_dir = Path(landsat_processed_dir, f"{product.name}_cropped")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check if the product is already processed
        if not any(output_dir.glob("*.tif")):
            print(f"\nProcessing Product {product.name}")
            process_product(product, output_dir, crop, bbox, shapefile)
        else:
            print(f"\nProduct {product.name} already processed - Skipping")

        # If output directory is empty, remove it as it is not needed
        if not list(output_dir.glob("*.tif")):
            output_dir.rmdir()
            print(f"Product {output_dir} is empty and has been removed.")


#####################################################################################################
#                                                                                                   #
#                                   Command Line Interface                                          #
#                                                                                                   #
#####################################################################################################
def main(
    session_id: str = None,
    start_date: datetime = "2024-01-01",
    end_date: datetime = "2024-01-08",
    min_lon: float = 116.01897,
    max_lon: float = 116.20093,
    min_lat: float = -32.30959,
    max_lat: float = -31.98176,
    shapefile: Path = None,
    collection: str = "ga_ls8c_ard_3",
    crop: bool = False,
    merge: bool = False,
    download_from_thredds: bool = True,
    raw_dir: Path = Path("data/raw_data/landsat_8_9"),
    processed_dir: Path = Path("data/processed_data/landsat_8_9"),
):
    """
    Main function to bulk download and process Landsat-8 and Landsat-9 data from DEA or NCI THREDDS.

    Args:
        start_date (datetime): Start date for the data acquisition. Defaults to "2024-01-01".
        end_date (datetime): End date for the data acquisition. Defaults to "2024-01-08".
        min_lon (float): Minimum longitude for the area of interest. Defaults to 116.01897.
        max_lon (float): Maximum longitude for the area of interest. Defaults to 116.20093.
        min_lat (float): Minimum latitude for the area of interest. Defaults to -32.30959.
        max_lat (float): Maximum latitude for the area of interest. Defaults to -31.98176.
        shapefile (Path): Path to the shapefile defining the area of interest. Defaults to None.
        collection (str): Collection name for Landsat data. Defaults to "ga_ls8c_ard_3".
        crop (bool): Whether to crop the products to the area of interest. Defaults to True.
        merge (bool): Whether to merge the downloaded products. Defaults to False.
        download_from_thredds (bool): If True, download from NCI THREDDS, else download from DEA. Defaults to True.
        raw_dir (Path): Directory to save the raw Landsat-8/9 data. Defaults to "data/raw_data/landsat_8_9".
        processed_dir (Path): Directory to save the processed Landsat-8/9 data. Defaults to "data/processed_data/landsat_8_9".

    Returns:
        None
    """
    # Generate session_id and create directories
    if session_id is None:
        session_id = unique_session_id()
    raw_dir = Path(raw_dir, session_id)
    processed_dir = Path(processed_dir, session_id)

    for directory in [raw_dir, processed_dir]:
        Path(directory).mkdir(parents=True, exist_ok=True)

    # Initialise the LandsatDownloaderTool
    client = LandsatDownloaderTool()

    # Check if the collection is valid
    client.check_collection(collection)

    # Search and Download Landsat-8 or Landsat-9 data
    search_criteria = client.search_parameters(
        min_lon=min_lon,
        max_lon=max_lon,
        min_lat=min_lat,
        max_lat=max_lat,
        shapefile=shapefile,
        start_date=start_date,
        end_date=end_date,
        collection=collection,
    )

    search_results = client.search(search_criteria=search_criteria, print_results=True)

    client.get_next_aquisition_date(search_criteria=search_criteria)

    client.download(
        search_results=search_results,
        download_dir=raw_dir,
        download_from_thredds=download_from_thredds,
    )

    # Process Landsat-8 and/or Landsat-9 products
    print("\nProcessing Landsat products...")
    landsat_dirs = list(raw_dir.glob("*"))
    landsat_dirs = [dir for dir in landsat_dirs if dir.is_dir()]

    process_products(
        product_paths=landsat_dirs,
        landsat_dirs=landsat_dirs,
        landsat_processed_dir=processed_dir,
        all_products=True,
        crop=crop,
        bbox=(min_lon, min_lat, max_lon, max_lat),
        shapefile=shapefile,
    )
    print("\nLandsat processing complete...")

    # Merge the processed Landsat-8/9 products
    if merge:
        print("\nMerging Landsat products...")
        merge_dirs = group_by_date(processed_dir=processed_dir)
        for dirs in merge_dirs:
            merge_products(dirs)
        print("\nLandsat products merged...")


if __name__ == "__main__":
    typer.run(main)
