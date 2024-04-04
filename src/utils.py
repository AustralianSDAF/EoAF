"""
Module containing utility functions for Eyes on Australian Forests downloading and processing.
Author: Calvin Pang
Date: 2024-01-31
"""

import shutil
import uuid
import warnings
import zipfile
from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import xarray as xa
from pystac_client import Client
from rasterio.merge import merge
from tqdm import tqdm

#####################################################################################################
#                                                                                                   #
#                                   Basic Utility Functions                                         #
#                                                                                                   #
#####################################################################################################


def clean_up_temp_files(product_paths: tuple, extracted_dir: Path, all_products: bool):
    """
    Removes temporary files from the project.

    Args:
        product_paths (tuple): The paths to the processed products.
        extracted_dir (Path): The directory containing the extracted products.
        all_products (bool): True if all products should be removed, False
            if only the processed products should be removed.

    Returns:
        None

    """
    warnings.warn(
        "This function is deprecated and will be removed in a future release.",
        DeprecationWarning,
    )

    # Clean up the Extracted Directories of the Processed Products
    if all_products:
        shutil.rmtree(extracted_dir)
    else:
        for path in product_paths:
            shutil.rmtree(path)

    # Clean up Temporary Files from eoreader
    shutil.rmtree("temp")


def extract_as_safe(input_path: Path, extract_dir: Path):
    """
    Extracts a downloaded Sentinel-2 product from .zip to .SAFE format.

    Args:
        input_path (Path): The path to the downloaded Sentinel-2 product.
        extract_dir (Path): The directory to extract the product to.

    Returns:
        None
    """
    warnings.warn(
        "This function is deprecated and will be removed in a future release.",
        DeprecationWarning,
    )
    with zipfile.ZipFile(input_path, "r") as zip_ref:
        zip_ref.extractall(Path(extract_dir, f"{input_path.stem}.SAFE"))


def unique_session_id():
    """
    Generates a unique session ID.

    Returns:
        str: A unique session ID.
    """
    return f"{date.today().strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"


#####################################################################################################
#                                                                                                   #
#                                   Downloader Utility Functions                                    #
#                                                                                                   #
#####################################################################################################


class DEADownloaderTool(ABC):
    """
    Class to search and download data from the DEA STAC API or NCI THREDDS.
    """

    DEA_STAC_ENDPOINT = "https://explorer.sandbox.dea.ga.gov.au/stac/"
    DEA_URL_HEADER = "https://data.dea.ga.gov.au"

    @abstractmethod
    def get_thredds_url_header(self) -> str:
        """
        Returns the NCI THREDDS URL header.
        """
        raise NotImplementedError

    def search_parameters(
        self,
        start_date: date,
        end_date: date,
        collection: str,
        min_lon: float = None,
        max_lon: float = None,
        min_lat: float = None,
        max_lat: float = None,
        shapefile: Path = None,
    ):
        """
        Returns search parameters for DEA STAC API as a dictionary.

        Args:
            start_date (date): Start date.
            end_date (date): End date.
            collection (str): Digital Earth Australia Collection.
            min_lon (float): Minimum longitude. (Optional)
            max_lon (float): Maximum longitude. (Optional)
            min_lat (float): Minimum latitude. (Optional)
            max_lat (float): Maximum latitude. (Optional)
            shape_file (str): Path to the shape file. (Optional)

        Returns:
            dict: The STAC API search criteria.

        Raises:
            ValueError: If no shapefile or bounding box is specified.
            ValueError: If no collection is specified.
        """

        # Check if a shapefile or bounding box is provided
        if shapefile is None and not all([min_lon, max_lon, min_lat, max_lat]):
            raise ValueError("Either shapefile path or bounding box must be specified")

        # Check if at least one collection is specified
        if not collection:
            raise ValueError("A DEA collection must be specified")

        # If no shapefile is provided, use bounding box
        if shapefile is None and all([min_lon, max_lon, min_lat, max_lat]):
            print("Searching using bounding box...")
            return {
                "bbox": [min_lon, min_lat, max_lon, max_lat],
                "datetime": f"{start_date.strftime('%Y-%m-%d')}T00:00:00Z/{end_date.strftime('%Y-%m-%d')}T23:59:59Z",
                "collections": collection,
            }
        else:
            print("Searching using shapefile...")
            gdf = gpd.read_file(shapefile)
            geometry = gdf.geometry.iloc[0]
            return {
                "intersects": geometry,
                "datetime": f"{start_date.strftime('%Y-%m-%d')}T00:00:00Z/{end_date.strftime('%Y-%m-%d')}T23:59:59Z",
                "collections": collection,
            }

    def search(self, search_criteria: dict, print_results: bool = False):
        """
        Searches for products from DEA STAC using the STAC API.

        Args:
            search_criteria (dict): The search criteria for the DEA STAC Search.

        Returns:
            pystac.ItemCollection: The search results.

        Raises:
            ValueError: If no products are found for the selected area of interest.
        """
        client = Client.open(self.DEA_STAC_ENDPOINT)

        if search_criteria.get("intersects"):  # If shapefile is provided
            search = client.search(
                intersects=search_criteria["intersects"],
                datetime=search_criteria["datetime"],
                collections=search_criteria["collections"],
            )
        elif search_criteria.get("bbox"):  # If bounding box is provided
            search = client.search(
                collections=search_criteria["collections"],
                bbox=search_criteria["bbox"],
                datetime=search_criteria["datetime"],
            )

        # Check if any products are found
        if not list(search.items()):
            raise ValueError("No products found for selected area of interest.")

        if print_results:
            print(
                f"Found {len(list(search.items()))} products for selected area of interest."
            )
            for item in search.items():
                print(item.properties["title"])
            print("\n")
        return search


class TqdmUpTo(tqdm):
    """
    Class to display download progress bar.

    This class extends the `tqdm` class to provide a custom implementation
    for updating the progress bar during a download.

    Attributes:
        total (int): The total size of the file being downloaded.

    Methods:
        update_to(b, bsize, tsize): Updates the progress bar based on the
            number of bytes downloaded, the size of each chunk, and the
            total size of the file.
    """

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        Updates the progress bar based on the number of bytes downloaded,
        the size of each chunk, and the total size of the file.

        Args:
            b (int): The number of chunks downloaded.
            bsize (int): The size of each chunk.
            tsize (int): The total size of the file being downloaded.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


#####################################################################################################
#                                                                                                   #
#                                   Processor Utility Functions                                     #
#                                                                                                   #
#####################################################################################################
def xa_int16_to_float32(xarray: xa.DataArray, fill_value: int = -999) -> xa.DataArray:
    """
    Convert the data stored within an xarray.DataArray from int16 to float32

    Args:
        xarray (xarray.DataArray): The xarray to convert
        fill_value (int, optional): The value representing missing or invalid data. Defaults to -999.

    Returns:
        xarray.DataArray: The converted xarray
    """
    # Convert the data from int16 to float32
    xarray.data = xarray.data.astype(np.float32)

    # Replace _FillValue with np.nan
    xarray.data[xarray.data == fill_value] = np.nan
    xarray.attrs["_FillValue"] = np.nan

    # Rescale the data from 0-10000 to 0-1
    xarray.data[xarray.data > 1e10] = np.nan
    mask = xarray.data > 0
    xarray.data[mask] = xarray.data[mask] / 10000

    return xarray


def group_by_date(processed_dir: Path):
    """
    Groups directories in the given `processed_dir` by date.

    Args:
        processed_dir (Path): The directory containing the processed directories.

    Returns:
        list: A list of grouped directories.

    """
    grouped_paths = []

    processed_dirs = [dir for dir in processed_dir.iterdir() if dir.is_dir()]
    unique_dates = sorted({dir.name[25:35] for dir in processed_dirs})

    for unique_date in unique_dates:
        directories = tuple(
            f for f in processed_dir.glob(f"*{unique_date}*") if f.is_dir()
        )
        grouped_paths.append(directories)

    return grouped_paths


def merge_products(product_paths: tuple):
    """
    Merges satellite image rasters into a single raster.

    Args:
        product_paths (tuple): The paths to the products to be merged.
    Returns:
        None
    """
    if not product_paths:
        raise ValueError(
            "No products selected for merging. Please select at least one product."
        )

    input_dirs = sorted(product_paths)
    # Iterate through the directories and use .glob to find the relevant files
    nbr = [file for dir in input_dirs for file in dir.glob("*nbr.tif")]
    ndmi = [file for dir in input_dirs for file in dir.glob("*ndmi.tif")]
    ndvi = [file for dir in input_dirs for file in dir.glob("*ndvi.tif")]

    # Specifying the output directory
    output_dir = Path(input_dirs[0].parent, f"{input_dirs[0].name}_merged")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Merging the products
    bands = [nbr, ndmi, ndvi]

    for band in bands:
        rasters = [rasterio.open(file) for file in band]

        # Merge function returns a single mosaic array and the transformation info
        mosaic, out_trans = merge(rasters, nodata=np.nan, method="first")

        # Copy the metadata
        out_meta = rasters[0].meta.copy()

        # Update the metadata
        out_meta.update(
            {
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "crs": rasters[0].crs,
            }
        )

        # Determine the product name and custom metadata based on the band
        if band == nbr:
            product_name = f"{output_dir.name}_nbr.tif"
            custom_name = f"{output_dir.name}_nbr"
        elif band == ndmi:
            product_name = f"{output_dir.name}_ndmi.tif"
            custom_name = f"{output_dir.name}_ndmi"
        else:  # Assuming the last case is for ndvi_files
            product_name = f"{output_dir.name}_ndvi.tif"
            custom_name = f"{output_dir.name}_ndvi"

        # Write the mosaic and custom metadata to disk
        with rasterio.open(output_dir / product_name, "w", **out_meta) as dest:
            dest.write(mosaic)
            dest.update_tags(
                name=custom_name
            )  # This updates the TIFF TAGs with your custom 'name' metadata
    print(
        f"Merging Complete! - {[product_path.name for product_path in product_paths]}"
    )
