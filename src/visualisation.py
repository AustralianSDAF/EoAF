"""
Module containing functions to visualise NDVI, NDMI and NBR from netCDF files using ipyleaflet.
Author: Calvin Pang & Leigh Tyers
Date: 2024-01-31
"""

from base64 import b64encode
from io import BytesIO
from pathlib import Path

import ipyleaflet
import numpy as np
import rioxarray as rio
import xarray as xa
from ipyleaflet import (
    ImageOverlay,
    LayerGroup,
    LayersControl,
    Map,
    basemaps,
)
from ipywidgets import Layout
from matplotlib import cm
from PIL import Image
from rasterio.enums import Resampling


def load_processed_sentinel_data(
    input_file: Path,
) -> tuple[xa.DataArray, xa.DataArray, xa.DataArray]:
    """
    Load the processed Sentinel-2 data and set CRS, _FillValue and NaNs.
    Deprecated: Use load_processed_DEA_data instead.

    Args:
        input_file (Path): The path to the netCDF file.
    Returns:
        xa_ds["ndvi"] (xa.DataArray): xarray DataArray.
        xa_ds["ndmi"] (xa.DataArray): xarray DataArray.
        xa_ds["nbr"] (xa.DataArray): xarray DataArray.
    """
    # Open the netCDF file
    xa_ds = rio.open_rasterio(input_file)

    # Set CRS, _FillValue and NaNs
    bands = ["ndvi", "ndmi", "nbr"]
    for band in bands:
        xa_ds[band].attrs["name"] = f"{xa_ds.attrs['condensed_name']}_{band}"
        xa_ds[band].attrs["crs"] = xa_ds.attrs["crs"].lower()
        xa_ds[band].attrs["_FillValue"] = np.nan
        xa_ds[band].data[xa_ds[band].data > 1e10] = np.nan

    return xa_ds["ndvi"], xa_ds["ndmi"], xa_ds["nbr"]


def load_processed_DEA_data(
    input_dir: Path,
) -> tuple[xa.DataArray, xa.DataArray, xa.DataArray]:
    """
    Load the processed Sentinel-2 and Landsat 8/9 data.
    Args:
        input_dir (Path): The path to the directory containing the processed data.
    Returns:
        ndvi (xa.DataArray): xarray DataArray.
        ndmi (xa.DataArray): xarray DataArray.
        nbr (xa.DataArray): xarray DataArray.
    """
    ndvi = rio.open_rasterio(Path(input_dir, f"{input_dir.stem}_ndvi.tif"))
    ndmi = rio.open_rasterio(Path(input_dir, f"{input_dir.stem}_ndmi.tif"))
    nbr = rio.open_rasterio(Path(input_dir, f"{input_dir.stem}_nbr.tif"))

    return ndvi, ndmi, nbr


def load_processed_VIIRS_data(
    input_dir: Path,
) -> tuple[xa.DataArray, xa.DataArray, xa.DataArray]:
    """
    Load the processed VIIRS data.
    Args:
        input_dir (Path): The path to the directory containing the processed data.
    Returns:
        ndvi (xa.DataArray): xarray DataArray.
        ndmi (xa.DataArray): xarray DataArray.
        nbr (xa.DataArray): xarray DataArray.
    """
    ndvi = rio.open_rasterio(Path(input_dir, "ndvi.tif"))
    ndmi = rio.open_rasterio(Path(input_dir, "ndmi.tif"))
    nbr = rio.open_rasterio(Path(input_dir, "nbr.tif"))

    return ndvi, ndmi, nbr


def resample_xarray(
    input_xda: xa.DataArray, downscale_factor: int = 6, crs: str = None
) -> xa.DataArray:
    """
    Resamples xarray by the downscale factor, optionally changing the CRS.

    Args:
        input_array (xa.DataArray): The input xarray to be resampled.
        downscale_factor (int, optional): The factor by which to downscale the dataset. Defaults to 6.
        crs (str, optional): The coordinate reference system (CRS) to use for resampling. Defaults to None.

    Returns:
        xa.DataArray: The resampled xarray.
    """
    # Check if the CRS is set
    if crs is None:
        crs = input_xda.rio.crs

    # Calculate the new width and height
    new_width = int(input_xda.rio.width / downscale_factor)
    new_height = int(input_xda.rio.height / downscale_factor)

    # Resample the xarray
    resampled_xa = input_xda.rio.reproject(
        dst_crs=crs, shape=(new_height, new_width), resampling=Resampling.bilinear
    )

    # Set the CRS
    resampled_xa.attrs["crs"] = crs

    return resampled_xa


def load_image_from_xda_chunks(
    xda: xa.DataArray,
    cmap: cm = None,
    min_pc: int = 1,
    max_pc: int = 98,
    use_percentile: bool = True,
    chunk_size: int = -1,
) -> list:
    """
    Load an array from an xarray DataArray and return as URI string inputs for ipyleaflet.ImageOverlay.

    Args:
        xda (xa.DataArray): The xarray DataArray to load the image from.
        cmap (cm, optional): The colormap to use. Defaults to None.
        min_pc (int, optional): The minimum percentile to use for the colormap. Defaults to 1.
        max_pc (int, optional): The maximum percentile to use for the colormap. Defaults to 98.
        use_percentile (bool, optional): Whether to use percentiles for the colormap. Defaults to True.
        chunk_size (int, optional): The chunk size to use for loading the image. Defaults to -1.
    """
    # Default colormap options
    if cmap is None:
        cmap = cm.turbo

    # Default chunk size
    if chunk_size == -1:
        chunk_size = int(1e99)

    # Initialise output_url
    output_uri = []

    # Get Normalisation Factors
    if use_percentile:
        min_val, max_val = np.percentile(
            xda.data.flatten()[~np.isnan(xda.data.flatten())], (min_pc, max_pc)
        )
    else:
        min_val, max_val = np.nanmin(xda.data), np.nanmax(xda.data)

    # Loop through chunks
    for xi in range(xda.x.size // chunk_size + 1):
        x = xi * chunk_size

        for yi in range(xda.y.size // chunk_size + 1):
            y = yi * chunk_size

            if (xda.x.data.size - x <= 1) or (xda.y.data.size - y <= 1):
                continue

            # Normalise the array to between 0 & 1
            norm_array = np.array(
                xda[0, y : y + chunk_size, x : x + chunk_size], dtype=np.float32
            )
            w = ~np.isnan(norm_array)
            norm_array[w] = (norm_array[w] - min_val) / (max_val - min_val)
            norm_array[w] = np.minimum(norm_array[w], 1)
            norm_array[w] = np.maximum(norm_array[w], 0)

            img = Image.fromarray(
                np.uint8(cmap(norm_array[...]) * 255.0), "RGBA"
            ).convert("RGB")

            # Save the image to a buffer
            f = BytesIO()
            img.save(f, format="png", compress_level=0)
            del img

            data = b64encode(f.getvalue()).decode("ascii")
            img_uri = f"data:image/png;base64,{data}"
            output_uri.append(img_uri)
    return output_uri


def get_bounds_xda_chunks(
    xda: xa.DataArray,
    x_offset: int = 0,
    y_offset: int = 0,
    x_increase: int = 0,
    y_increase: int = 0,
    chunk_size: int = -1,
) -> list[np.array]:
    """
    Retrieves the boundaries of a xArray.Dataset in Lon, Lat (Requires an EPSG CRS)

    Args:
        xda (xa.DataArray): The xArray.Dataset to get the bounds of.
        x_offset (int, optional): The x offset to use for the image. Defaults to 0.
        y_offset (int, optional): The y offset to use for the image. Defaults to 0.
        x_increase (int, optional): The x increase to use for the image. Defaults to 0.
        y_increase (int, optional): The y increase to use for the image. Defaults to 0.
        chunk_size (int, optional): The chunk size to use for loading the image. Defaults to -1.

    Returns:
        list (np.array): A list of the boundaries of the dataset.
    """
    # Default chunk size
    if chunk_size == -1:
        chunk_size = int(1e99)

    # Retrieve the bounds
    bounds = []

    for xi in range(xda.x.size // chunk_size + 1):
        xv = xi * chunk_size
        for yi in range(xda.y.size // chunk_size + 1):
            yv = yi * chunk_size

            if xda.x.data.size - xv <= 1 or xda.y.data.size - yv <= 1:
                continue

            xd = xda[0, yv : yv + chunk_size, xv : xv + chunk_size]

            min_x = xd.x.data.min() + x_offset - x_increase
            max_y = xd.y.data.min() + y_offset - y_increase
            max_x = xd.x.data.max() + x_offset + x_increase
            min_y = xd.y.data.max() + y_offset + y_increase

            poly = np.array(
                [[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]]
            )

            min_x, min_y = np.min(poly[:, 0]), np.min(poly[:, 1])
            max_x, max_y = np.max(poly[:, 0]), np.max(poly[:, 1])
            bounds.append(np.array([[min_x, min_y], [max_x, max_y]]))
    return bounds


def create_map_xda(
    xda_list: list,
    cmap: cm = None,
    min_pc: int = 1,
    max_pc: int = 98,
    use_percentile: bool = True,
    chunk_size: int = -1,
    x_offset: int = 0,
    y_offset: int = 0,
    pixel_sizes: int = 0,
    zoom: int = 11,
) -> ipyleaflet.Map:
    """
    Generates an image from an xarray. Applies a percentile clip to the image.

    Args:
        xda_list (list): A list of xarray DataArrays to be displayed.
        cmap (cm, optional): The colormap to use. Defaults to None.
        min_pc (int, optional): The minimum percentile to use for the colormap. Defaults to 1.
        max_pc (int, optional): The maximum percentile to use for the colormap. Defaults to 98.
        use_percentile (bool, optional): Whether to use percentiles for the colormap. Defaults to True.
        chunk_size (int, optional): The chunk size to use for loading the image. Defaults to -1.
        x_offset (int, optional): The x offset to use for the image. Defaults to 0.
        y_offset (int, optional): The y offset to use for the image. Defaults to 0.
        pixel_sizes (float, optional): The pixel size to use for the image. Defaults to 10.
        zoom (int, optional): The zoom level to use for the image. Defaults to 11.

    Returns (ipyleaflet.Map): A map containing the image.
    """
    if not isinstance(xda_list, list):
        xda_list = [xda_list]

    layout = Layout(width="80%", height="800px")

    # Load ImageOverlay
    img_layers = []
    for j, xda in enumerate(xda_list):
        # Type Check & Load Image
        if type(xda) is xa.DataArray:
            img_uri_data = load_image_from_xda_chunks(
                xda=xda,
                cmap=cmap,
                chunk_size=chunk_size,
                min_pc=min_pc,
                max_pc=max_pc,
                use_percentile=use_percentile,
            )
        else:
            raise TypeError("Input is not an xa.DataArray")

        # Quick Name Check
        layer_name = f'{xda.attrs["name"]}' if "name" in xda.attrs else f"Layer: {j}"

        # Get Bounds for Display
        if isinstance(pixel_sizes, list):
            x_increase = pixel_sizes[j] / 2
            y_increase = pixel_sizes[j] / 2
        else:
            x_increase = pixel_sizes / 2
            y_increase = pixel_sizes / 2

        bounds = get_bounds_xda_chunks(
            xda,
            chunk_size=chunk_size,
            x_offset=x_offset,
            y_offset=y_offset,
            x_increase=x_increase,
            y_increase=y_increase,
        )
        bounds = [[(y, x) for (x, y) in bound_i] for bound_i in bounds]

        # Create ImageOverlay
        imgs = [
            ImageOverlay(url=img_uri_data[i], bounds=bounds[i])
            for i in range(len(img_uri_data))
        ]
        img_layers.append(LayerGroup(layers=imgs, name=layer_name))

    # Create Map
    centers = [list(np.mean(bound_i, axis=0)) for bound_i in bounds]
    center = np.mean(centers, axis=0)
    center = (center[0], center[1])

    m = Map(center=center, basemap=basemaps.Esri.WorldImagery, zoom=zoom, layout=layout)

    # Add Layers
    for layer in img_layers:
        m.add_layer(layer)
    control = LayersControl(position="topright")
    m.add_control(control)

    # # Add Measure Control
    # measure = MeasureControl(
    #     position="bottomleft", active_color="orange", primary_length_unit="meters"
    # )
    # m.add_control(measure)

    return m


def product_visualiser(
    product_paths: tuple,
    product_type: str,
    apply_resample: bool,
    resample_factor: float,
    use_percentile: bool,
    cmap: cm = None,
) -> ipyleaflet.Map:
    """
    Visualise a processed satellite image product.

    Args:
        product_path (Path): The path to the processed product.
        product_type (str): The type of product to visualise.
        apply_resample (bool): Whether to resample the data.
        resample_factor (float): The factor by which to resample the data.
        use_percentile (bool): Whether to use percentiles for the colormap.
        cmap (cm, optional): The matplotlb colormap to use.

    Returns:
        ipyleaflet.Map: The map containing the visualisation.
    """
    # Check if a product has been selected
    if len(product_paths) == 0:
        raise ValueError("Please select a product to visualise.")

    # Convert the product_paths to a list
    products = list(product_paths)

    layers = []

    for product in products:
        # Load the data
        if product_type == "DEA":
            ndvi, ndmi, nbr = load_processed_DEA_data(product)
        elif product_type == "VIIRS":
            ndvi, ndmi, nbr = load_processed_VIIRS_data(product)
        else:
            raise ValueError("Invalid product type")

        # Resample the data
        if apply_resample:
            ndvi = resample_xarray(ndvi, resample_factor, "EPSG:4326")
            ndmi = resample_xarray(ndmi, resample_factor, "EPSG:4326")
            nbr = resample_xarray(nbr, resample_factor, "EPSG:4326")

        layers.extend((ndvi, ndmi, nbr))

    return create_map_xda(
        xda_list=layers,
        cmap=cmap,
        min_pc=1,
        max_pc=98,
        use_percentile=use_percentile,
        chunk_size=-1,
        x_offset=0,
        y_offset=0,
        pixel_sizes=0,
        zoom=10,
    )
