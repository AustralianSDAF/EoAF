"""
Gradient WebUI for Sentinel-2 satellite data search, download, processing and visualisation.
Author: Calvin Pang
Date: 2024-03-14
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from datetime import date
from pathlib import Path

import gradio as gr
import yaml
from matplotlib import cm
from webui_utils import Logger, display_logo, read_logs

from src.sentinel import SentinelDownloaderTool, process_products
from src.utils import merge_products, unique_session_id
from src.visualisation import product_visualiser

#####################################################################################################
#                                                                                                   #
#                                               Config                                              #
#                                                                                                   #
#####################################################################################################
with open(file="config.yaml", mode="r", encoding="UTF-8") as file:
    config = yaml.safe_load(file)

session_id = unique_session_id()
raw_dir = Path(config["SENTINEL_RAW"], session_id)
processed_dir = Path(config["SENTINEL_PROCESSED"], session_id)
visualisation_dir = Path(config["SENTINEL_VISUALISED"], session_id)

raw_dir.mkdir(parents=True, exist_ok=True)
processed_dir.mkdir(parents=True, exist_ok=True)
visualisation_dir.mkdir(parents=True, exist_ok=True)

sys.stdout = Logger("webui/output.log")


#####################################################################################################
#                                                                                                   #
#                                     Search & Download                                             #
#                                                                                                   #
#####################################################################################################


def sentinel_search(
    start_date: str,
    end_date: str,
    collection: str,
    min_lon: float,
    max_lon: float,
    min_lat: float,
    max_lat: float,
    shapefile_path: str,
):
    """
    Perform a search for Sentinel satellite data based on the specified parameters from Gradio components.

    Args:
        start_date (str): The start date of the search in ISO format (YYYY-MM-DD).
        end_date (str): The end date of the search in ISO format (YYYY-MM-DD).
        collection (str): The collection ID to search within.
        min_lon (float): The minimum longitude of the search area.
        max_lon (float): The maximum longitude of the search area.
        min_lat (float): The minimum latitude of the search area.
        max_lat (float): The maximum latitude of the search area.
        shapefile_path (str): The path to the shapefile defining the search area.

    Returns:
        None
    """
    # Convert the start and end dates from strings to date objects
    start_date = date.fromisoformat(start_date)
    end_date = date.fromisoformat(end_date)

    # Search for data
    client = SentinelDownloaderTool()

    search_params = client.search_parameters(
        start_date,
        end_date,
        collection,
        min_lon,
        max_lon,
        min_lat,
        max_lat,
        shapefile_path,
    )

    client.search(search_criteria=search_params, print_results=True)
    client.get_next_aquisition_date(search_criteria=search_params)
    return None


def sentinel_download(
    start_date: str,
    end_date: str,
    collection: str,
    min_lon: float,
    max_lon: float,
    min_lat: float,
    max_lat: float,
    shapefile_path: str,
    download_from_thredds: bool,
):
    """
    Downloads Sentinel satellite data based on the specified parameters from Gradio components.

    Args:
        start_date (str): The start date of the data to be downloaded in ISO format (YYYY-MM-DD).
        end_date (str): The end date of the data to be downloaded in ISO format (YYYY-MM-DD).
        collection (str): The collection of Sentinel data to be downloaded.
        min_lon (float): The minimum longitude of the area of interest.
        max_lon (float): The maximum longitude of the area of interest.
        min_lat (float): The minimum latitude of the area of interest.
        max_lat (float): The maximum latitude of the area of interest.
        shapefile_path (str): The path to the shapefile defining the area of interest.
        download_from_thredds (bool): Flag indicating whether to download data from THREDDS server.

    Returns:
        None
    """
    # Convert the start and end dates from strings to date objects
    start_date = date.fromisoformat(start_date)
    end_date = date.fromisoformat(end_date)

    # Search and download data
    client = SentinelDownloaderTool()

    search_params = client.search_parameters(
        start_date,
        end_date,
        collection,
        min_lon,
        max_lon,
        min_lat,
        max_lat,
        shapefile_path,
    )

    search_results = client.search(search_params, False)
    client.download(search_results, raw_dir, download_from_thredds)
    print("Download complete!")
    return None


#####################################################################################################
#                                                                                                   #
#                                     Processing                                                    #
#                                                                                                   #
#####################################################################################################


def sentinel_process(
    selected_products: list,
    select_all_products: bool,
    crop_to_aoi: bool,
    min_lon: float,
    max_lon: float,
    min_lat: float,
    max_lat: float,
    shapefile_path: str,
):
    """
    Process Sentinel satellite data based on the specified parameters from Gradio components.

    Args:
        selected_products (list): The list of selected products to be processed.
        select_all_products (bool): Flag indicating whether to process all products.
        crop_to_aoi (bool): Flag indicating whether to crop the products to the area of interest.
        min_lon (float): The minimum longitude of the area of interest.
        max_lon (float): The maximum longitude of the area of interest.
        min_lat (float): The minimum latitude of the area of interest.
        max_lat (float): The maximum latitude of the area of interest.
        shapefile_path (str): The path to the shapefile defining the area of interest.

    Returns:
        None
    """
    # Retrieving all downloaded Sentinel-2 directories
    sentinel_dirs = [dir for dir in list(raw_dir.glob("*")) if dir.is_dir()]

    # Convert every path in selected_products into a Path object
    selected_products = [Path(product) for product in selected_products]
    # Get the parent directory of each selected product
    selected_products = [product.parent for product in selected_products]
    selected_products = sorted(set(selected_products))
    selected_products = selected_products[1:]

    print("\n")
    process_products(
        product_paths=selected_products,
        sentinel_dirs=sentinel_dirs,
        sentinel_processed_dir=processed_dir,
        all_products=select_all_products,
        crop=crop_to_aoi,
        bbox=(min_lon, min_lat, max_lon, max_lat),
        shapefile=shapefile_path,
    )

    print("Processing complete.")
    return None


def sentinel_merge(selected_products: list):
    """
    Merge processed Sentinel satellite data based on the specified parameters from Gradio components.

    Args:
        selected_products (list): The list of selected products to be merged.

    Returns:
        None
    """
    # Convert every path in selected_products into a Path object
    selected_products = [Path(product) for product in selected_products]
    # Get the parent directory of each selected product
    selected_products = [product.parent for product in selected_products]
    selected_products = sorted(set(selected_products))
    selected_products = selected_products[1:]

    print("\n")
    print(f"Merging products: {selected_products}")
    merge_products(product_paths=selected_products)
    print("Merge complete!")
    return None


def sentinel_visualisation(
    selected_products: list,
    resample_selection: bool,
    resample_factor: float,
    percentile_selection: bool,
    colormap: cm,
    filename: str,
):
    # Convert every path in selected_products into a Path object
    selected_products = [Path(product) for product in selected_products]
    # Get the parent directory of each selected product
    selected_products = [product.parent for product in selected_products]
    selected_products = sorted(set(selected_products))
    selected_products = selected_products[1:]

    # Defining the colormap
    colormaps = {"Viridis": cm.viridis, "Turbo": cm.turbo, "Jet": cm.jet}
    colormap = colormaps[colormap]

    print("\nVisualising products: ", selected_products)
    ipymap = product_visualiser(
        product_paths=selected_products,
        product_type="DEA",
        apply_resample=resample_selection,
        resample_factor=resample_factor,
        use_percentile=percentile_selection,
        cmap=colormap,
    )

    output_path = Path(visualisation_dir, f"{filename}.html")
    ipymap.save(output_path)
    print("Visualisation saved to: ", output_path)
    return output_path


#####################################################################################################
#                                                                                                   #
#                                     Gradio WebUI                                                  #
#                                                                                                   #
#####################################################################################################
if __name__ == "__main__":
    with gr.Blocks() as demo:
        gr.HTML(display_logo("webui/ASDAF_Logo_White.png"))
        gr.Markdown("# Eyes on Australian Forests - Sentinel-2")
        gr.Textbox(label="Session ID", value=session_id, interactive=False)

        with gr.Tab("Search & Download"):
            gr.Markdown("## Search for Sentinel-2 Products")
            gr.Markdown(
                "Please enter your Sentinel-2 search parameters below. You may either define a bounding box or optionally upload a shapefile."
            )
            with gr.Row():
                start_date = gr.Textbox(
                    label="Start Date", value="2024-01-01", interactive=True
                )
                end_date = gr.Textbox(
                    label="End Date", value="2024-01-08", interactive=True
                )
            with gr.Row():
                min_lon = gr.Number(
                    label="Minimum Longitude", value=116.01897, interactive=True
                )
                max_lon = gr.Number(
                    label="Maximum Longitude", value=116.20093, interactive=True
                )
                min_lat = gr.Number(
                    label="Minimum Latitude", value=-32.30959, interactive=True
                )
                max_lat = gr.Number(
                    label="Maximum Latitude", value=-31.98176, interactive=True
                )
                shapefile_path = gr.FileExplorer(
                    label="Upload a Shapefile",
                    root_dir="./shapefiles",
                    glob="*.shp",
                    file_count="single",
                )
            with gr.Row():
                collection = gr.Radio(
                    label="Select a Sentinel-2 collection",
                    choices=["ga_s2am_ard_3", "ga_s2bm_ard_3"],
                    value="ga_s2am_ard_3",
                    interactive=True,
                )
                download_from_thredds = gr.Checkbox(
                    label="Download from NCI THREDDS?", value=False, interactive=True
                )

            search_button = gr.Button("Search")
            search_button.click(
                fn=sentinel_search,
                inputs=[
                    start_date,
                    end_date,
                    collection,
                    min_lon,
                    max_lon,
                    min_lat,
                    max_lat,
                    shapefile_path,
                ],
                outputs=None,
            )

            gr.Markdown("## Download Sentinel-2 Products")
            gr.Markdown(
                'Please confirm your search results in the Terminal Output below and click "Download" to proceed. Please be patient.'
            )

            download_button = gr.Button("Download")
            download_button.click(
                fn=sentinel_download,
                inputs=[
                    start_date,
                    end_date,
                    collection,
                    min_lon,
                    max_lon,
                    min_lat,
                    max_lat,
                    shapefile_path,
                    download_from_thredds,
                ],
                outputs=None,
            )

        with gr.Tab("Process"):
            gr.Markdown("## NDVI, NDMI and NBR Processing")
            with gr.Column():
                gr.Markdown(
                    "Please select your raw Sentinel-2 products to process. You may select all products by checking the box below."
                )
                selected_raw_products = gr.FileExplorer(
                    label="Select products(s) to process",
                    root_dir=raw_dir,
                    every=5,
                )
                select_all_products = gr.Checkbox(
                    label="Select all products?", value=False
                )

            gr.Markdown(
                "You may also crop the products to a specified area of interest based on a bounding box or shapefile."
            )
            with gr.Row():
                crop_to_aoi = gr.Checkbox(
                    label="Crop to Area of Interest?", value=False
                )
                min_lon_1 = gr.Number(
                    label="Minimum Longitude", value=116.01897, interactive=True
                )
                max_lon_1 = gr.Number(
                    label="Maximum Longitude", value=116.20093, interactive=True
                )
                min_lat_1 = gr.Number(
                    label="Minimum Latitude", value=-32.30959, interactive=True
                )
                max_lat_1 = gr.Number(
                    label="Maximum Latitude", value=-31.98176, interactive=True
                )
                shapefile_path_1 = gr.FileExplorer(
                    label="Upload a Shapefile",
                    root_dir="./shapefiles",
                    glob="*.shp",
                    file_count="single",
                )

            process_button = gr.Button("Process Products")
            process_button.click(
                fn=sentinel_process,
                inputs=[
                    selected_raw_products,
                    select_all_products,
                    crop_to_aoi,
                    min_lon_1,
                    max_lon_1,
                    min_lat_1,
                    max_lat_1,
                    shapefile_path_1,
                ],
                outputs=None,
            )

            with gr.Accordion(label="Merge Processed Products (Optional)", open=True):
                gr.Markdown(
                    "You may merge multiple processed Sentinel-2 products into a single product. Please select the processed products to merge."
                )
                selected_processed_products = gr.FileExplorer(
                    label="Select products(s) to merge", root_dir=processed_dir
                )
                merge_button = gr.Button("Merge Products")
                merge_button.click(
                    fn=sentinel_merge,
                    inputs=selected_processed_products,
                    outputs=None,
                )

        with gr.Tab("Visualisation"):
            gr.Markdown("## Visualisation")
            gr.Markdown(
                "Please select your processed Sentinel-2 products to visualise. It will be saved as a HTML file for you to download and open via a web browser."
            )
            with gr.Row():
                selected_processed_products_1 = gr.FileExplorer(
                    label="Select products(s) to visualise",
                    root_dir=processed_dir,
                    every=5,
                )

            gr.Markdown("Please select visualisation options below.")
            with gr.Row():
                resample_selection = gr.Checkbox(label="Apply a downscaling factor to reduce the spatial resolution of product for rapid display?", value=True)
                resample_factor = gr.Number(label="Downscaling Factor", value=6)
                colormap_selection = gr.Radio(
                    label="Select a colormap",
                    choices=["Viridis", "Turbo", "Jet"],
                    value="Jet",
                    interactive=True,
                )
                percentile_selection = gr.Checkbox(
                    label="Apply percentiles to potentially enhance contrast and improve image display?", value=False
                )

            with gr.Row():
                filename = gr.Textbox(
                    label="Enter a filename",
                    value="Test_Visualisation",
                    interactive=True,
                )

            visualise_button = gr.Button("Generate Visualisation")
            visualise_button.click(
                fn=sentinel_visualisation,
                inputs=[
                    selected_processed_products_1,
                    resample_selection,
                    resample_factor,
                    percentile_selection,
                    colormap_selection,
                    filename,
                ],
                outputs=gr.DownloadButton(label="Download Visualisation"),
            )
        # Terminal
        logs = gr.Textbox(label="Terminal Output")
        demo.load(read_logs, None, logs, every=1)
    demo.launch()
