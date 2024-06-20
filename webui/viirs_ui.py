"""
Gradio WebUI for VIIRS data search, download, processing and visualisation.
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

from src.utils import unique_session_id
from src.viirs import APPEEARSDownloader, process_products
from src.visualisation import product_visualiser

#####################################################################################################
#                                                                                                   #
#                                               Config                                              #
#                                                                                                   #
#####################################################################################################
with open(file="config.yaml", mode="r", encoding="UTF-8") as file:
    config = yaml.safe_load(file)

session_id = unique_session_id()
raw_dir = Path(config["VIIRS_RAW"], session_id)
processed_dir = Path(config["VIIRS_PROCESSED"], session_id)
visualisation_dir = Path(config["VIIRS_VISUALISED"], session_id)

raw_dir.mkdir(parents=True, exist_ok=True)
processed_dir.mkdir(parents=True, exist_ok=True)
visualisation_dir.mkdir(parents=True, exist_ok=True)

sys.stdout = Logger("webui/output.log")

#####################################################################################################
#                                                                                                   #
#                                     Search & Download                                             #
#                                                                                                   #
#####################################################################################################


def appeears_submit_task(
    username: str,
    password: str,
    start_date: str,
    end_date: str,
    min_lon: float,
    max_lon: float,
    min_lat: float,
    max_lat: float,
    shapefile_path: str,
    product: str,
    task_name: str,
):
    """
    Submits a task to AppEEARS for downloading VIIRS products within a specified region and time range.

    Args:
        username (str): The username for logging into AppEEARS.
        password (str): The password for logging into AppEEARS.
        start_date (str): The start date of the time range in ISO format (YYYY-MM-DD).
        end_date (str): The end date of the time range in ISO format (YYYY-MM-DD).
        min_lon (float): The minimum longitude of the region.
        max_lon (float): The maximum longitude of the region.
        min_lat (float): The minimum latitude of the region.
        max_lat (float): The maximum latitude of the region.
        shapefile_path (str): The file path to the shapefile defining the region.
        product (str): The type of VIIRS product to download.
        task_name (str): The name of the AppEEARS task.

    Returns:
        None
    """
    # Login to AppEEARS
    client = APPEEARSDownloader(username=username, password=password)
    token = client.get_token()
    print("Successfully logged onto AppEEARS")

    # Convert start and end dates to date objects
    start_date = date.fromisoformat(start_date)
    end_date = date.fromisoformat(end_date)

    print(f"Product: {product}")
    # Generate an AppEEARS task
    task = client.generate_task(
        task_name=task_name,
        min_lat=min_lat,
        min_lon=min_lon,
        max_lat=max_lat,
        max_lon=max_lon,
        shapefile=shapefile_path,
        start_date=start_date,
        end_date=end_date,
        product_type=product,
    )

    # Submit the task to AppEEARS
    task_id = client.submit_task(task, token)
    print(f"Successfully submitted task {task_id} for VIIRS products\n")


def appeears_download_task(
    username: str, password: str, task_id: str, task_name: str, product: str
):
    """
    Download VIIRS products from AppEEARS.

    Args:
        username: Username for AppEEARS login.
        password: Password for AppEEARS login.
        task_id: ID of the download task.
        task_name: Name of the download task.
        product: Type of product to download.

    Returns:
        None
    """
    if not task_id:
        raise ValueError("Please enter a valid Task ID")

    # Login to AppEEARS
    client = APPEEARSDownloader(username=username, password=password)
    token = client.get_token()
    print("Successfully logged onto AppEEARS")

    # Check the status of the task
    if client.check_task(task_id, token, 120):
        client.download(
            task_name=task_name,
            task_id=task_id,
            product_type=product,
            download_dir=raw_dir,
            token=token,
        )
        print("Successfully downloaded VIIRS products")

    client.logout(token)
    print("Successfully logged out of AppEEARS\n")
    return None


#####################################################################################################
#                                                                                                   #
#                                     Processing                                                    #
#                                                                                                   #
#####################################################################################################


def viirs_process(selected_products: list, select_all_products: bool):
    """
    Process the selected VIIRS products.

    Args:
        selected_products (list): A list of selected product paths.
        select_all_products (bool): A flag indicating whether to process all products.

    Returns:
        None
    """
    viirs_dirs = [
        dir
        for dir in list(Path(raw_dir, "VNP09GA.001").glob("*"))
        + list(Path(raw_dir, "VNP09GA.002").glob("*"))
        if dir.is_dir()
    ]

    print(selected_products)
    # Convert every path in selected_products into a Path object
    selected_products = [Path(product) for product in selected_products]
    # Get the parent directory of each selected product
    selected_products = [product.parent for product in selected_products]
    # Remove duplicates
    selected_products = list(set(selected_products))
    # Remove the first element of the list (which is the raw_dir)
    selected_products.sort()
    selected_products = selected_products[1:]
    print("\nProcessing products: ", selected_products)

    process_products(
        product_paths=selected_products,
        viirs_dirs=viirs_dirs,
        viirs_processed_dir=processed_dir,
        all_products=select_all_products,
    )
    return None


def viirs_visualisation(
    selected_products: list,
    percentile_selection: bool,
    colormap: cm,
    filename: str,
):
    """
    Visualize VIIRS products on a map.

    Args:
        selected_products: List of paths to selected products.
        percentile_selection: Selection of percentile for visualization.
        colormap: Colormap for visualization.
        filename: Name of the output file.

    Returns:
        Path: Path to the saved visualization file.
    """
    # Convert every path in selected_products into a Path object
    selected_products = [Path(product) for product in selected_products]
    # Get the parent directory of each selected product
    selected_products = [product.parent for product in selected_products]
    # Remove duplicates
    selected_products = list(set(selected_products))
    # Remove the first element of the list (which is the raw_dir)
    selected_products.sort()
    selected_products = selected_products[1:]

    # Defining the colormap
    colormaps = {"Viridis": cm.viridis, "Turbo": cm.turbo, "Jet": cm.jet}
    colormap = colormaps[colormap]

    print("\nVisualising products: ", selected_products)
    ipymap = product_visualiser(
        product_paths=selected_products,
        product_type="VIIRS",
        apply_resample=False,
        resample_factor=1,
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
        gr.Markdown("# Eyes on Australian Forests - VIIRS")

        with gr.Tab("Search & Download"):
            gr.Markdown("## Submit a request for VIIRS Products")

            gr.Markdown("Please enter your AppEEARS Login Credentials")
            with gr.Row():
                username = gr.Textbox(label="Username", type="text")
                password = gr.Textbox(label="Password", type="password")

            gr.Markdown(
                "Please enter your VIIRS search parameters below. You may either define a bounding box or optionally upload a shapefile."
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
                product = gr.Radio(
                    label="Select a VIIRS product",
                    choices=["VNP09GA.001", "VNP09GA.002"],
                    value="VNP09GA.001",
                    interactive=True,
                )
                task_name = gr.Textbox(
                    label="Task Name", value=session_id, interactive=False
                )

            submit_task = gr.Button("Submit Task")
            submit_task.click(
                fn=appeears_submit_task,
                inputs=[
                    username,
                    password,
                    start_date,
                    end_date,
                    min_lon,
                    max_lon,
                    min_lat,
                    max_lat,
                    shapefile_path,
                    product,
                    task_name,
                ],
            )

            gr.Markdown(
                "Please enter your generated Task ID, found in the Terminal Output below."
            )
            gr.Markdown(
                "Depending on your search parameters, AppEEARS may take a while to process a request. Please monitor the progress in the Terminal Output below."
            )
            task_id = gr.Textbox(label="Task ID", placeholder="Enter Task ID")

            download_task = gr.Button("Check & Download Task")
            download_task.click(
                fn=appeears_download_task,
                inputs=[username, password, task_id, task_name, product],
            )

        with gr.Tab("Process"):
            gr.Markdown("## NDVI, NDMI and NBR Processing")
            with gr.Column():
                gr.Markdown(
                    "Please select your raw VIIRS products to process. You may select all products by checking the box below."
                )
                selected_raw_products = gr.FileExplorer(
                    label="Select product(s) to process",
                    root_dir=raw_dir,
                    glob="*",
                )
                select_all_products = gr.Checkbox(
                    label="Select all products?", value=False
                )

            process_button = gr.Button("Process Products")
            process_button.click(
                fn=viirs_process,
                inputs=[selected_raw_products, select_all_products],
            )

        with gr.Tab("Visualisation"):
            gr.Markdown("## Visualisation")
            gr.Markdown(
                "Please select your processed VIIRS products to visualise. It will be saved as a HTML file for you to download and open via a web browser."
            )
            with gr.Row():
                selected_processed_products_1 = gr.FileExplorer(
                    label="Select products(s) to visualise",
                    root_dir=processed_dir,
                    every=5,
                    glob="*",
                )

            gr.Markdown("Please select visualisation options below.")
            with gr.Row():
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
                fn=viirs_visualisation,
                inputs=[
                    selected_processed_products_1,
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
