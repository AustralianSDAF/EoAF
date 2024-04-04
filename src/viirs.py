import getpass
import sys
import time
from datetime import date, datetime
from pathlib import Path

import geopandas as gpd
import requests
import rioxarray as rio
import typer
from tqdm.auto import tqdm

# Add the parent directory to the path so that the module can be run from the command line
try:
    from utils import xa_int16_to_float32
except ModuleNotFoundError:
    sys.path.append("..")
    from src.utils import (
        xa_int16_to_float32,
    )


#####################################################################################################
#                                                                                                   #
#                                   Download Functions                                              #
#                                                                                                   #
#####################################################################################################
class APPEEARSDownloader:
    """
    Class for downloading data from appEEARS API.
    """

    def __init__(self, username: str, password: str):
        """
        Initialises an instance of the APPEEARSDownloader class.

        Args:
            username (str): Username for appEEARS.
            password (str): Password for appEEARS.

        Returns:
            None
        """
        self.api = "https://appeears.earthdatacloud.nasa.gov/api"
        self.username = username
        self.password = password

    def get_token(self) -> str:
        """
        Retrieves the authentication token from appEEARS API.

        Args:
            None

        Returns:
            str: The authentication token.
        """
        token_response = requests.post(
            f"{self.api}/login", auth=(self.username, self.password), timeout=60
        )

        if token_response.status_code != 200:
            raise requests.exceptions.HTTPError(
                f"Error getting token: {token_response.status_code}"
            )
        print("Logged into APPEEARS...")
        return token_response.json()["token"]

    def generate_task(
        self,
        task_name: str,
        start_date: date,
        end_date: date,
        product_type: str,
        min_lon: float = None,
        max_lon: float = None,
        min_lat: float = None,
        max_lat: float = None,
        shapefile: Path = None,
    ) -> dict:
        """
        Generates a task to download data from APPEEARS API.

        Args:
            task_name (str): Name of the task.
            start_date (date): Start date of the data.
            end_date (date): End date of the data.
            product_type (str): Type of product to download.
            min_lon (float, optional): Minimum longitude.
            max_lon (float, optional): Maximum longitude.
            min_lat (float, optional): Minimum latitude.
            max_lat (float, optional): Maximum latitude.
            shapefile (Path, optional): Path to the shape file.
        Returns:
            dict: The task to be submitted to APPEEARS.
        """
        # Pre-defined layers to download
        if product_type == "VNP09GA.001":
            layers = [
                {"layer": "VNP_Grid_1km_2D_SurfReflect_M5_1", "product": "VNP09GA.001"},
                {"layer": "VNP_Grid_1km_2D_SurfReflect_M7_1", "product": "VNP09GA.001"},
                {
                    "layer": "VNP_Grid_1km_2D_SurfReflect_M10_1",
                    "product": "VNP09GA.001",
                },
                {
                    "layer": "VNP_Grid_1km_2D_SurfReflect_M11_1",
                    "product": "VNP09GA.001",
                },
                {
                    "layer": "VNP_Grid_500m_2D_SurfReflect_I2_1",
                    "product": "VNP09GA.001",
                },
            ]
        elif product_type == "VNP09GA.002":
            layers = [
                {"layer": "SurfReflect_M5_1", "product": "VNP09GA.002"},
                {"layer": "SurfReflect_M7_1", "product": "VNP09GA.002"},
                {"layer": "SurfReflect_M10_1", "product": "VNP09GA.002"},
                {"layer": "SurfReflect_M11_1", "product": "VNP09GA.002"},
                {"layer": "SurfReflect_I2_1", "product": "VNP09GA.002"},
            ]

        # Checking if the bounding box or shapefile is provided
        if not any([min_lon, max_lon, min_lat, max_lat, shapefile]):
            raise ValueError("Please provide either a bounding box or a shapefile.")

        # Defining the area to download
        if not shapefile and all([min_lon, max_lon, min_lat, max_lat]):
            print("Generating task from bounding box...")
            polygon = [
                [min_lon, min_lat],
                [min_lon, max_lat],
                [max_lon, max_lat],
                [max_lon, min_lat],
                [min_lon, min_lat],
            ]
        else:
            print("Generating task from shape file...")
            gdf = gpd.read_file(shapefile)
            geometry = gdf.geometry.iloc[0]
            polygon = list(zip(geometry.exterior.xy[0], geometry.exterior.xy[1]))

        geo = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [polygon],
                    },
                }
            ],
        }

        # Returning the task details to be submitted
        return {
            "task_type": "area",
            "task_name": task_name,
            "params": {
                "dates": [
                    {
                        "startDate": start_date.strftime("%m-%d-%Y"),
                        "endDate": end_date.strftime("%m-%d-%Y"),
                    }
                ],
                "layers": layers,
                "output": {"format": {"type": "netcdf4"}, "projection": "geographic"},
                "geo": geo,
            },
        }

    def submit_task(self, task: dict, token: str) -> str:
        """
        Submits a task to appEEARS API.

        Args:
            task (dict): The task to be submitted.
            token (str): The authentication token.

        Returns:
            str: The task ID of the submitted task.
        """
        response = requests.post(
            f"{self.api}/task",
            json=task,
            headers={"Authorization": f"Bearer {token}"},
            timeout=60,
        ).json()

        if response.get("task_id") is None:
            raise requests.exceptions.HTTPError(
                f"Error submitting task: {response['message']}"
            )

        print(
            f"Task: {task['task_name']} submitted to appEEARS. \nTask ID: {response['task_id']}"
        )
        return response["task_id"]

    def download(
        self,
        task_name: str,
        task_id: str,
        product_type: str,
        download_dir: Path,
        token: str,
    ):
        """
        Downloads the files associated with a task from appEEARS API.

        Args:
            task_name (str): Name of the task.
            task_id (str): ID of the task.
            product_type (str): Type of product.
            download_dir (Path): Directory to save the files.
            token (str): The authentication token.

        Returns:
            None
        """
        # Creating the directory to save the files
        save_dir = Path(download_dir, product_type, task_name)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Bundling the files
        bundle = requests.get(
            f"{self.api}/bundle/{task_id}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=60,
        ).json()
        files = {f["file_id"]: f["file_name"] for f in bundle["files"]}

        # Downloading the files
        print(f"\nDownloading {task_name} from APPEEARS")
        for file in tqdm(files):
            downloader = requests.get(
                f"{self.api}/bundle/{task_id}/{file}",
                headers={"Authorization": f"Bearer {token}"},
                stream=True,
                allow_redirects=True,
                timeout=60,
            )

            file_path = Path(save_dir, files[file])

            with open(file_path, "wb") as f:
                for data in downloader.iter_content(chunk_size=8192):
                    f.write(data)

    def check_task(
        self,
        task_id: str,
        token: str,
        timeout: int = 120,
    ) -> bool:
        """
         Checks the status of a task and downloads the files when the task is completed.

        Args:
            task_id (str): The ID of the task.
            token (str): The authentication token.
            timeout (int, optional): The time interval (in seconds) between status checks. Defaults to 120.

        Returns:
            None
        """
        # Checking the status of the submitted task
        status = requests.get(
            f"{self.api}/task/{task_id}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=60,
        ).json()

        print(
            f"\nCurrent Time: {time.asctime()}\nTask ID: {task_id}\nStatus: {status['status']}"
        )

        # Waiting for the task to be completed
        while status["status"] != "done":
            time.sleep(timeout)
            status = requests.get(
                f"{self.api}/task/{task_id}",
                headers={"Authorization": f"Bearer {token}"},
                timeout=60,
            ).json()
            print(
                f"\nCurrent Time: {time.asctime()}\nTask ID: {task_id}\nStatus: {status['status']}"
            )
        return True

    def logout(self, token: str):
        """
        Logs out of appEEARS API.

        Args:
            token (str): The authentication token.
        """
        response = requests.post(
            f"{self.api}/logout",
            headers={"Authorization": f"Bearer {token}"},
            timeout=60,
        )

        if response.status_code != 204:
            raise requests.exceptions.HTTPError(
                f"Error logging out: {response.status_code}"
            )
        else:
            print("Logged out of APPEEARS...")


#####################################################################################################
#                                                                                                   #
#                                   Processing Functions                                            #
#                                                                                                   #
#####################################################################################################


def load_product(input_dir: Path) -> tuple[str, dict]:
    """
    Loads a VIIRS product from the input directory.

    Args:
        input_dir (Path): The directory containing the VIIRS product.

    Returns:
        product_title (str): The title of the VIIRS product.
        bands (dict): A dictionary containing the required bands of the VIIRS product.
    """
    if input_dir.parent.name == "VNP09GA.001":
        print(f"Loading VNP09GA.001 product: {input_dir.name}")
        I_Bands = rio.open_rasterio(Path(input_dir, "VNP09GA.001_500m_aid0001.nc"))
        M_Bands = rio.open_rasterio(Path(input_dir, "VNP09GA.001_1km_aid0001.nc"))

        bands = {
            "red": M_Bands["VNP_Grid_1km_2D_SurfReflect_M5_1"],
            "nir": M_Bands["VNP_Grid_1km_2D_SurfReflect_M7_1"],
            "swir_1": M_Bands["VNP_Grid_1km_2D_SurfReflect_M10_1"],
            "swir_2": M_Bands["VNP_Grid_1km_2D_SurfReflect_M11_1"],
            "ndvi": I_Bands,
        }
    elif input_dir.parent.name == "VNP09GA.002":
        print(f"Loading VNP09GA.002 product: {input_dir.name}")
        I_Bands = rio.open_rasterio(Path(input_dir, "VNP09GA.002_500m_aid0001.nc"))
        M_Bands = rio.open_rasterio(Path(input_dir, "VNP09GA.002_1km_aid0001.nc"))

        bands = {
            "red": M_Bands["SurfReflect_M5_1"],
            "nir": M_Bands["SurfReflect_M7_1"],
            "swir_1": M_Bands["SurfReflect_M10_1"],
            "swir_2": M_Bands["SurfReflect_M11_1"],
            "ndvi": I_Bands,
        }
    else:
        raise ValueError("Invalid Product")

    product_title = M_Bands.attrs["title"].split(" for ")[0]

    return product_title, bands


def process_product(product_title: str, bands: dict, output_dir: Path):
    """
    Processes a VIIRS product, calculating the NDVI, NDMI and NBR.
    Then saves the results as GeoTIFFs in the output directory.

    Args:
        input_dir (Path): The path to the VIIRS product.
        output_dir (Path): The directory to save the processed product to.
    Returns:
        None
    """
    for band_name in ["red", "nir", "swir_1", "swir_2", "ndvi"]:
        # Reproject to EPSG:4326
        bands[band_name] = bands[band_name].rio.reproject("EPSG:4326")

        # Update the CRS Metadata
        bands[band_name].rio.write_crs(4326, inplace=True)
        bands[band_name].attrs["crs"] = "EPSG:4326"

        # Convert from int16 to float32
        bands[band_name] = xa_int16_to_float32(
            bands[band_name], bands[band_name].attrs["_FillValue"]
        )

    # Calculating the NDVI, NDMI and NBR
    # ndvi = bands["ndvi"]
    ndvi = (bands["nir"] - bands["red"]) / (bands["nir"] + bands["red"])
    ndmi = (bands["nir"] - bands["swir_1"]) / (bands["nir"] + bands["swir_1"])
    nbr = (bands["nir"] - bands["swir_2"]) / (bands["nir"] + bands["swir_2"])

    # Iterating over the products
    for product in [ndvi, ndmi, nbr]:
        # Specifying the file suffix
        if product is ndvi:
            suffix = "ndvi"
        elif product is ndmi:
            suffix = "ndmi"
        elif product is nbr:
            suffix = "nbr"

        # Iterating over the time dimension for each product
        for i in range(len(product)):
            acquisition_date = product[i]["time"].values.item()

            # Copying the product to write the attributes (Appears to be read only whilst iterating over the time dimension...?)
            temp_product = product[i].copy()

            # Writing the product_name and CRS to the Attributes
            temp_product.attrs["name"] = (
                f"{acquisition_date.year}{str(acquisition_date.month).zfill(2)}{str(acquisition_date.day).zfill(2)}_{product_title}_{suffix}"
            )
            temp_product.attrs["crs"] = "EPSG:4326"
            temp_product.rio.write_crs(4326, inplace=True)

            # Saving the product as a GeoTIFF
            save_path = Path(
                output_dir,
                f"{acquisition_date.year}{str(acquisition_date.month).zfill(2)}{str(acquisition_date.day).zfill(2)}_{product_title}",
                f"{suffix}.tif",
            )
            save_path.parent.mkdir(parents=True, exist_ok=True)

            temp_product.rio.to_raster(save_path)


def process_products(
    product_paths: tuple,
    viirs_dirs: list,
    viirs_processed_dir: Path,
    all_products: bool,
):
    """
    Loads and processes a VIIRS product.

    Args:
        input_dir (Path): The directory containing the VIIRS product.
        output_dir (Path): The directory to save the processed product to.

    Returns:
        None
    """
    # Processes all VIIRS products in the directory, if all_products is True
    products = viirs_dirs if all_products else product_paths

    for product in tqdm(products):
        product_title, bands = load_product(product)
        process_product(product_title, bands, viirs_processed_dir)
        print(f"\nVIIRS product processed and saved to {viirs_processed_dir}")


#####################################################################################################
#                                                                                                   #
#                                   Command Line Interface                                          #
#                                                                                                   #
#####################################################################################################
def main(
    raw_dir: Path = Path("data/raw_data/viirs"),
    processed_dir: Path = Path("data/processed_data/viirs"),
    task_name: str = "VIIRS_Task",
    min_lon: float = 116.01897,
    max_lon: float = 116.20093,
    min_lat: float = -32.30959,
    max_lat: float = -31.98176,
    shapefile: Path = None,
    start_date: datetime = "2024-01-01",
    end_date: datetime = "2024-01-08",
    product_type: str = "VNP09GA.001",
):
    """
    Main function for processing VIIRS data.

    Args:
        raw_dir (Path): Directory to store raw data. Default is "data/raw_data/viirs".
        processed_dir (Path): Directory to store processed data. Default is "data/processed_data/viirs".
        task_name (str): Name of the task. Default is "VIIRS_Task".
        min_lon (float): Minimum longitude for data extraction. Default is 116.01897.
        max_lon (float): Maximum longitude for data extraction. Default is 116.20093.
        min_lat (float): Minimum latitude for data extraction. Default is -32.30959.
        max_lat (float): Maximum latitude for data extraction. Default is -31.98176.
        shapefile (Path): Path to the shapefile for spatial filtering. Default is None.
        start_date (datetime): Start date for data extraction. Default is "2024-01-01".
        end_date (datetime): End date for data extraction. Default is "2024-01-08".
        product_type (str): Type of VIIRS product. Default is "VNP09GA.001".
    """
    # Create the directories
    raw_dir = Path(raw_dir, task_name)
    processed_dir = Path(processed_dir, task_name)

    for directory in [raw_dir, processed_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    # Get Username and Password
    username = input("Enter your APPEEARS username: ")
    password = getpass.getpass("Enter your APPEEARS password: ")

    # Initialise the APPEEARS Downloader
    appeears = APPEEARSDownloader(username, password)
    token = appeears.get_token()

    # Generate the task
    task = appeears.generate_task(
        task_name=task_name,
        start_date=start_date,
        end_date=end_date,
        product_type=product_type,
        min_lon=min_lon,
        max_lon=max_lon,
        min_lat=min_lat,
        max_lat=max_lat,
        shapefile=shapefile,
    )

    # Submit the task
    task_id = appeears.submit_task(task, token)

    # Check the task status
    check = appeears.check_task(task_id, token, 120)

    # Download once task is ready
    if check:
        appeears.download(task_name, task_id, product_type, raw_dir, token)

    # Logout of APPEEARS
    appeears.logout(token)

    # Process the downloaded products
    viirs_dirs = list(Path(raw_dir, product_type).glob("*"))
    viirs_dirs = [x for x in viirs_dirs if x.is_dir()]

    process_products(
        product_paths=viirs_dirs,
        viirs_dirs=viirs_dirs,
        viirs_processed_dir=processed_dir,
        all_products=True,
    )

    print("\nVIIRS processing complete!")


if __name__ == "__main__":
    typer.run(main)
