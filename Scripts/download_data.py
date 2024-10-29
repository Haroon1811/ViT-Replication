import os 
import zipfile 

from pathlib import Path
import requests 

def download_data(source: str,
                  destination: str,
                  remove_source: bool=True):
    """ Downloads a zipped dataset from source and unzips it to destinations.

    Args:
        source(str): A link to a zipped file containing data.
        destination(str): A target directory to unzip data to.
        remove_source(bool): Whether to remove the source after downlaoding and extracting or not.
    Returns:
        pathlib.Path to downloaded data
    """
    # Setup path to data folder 
    data_path = Path("/home/haroon/ViT/DATA/")
    image_path = data_path / destination

    # If the image folder doesn't exist, download it and prepare it...
    if image_path.is_dir():
        print(f"[INFO] {image_path} directory exists, skipping download.")
    else:
        print(f"[INFO] did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)

        # Download the dataset 
        target_file = Path(source).name
        with open(data_path / target_file, "wb") as f:
            request = requests.get(source)
            print(f"[INFO] Downloading {target_file} data...")
            f.write(request.content)
            print(f"Done")
        # Unzipping the file
        with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
            print(f"[INFO] Unzippig {target_file} data...")
            zip_ref.extractall(image_path)
            print("Done")
        # Remove .zip file 
        if remove_source:
            os.remove(data_path / target_file)
    return image_path

