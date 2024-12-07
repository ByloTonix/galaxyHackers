"""Utility functions and enumerations for handling datasets and legacy survey data."""

import sys
from enum import Enum
from typing import Any, Generator, List

import numpy as np
import pandas as pd
import torch
from astroquery.vizier import Vizier


class DataSource(str, Enum):
    """Enumeration of data sources."""

    MAP_ACT = "map_act"
    DR5 = "dr5"
    MC = "mc"
    SGA = "sga"
    TYC2 = "tyc2"
    GAIA = "gaia"
    UPC_SZ = "upc_sz"  # UPCluster-SZ catalog, обобщённый планк
    SPT_SZ = "spt_sz"
    PSZSPT = "pszspt"
    CCOMPRASS = "comprass"
    SPT2500D = "spt2500d"
    SPTECS = "sptecs"
    SPT100 = "spt100"
    ACT_MCMF = "act_mcmf"

    TEST_SAMPLE = "test_sample"
    RANDOM = "rand"


class IsCluster(int, Enum):
    """Enumeration for cluster classification."""

    IS_CLUSTER = 1
    NOT_CLUSTER = 0


required_columns = set(["idx", "ra_deg", "dec_deg", "name", "source", "target"])
optional_columns = set(["red_shift", "red_shift_type"])


def inherit_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensures the DataFrame has required and optional columns.

    Args:
        frame (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with required and optional columns ensured.
    """
    frame["idx"] = np.arange(len(frame))

    frame_columns = set(frame.columns)

    assert required_columns.issubset(frame_columns), "Some required columns are missed"

    missed_optional = optional_columns.difference(frame_columns)

    if missed_optional:
        for col in missed_optional:
            frame[col] = pd.NA

    frame = frame.reset_index(drop=True)

    return frame


def read_vizier(catalogue: str) -> pd.DataFrame:
    """Fetches a catalogue from Vizier and converts it to a pandas DataFrame.

    Args:
        catalogue (str): Name or identifier of the catalogue.

    Returns:
        pd.DataFrame: DataFrame containing the catalogue data.
    """
    catalog_list = Vizier.find_catalogs(catalogue)
    Vizier.ROW_LIMIT = -1

    catalogs = Vizier.get_catalogs(catalog_list.keys())
    frame = catalogs[0].to_pandas().reset_index(drop=True)
    return frame


def bar_progress(current: int, total: int, width: int = 80) -> None:
    """Displays a progress bar for downloads.

    Args:
        current (int): Current number of bytes downloaded.
        total (int): Total number of bytes to download.
        width (int, optional): Width of the progress bar. Defaults to 80.
    """
    progress_message = "Downloading: %d%% [%d / %d] bytes" % (
        current / total * 100,
        current,
        total,
    )
    # Don't use print() as it will print in new line every time.
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


def to_hms_format(time_str: str) -> str:
    """Converts a time string to HMS format.

    Args:
        time_str (str): Time string in space-separated format (e.g., "12 34 56").

    Returns:
        str: Time string in HMS format (e.g., "12h34m56s").
    """
    parts = time_str.split()
    return f"{parts[0]}h{parts[1]}m{parts[2]}s"


def to_dms_format(time_str: str) -> str:
    """Converts a time string to DMS format.

    Args:
        time_str (str): Time string in space-separated format (e.g., "12 34 56").

    Returns:
        str: Time string in DMS format (e.g., "12d34m56s").
    """
    parts = time_str.split()
    return f"{parts[0]}d{parts[1]}m{parts[2]}s"


def divide_chunks(data_list: List[Any], chunk_size: int) -> Generator[List[Any], None, None]:
    """Divides a list into chunks of a specified size.

    Args:
        data_list (List[Any]): Input list to divide.
        chunk_size (int): Size of each chunk.

    Yields:
        Generator[List[Any], None, None]: Generator yielding list chunks.
    """
    for i in range(0, len(l), n):
        yield l[i : i + n]


def fits_to_rgb_image(tensor: torch.Tensor) -> torch.Tensor:
    """Converts a 2-channel tensor to an RGB image tensor.

    Args:
        tensor (torch.Tensor): Input tensor of shape (2, H, W).

    Returns:
        torch.Tensor: RGB image tensor of shape (3, H, W).

    Raises:
        ValueError: If the input tensor does not have 2 channels.
    """
    # Ensure the tensor has the correct shape
    if tensor.shape[0] != 2:
        raise ValueError("Input tensor must have 2 channels.")

    # Extract channels
    channel1 = tensor[0]
    channel2 = tensor[1]

    # Normalize channels individually
    channel1_norm = (channel1 - channel1.min()) / (channel1.max() - channel1.min())
    channel2_norm = (channel2 - channel2.min()) / (channel2.max() - channel2.min())

    # Option 1: Map channels directly to RGB
    # Assign channel1 to Red, channel2 to Green, and set Blue to the average
    red = channel1_norm
    green = channel2_norm
    blue = (channel1_norm + channel2_norm) / 2

    # Stack the channels to form an RGB image
    rgb_image = torch.stack([red, green, blue], dim=0)

    # Optional: Clip values to ensure they are between 0 and 1
    rgb_image = torch.clamp(rgb_image, 0, 1)

    return rgb_image
