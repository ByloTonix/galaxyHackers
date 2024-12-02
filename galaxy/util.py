import sys
import torch
from astroquery.vizier import Vizier
from enum import Enum
import numpy as np
import pandas as pd


class DataSource(str, Enum):

    MAP_ACT = "map_act"
    DR5 = "dr5"
    MC = "mc"
    SGA = "sga"
    TYC2 = "tyc2"
    GAIA = "gaia"
    UPC_SZ = "upc_sz" # UPCluster-SZ catalog, обобщённый планк
    SPT_SZ = "spt_sz"
    PSZSPT = "pszspt"
    CCOMPRASS = "comprass"
    SPT2500D = "spt2500d"
    SPTECS = "sptecs"
    SPT100 = "spt100"

    TEST_SAMPLE = "test_sample"
    RANDOM = "rand"


class IsCluster(int, Enum):

    IS_CLUSTER = 1
    NOT_CLUSTER = 0


required_columns = set(["idx", "ra_deg", "dec_deg", "name", "source", "is_cluster"])
optional_columns = set(["red_shift", "red_shift_type"])


def inherit_columns(frame: pd.DataFrame):

    frame['idx'] = np.arange(len(frame))

    frame_columns = set(frame.columns)

    assert required_columns.issubset(frame_columns), "Some required columns are missed"

    missed_optional = optional_columns.difference(frame_columns)

    if missed_optional:
        for col in missed_optional:
            frame[col] = pd.NA

    frame = frame.reset_index(drop=True)

    return frame


def read_vizier(catalogue):    
    catalog_list = Vizier.find_catalogs(catalogue)
    Vizier.ROW_LIMIT = -1

    catalogs = Vizier.get_catalogs(catalog_list.keys())
    frame = catalogs[0].to_pandas().reset_index(drop=True)
    return frame



def bar_progress(current, total, width=80):
    progress_message = "Downloading: %d%% [%d / %d] bytes" % (
        current / total * 100,
        current,
        total,
    )
    # Don't use print() as it will print in new line every time.
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


def to_hms_format(time_str):
    parts = time_str.split()
    return f"{parts[0]}h{parts[1]}m{parts[2]}s"


def to_dms_format(time_str):
    parts = time_str.split()
    return f"{parts[0]}d{parts[1]}m{parts[2]}s"


def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i : i + n]


def fits_to_rgb_image(tensor):
    """
    Converts a 2-channel tensor from the legacy survey to an RGB image tensor.

    Parameters:
    - tensor (Tensor): Input tensor of shape (2, H, W)

    Returns:
    - Tensor: RGB image tensor of shape (3, H, W)
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

