import sys
import torch


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

