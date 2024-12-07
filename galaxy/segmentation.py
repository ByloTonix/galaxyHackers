"""Script to create segmentation maps for models.

To create segmentation maps for randomly chosen clusters, random objects and stars use function saveSegMaps().
To create a segmentation map with larger scale for a randomly chosen cluster use function saveBigSegMap().
"""

import gc
import os
from enum import Enum
from pathlib import Path
from typing import Generator

import astropy.coordinates as coord
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from astropy import units as u
from matplotlib import cm
from torch.utils.data import DataLoader

from galaxy import data, grabber, train
from galaxy.config import settings


def cleanup_memory(*args) -> None:
    """Cleans up memory by deleting variables and freeing GPU memory.

    Args:
        *args: Variables to be deleted.
    """
    for var in args:
        del var
    torch.cuda.empty_cache()
    gc.collect()


def load_model(
    model: torch.nn.Module, optimizer_name: str, device: torch.device
) -> torch.nn.Module:
    """Loads a trained model with specific weights.

    Args:
        model (torch.nn.Module): The model to be loaded.
        optimizer_name (str): Name of the optimizer used during training.
        device (torch.device): Device to load the model on.

    Returns:
        torch.nn.Module: The loaded model.
    """

    model = model.load_model()
    weights_name = f"best_weights_{model.__class__.__name__}_{optimizer_name}.pth"
    weights_path = Path(settings.BEST_MODELS_PATH, weights_name)
    loaded_model = torch.load(weights_path, map_location=device)

    model.load_state_dict(loaded_model)
    model = model.to(device)

    return model


def predict_test(model: torch.nn.Module, optimizer_name: str) -> None:
    """Runs predictions on the test and bright stars datasets.

    Args:
        model (torch.nn.Module): The model to make predictions with.
        optimizer_name (str): Name of the optimizer used during training.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    load_model(model, optimizer_name, device)
    predictor = train.Predictor(model=model, device=device)
    dataloaders = data.train_val_test_split()
    for part in [data.DataPart.TEST, data.DataPart.BRIGHT_STARS]:
        predictions = predictor.predict(dataloader=dataloaders[part])
        predictions.to_csv(Path(settings.PREDICTIONS_PATH, f"{part}.csv"))


class SampleName(str, Enum):
    """Enumeration for sample names."""

    CLUSTER_SMALL = "cluster_small"
    NEGATIVE_SMALL = "random_small"
    TYC2_SMALL = "tyc2_small"
    # CLUSTER_BIG = "cluster_big"
    # MC_BIG = "mc_big"


sample_sizes: dict = {
    SampleName.CLUSTER_SMALL: 10,
    SampleName.NEGATIVE_SMALL: 5,
    SampleName.TYC2_SMALL: 5,
    # SampleName.CLUSTER_BIG: 1,
    # SampleName.MC_BIG: 1,
}


class MapType(str, Enum):
    """Enumeration for map types."""

    SMALL = 0
    BIG = 1


plot_radius = {
    MapType.SMALL: 10,
    MapType.BIG: 15,
}

# Map type, Data part and target class for each sample
sample_sources = {
    SampleName.CLUSTER_SMALL: (MapType.SMALL, data.DataPart.TEST, 1),
    SampleName.NEGATIVE_SMALL: (MapType.SMALL, data.DataPart.TEST, 0),
    SampleName.TYC2_SMALL: (
        MapType.SMALL,
        data.DataPart.BRIGHT_STARS,
        0,
    ),  # use bright stars
    # SampleName.CLUSTER_BIG: (MapType.BIG, data.DataPart.TEST, 1),
    # SampleName.MC_BIG: (MapType.BIG, data.DataPart.MC, 1),
}


def create_sample(
    sample_name, predictor: train.Predictor
) -> tuple[pd.DataFrame, pd.DataFrame, MapType]:
    """Creates a sample dataset for segmentation map generation.

    Args:
        sample_name (SampleName): Name of the sample.
        predictor (train.Predictor): Predictor object for making predictions.

    Returns:
        tuple: Sample data, sample predictions, and map type.
    """
    sample_size = sample_sizes[sample_name]
    map_type, source, target_class = sample_sources[sample_name]

    description = pd.read_csv(
        Path(settings.DESCRIPTION_PATH, f"{source.value}.csv"), index_col=0
    )
    description = description.loc[description["target"] == target_class]

    min_ra, min_dec = -float("inf"), -float("inf")
    max_ra, max_dec = float("inf"), float("inf")

    match (map_type):
        case MapType.SMALL:
            required_space = (
                10 / 120
            )  # radius required around clusters for segmentation maps (in minutes)
        case MapType.BIG:
            required_space = 30 / 120  # radius required around cluster (in minutes)

    while (
        (max_ra + required_space) > 360
        or (max_dec + required_space) > 90
        or (min_dec - required_space) < -90
        or (min_ra - required_space) < 0
    ):

        sample = description.sample(sample_size, random_state=settings.SEED)
        max_ra = sample["ra_deg"].max()
        max_dec = sample["dec_deg"].max()
        min_ra = sample["ra_deg"].min()
        min_dec = sample["dec_deg"].min()

    sample_description_path = Path(
        settings.SEGMENTATION_SAMPLES_DESCRIPTION_PATH,
        f"{sample_name.value}.csv",
        index=True,
    )

    sample.to_csv(sample_description_path)

    dataset = data.ClusterDataset(
        images_dir_path=Path(settings.DATA_PATH, source.value),
        description_csv_path=sample_description_path,
    )

    dataloader = DataLoader(dataset, batch_size=len(dataset))
    sample_predictions = predictor.predict(dataloader)

    return sample, sample_predictions, map_type


def grab_surrounding(points_on_radius: int) -> Generator[tuple[int, int], None, None]:
    """Generates surrounding grid points for a given radius.

    Args:
        points_on_radius (int): Radius in grid points.

    Yields:
        tuple[int, int]: Coordinates of grid points.
    """
    for x in range(-points_on_radius, points_on_radius + 1):
        for y in range(-points_on_radius, points_on_radius + 1):
            yield (x, y)


def create_map_dataloader(
    map_type: MapType,
    ra_start: float,
    dec_start: float,
    map_dir: Path,
    survey=None,
    bands=None,
    imgsize_pix=224,
) -> DataLoader:  # id: 0 for small segmentation maps, 1 - for a big one
    """Creates a dataloader for a segmentation map.

    Args:
        map_type (MapType): Type of the map.
        ra_start (float): Starting RA coordinate.
        dec_start (float): Starting Dec coordinate.
        map_dir (Path): Directory to save the map data.
        survey (str, optional): Survey name. Defaults to None.
        bands (list, optional): List of bands. Defaults to None.
        imgsize_pix (int, optional): Image size in pixels. Defaults to 224.

    Returns:
        DataLoader: Dataloader object for the segmentation map.
    """
    name, idxs, ras, decs = [], [], [], []

    match map_type:
        case MapType.SMALL:
            cycle_step = 0.5  # шаг в 0.5 минуту
            step = cycle_step / 60  # шаг в 0.5 минуту, выражено в градусах
        case MapType.BIG:
            cycle_step = 1  # шаг в 1 минуту
            step = cycle_step / 60  # шаг в 1 минуту, выражено в градусах

    # cluster in centre (0, 0) and its surrounding
    surrounding = grab_surrounding(plot_radius[map_type])
    for i, (x, y) in enumerate(surrounding):
        ra_current = ra_start + step * x
        dec_current = dec_start + step * y

        coords = coord.SkyCoord(
            ra=ra_current * u.degree, dec=dec_current * u.degree, frame="icrs"
        )

        ras.append(coords.ra.degree)
        decs.append(coords.dec.degree)
        idxs.append(i)

        b = coords.galactic.b.degree
        l = coords.galactic.l.degree

        name.append(f"Map {l:.3f}{b:+.3f}")

    description_path = Path(map_dir, f"description.csv")

    map_data = pd.DataFrame(
        {"name": name, "ra_deg": ras, "dec_deg": decs}, index=pd.Index(idxs, name="idx")
    )
    map_data["red_shift"] = np.nan
    map_data["red_shift_type"] = "nan"
    map_data["target"] = 0

    map_data.to_csv(description_path)

    g = grabber.Grabber()
    g.grab_cutouts(targets=map_data, output_dir=map_dir)

    dataset = data.ClusterDataset(map_dir, description_path)

    dataloader = DataLoader(dataset, batch_size=settings.BATCH_SIZE)

    return dataloader


def prepare_sample_dataloaders(
    data: pd.DataFrame, sample_name: SampleName, map_type: MapType
) -> list[tuple[int, DataLoader]]:
    """Prepares dataloaders for segmentation map samples.

    Args:
        data (pd.DataFrame): Data describing the samples.
        sample_name (SampleName): Name of the sample.
        map_type (MapType): Type of map to generate.

    Returns:
        list[tuple[int, DataLoader]]: List of tuples containing index and dataloader.
    """
    dataloaders = []

    for idx, row in data.iterrows():
        directory = Path(
            settings.SEGMENTATION_SAMPLES_PATH, sample_name.value, str(idx)
        )
        os.makedirs(directory, exist_ok=True)

        dataloader = create_map_dataloader(
            map_type=map_type,
            ra_start=row["ra_deg"],
            dec_start=row["dec_deg"],
            map_dir=directory,
        )

        dataloaders.append((idx, dataloader))

    return dataloaders


def create_segmentation_plot(
    model_name: str,
    optimizer_name: str,
    predictor: train.Predictor,
    sample_name: SampleName,
    n_cols: int = 5,
) -> None:
    """Creates a segmentation map plot for a given sample.

    Args:
        model_name (str): Name of the model.
        optimizer_name (str): Name of the optimizer used during training.
        predictor (train.Predictor): Predictor for making predictions.
        sample_name (SampleName): Name of the sample.
        n_cols (int, optional): Number of columns in the plot. Defaults to 5.
    """
    sample, sample_predictions, map_type = create_sample(
        sample_name=sample_name, predictor=predictor
    )

    dataloaders = prepare_sample_dataloaders(
        data=sample, sample_name=sample_name, map_type=map_type
    )

    n_rows = max(1, (len(sample) + 1) // n_cols)
    n_cols = min(n_cols, len(sample))

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 8))

    for i, (idx, dataloader) in enumerate(dataloaders):

        cur_col = i % n_cols

        if n_rows > 1:
            cur_row = i // n_cols
            cur_position = (cur_row, cur_col)
        else:
            cur_position = cur_col

        if len(sample) > 1:
            cur_ax = axes[cur_position]
        else:
            cur_ax = axes

        predictions = predictor.predict(dataloader)

        path = Path(
            settings.SEGMENTATION_SAMPLES_PATH,
            sample_name.value,
            str(idx),
            "predictions.csv",
        )
        predictions.to_csv(path)

        cur_ax.plot()
        subtitle = "Probability: " + "{:.4f}".format(
            float(sample_predictions.loc[int(idx), "y_prob"])
        )
        cur_ax.set_title(subtitle)

        plot_size = plot_radius[map_type] * 2 + 1
        center = int(plot_size // 2)

        im = cur_ax.imshow(
            predictions["y_prob"].values.reshape(plot_size, plot_size).astype(float),
            cmap=cm.PuBu,
            vmin=0,
            vmax=1,
        )
        cur_ax.axis("off")
        cur_ax.plot(center, center, "o", ms=3, color="red")

    if len(sample) > 1:
        axes_ravel = axes.ravel().tolist()
    else:
        axes_ravel = axes

    fig.colorbar(
        im,
        ax=axes_ravel,
        label="Cluster probability",
        orientation="horizontal",
        aspect=40,
    )
    # plt.suptitle(all_samples[i][0], size='xx-large')

    plt.savefig(
        Path(
            settings.SEGMENTATION_MAPS_PATH,
            f"{map_type}_{model_name}_{optimizer_name}_{sample_name.value}.png",
        )
    )
    plt.close()


def create_segmentation_plots(
    model: torch.nn.Module,
    model_name: str,
    optimizer_name: str,
    map_type: MapType = MapType.SMALL,
) -> None:
    """Creates segmentation maps for all samples.

    Args:
        model (torch.nn.Module): The model used for predictions.
        model_name (str): Name of the model.
        optimizer_name (str): Name of the optimizer used during training.
        map_type (MapType, optional): Type of map to generate. Defaults to MapType.SMALL.

    Returns:
        None
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = load_model(model, optimizer_name, device)

    predictor = train.Predictor(model, device=device)

    for sample_name in list(SampleName):
        create_segmentation_plot(
            model_name=model_name,
            optimizer_name=optimizer_name,
            predictor=predictor,
            sample_name=sample_name,
        )
