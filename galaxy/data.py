"""Script to get dataloaders"""

"""To create dataloaders you need to address only function create_dataloaders()"""

import os
from collections import defaultdict
from enum import Enum
from pathlib import Path

import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wget
from astropy.io import fits
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from galaxy import collect_clusters, collect_not_clusters, grabber, util
from galaxy.config import settings
from galaxy.util import DataSource, IsCluster, inherit_columns


np.random.seed(settings.SEED)

TORCHVISION_MEAN = [23.19058950345032, 22.780995295792817]
TORCHVISION_STD = [106.89880134344101, 100.32284196853638]

main_transforms = [
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=TORCHVISION_MEAN, std=TORCHVISION_STD),
]


class DataPart(str, Enum):

    TRAIN = "train"
    VALIDATE = "validate"
    TEST = "test"
    MC = "mc"
    BRIGHT_STARS = "bright_stars"
    TEST_SAMPLE = "test_sample"


# class SurveyLayer(str, Enum):
#     UNWISE_NEO7 = "unwise-neo7"
#     VLASS1_2 = "vlass1.2"


def shift_and_mirror_pad(
    image: torch.Tensor, shift_x: int, shift_y: int
) -> torch.Tensor:
    """
    Shifts the image by (shift_x, shift_y) and applies mirror-padding.

    Parameters:
    - image (Tensor): Input tensor of shape (C, H, W)
    - shift_x (int): Horizontal shift (positive: right, negative: left)
    - shift_y (int): Vertical shift (positive: down, negative: up)

    Returns:
    - Tensor: Augmented image tensor of shape (C, H, W)
    """
    C, H, W = image.shape

    # Initialize padding
    pad_left = max(shift_x, 0)
    pad_right = max(-shift_x, 0)
    pad_top = max(shift_y, 0)
    pad_bottom = max(-shift_y, 0)

    # Apply mirror padding
    padded_image = F.pad(
        image, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect"
    )

    # Calculate cropping indices
    crop_left = pad_right
    crop_right = crop_left + W
    crop_top = pad_bottom
    crop_bottom = crop_top + H

    # Crop the image to original size
    shifted_image = padded_image[:, crop_top:crop_bottom, crop_left:crop_right]

    return shifted_image


# TODO(попозже): actually should apply shift_and_mirror_pad() to images to save time on ddos
# currently simply shifts coordinates
class ShiftAndMirrorPadTransform:
    """Applies random shifts to coordinates for data augmentation."""

    def __init__(self, max_shift_x: int = 20, max_shift_y: int = 20):
        self.max_shift_x = max_shift_x
        self.max_shift_y = max_shift_y

        self.shift_x = 0
        self.shift_y = 0

    def __call__(self):
        """Generates random shift values within the specified range."""
        # Random shift values within the specified range
        self.shift_x = torch.randint(
            -self.max_shift_x, self.max_shift_x + 1, (1,)
        ).item()
        self.shift_y = torch.randint(
            -self.max_shift_y, self.max_shift_y + 1, (1,)
        ).item()

    def apply_shift(self, ra_deg: float, dec_deg: float) -> tuple[float, float]:
        """Applies the generated shift to given coordinates.

        Args:
            ra_deg (float): Right ascension in degrees.
            dec_deg (float): Declination in degrees.

        Returns:
            tuple[float, float]: Shifted coordinates (RA, Dec).
        """
        ra_deg += self.shift_x
        dec_deg += self.shift_y
        return ra_deg, dec_deg

    # def __call__(self, image):
    #     shift_x = torch.randint(-self.max_shift_x, self.max_shift_x + 1, (1,)).item()
    #     shift_y = torch.randint(-self.max_shift_y, self.max_shift_y + 1, (1,)).item()
    #     return shift_and_mirror_pad(image, shift_x, shift_y)


class ClusterDataset(Dataset):
    """Custom PyTorch Dataset for cluster data."""

    def __init__(self, images_dir_path: str, description_csv_path: str, transform=None):
        """
        Args:
            images_dir_path (str): Path to the directory containing FITS images.
            description_csv_path (str): Path to the CSV file with metadata.
            transform (callable, optional): Optional transform to apply to the images.
        """
        super().__init__()

        self.images_dir_path = images_dir_path
        self.description_df = pd.read_csv(
            description_csv_path, index_col=0, dtype={"target": int}
        )
        if not os.path.exists(description_csv_path):
            raise FileNotFoundError(
                f"Description file not found: {description_csv_path}"
            )

        self.description_df.index = self.description_df.index.astype(str)

        if "red_shift" in self.description_df.columns:
            self.description_df.loc[:, "red_shift"] = self.description_df[
                "red_shift"
            ].astype(float)

        self.description_df.index.name = "idx"
        self.transform = transform

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.description_df)

    def __getitem__(self, idx: int) -> dict:
        """Gets a single sample by index.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: Dictionary containing image, label, and metadata.
        """
        row_index = self.description_df.index[idx]
        row = self.description_df.loc[row_index]

        img_name = row_index

        img_path = Path(self.images_dir_path, f"{img_name}.fits")
        img = self._read_img(img_path)

        if self.transform:
            img = self.transform(img)

        description = {
            "idx": img_name,
            "red_shift": row["red_shift"],
            "red_shift_type": str(row["red_shift_type"]),
        }
        sample = {"image": img, "label": row["target"], "description": description}

        return sample

    @staticmethod
    def _read_img(fits_path: Path) -> torch.Tensor:
        """Reads a FITS image from the given path.

        Args:
            fits_path (Path): Path to the FITS file.

        Returns:
            torch.Tensor: Tensor representation of the image.
        """
        with fits.open(fits_path) as hdul:
            img = torch.Tensor(hdul[0].data.astype(np.float64))

        return img


"""Collecting clusters of galaxies for positive class"""


def download_data() -> None:
    """Downloads required data files."""

    for config in [
        settings.MAP_ACT_CONFIG,
        settings.DR5_CONFIG,
        settings.SGA_CONFIG,
        settings.SPT100_CONFIG,
    ]:
        if not os.path.exists(config.OUTPUT_PATH):
            wget.download(
                url=config.URL, out=str(settings.DATA_PATH), bar=util.bar_progress
            )

            rename_dict = config.RENAME_DICT
            os.rename(rename_dict.SOURCE, rename_dict.TARGET)

            # except Exception:
            #     # Getting 403, what credentials needed?
            #     wget.download(
            #         url=config.FALLBACK_URL,
            #         out=config.OUTPUT_PATH,
            #         bar=bar_progress,
            #     )


#


"""Combining datasets"""


def get_positive_class() -> pd.DataFrame:
    """Combines multiple datasets to create the positive class.

    Returns:
        pd.DataFrame: Combined positive class dataset.
    """
    dr5 = collect_clusters.read_dr5()
    upc_sz = collect_clusters.read_upc_sz()
    spt_sz = collect_clusters.read_spt_sz()
    pszspt = collect_clusters.read_pszspt()
    comprass = collect_clusters.read_comprass()
    spt2500d = collect_clusters.read_spt2500d()
    sptecs = collect_clusters.read_sptecs()
    spt100 = collect_clusters.read_spt100()
    act_mcmf = collect_clusters.read_act_mcmf()

    clusters = pd.concat(
        [
            dr5,
            upc_sz,
            spt_sz,
            pszspt,
            comprass,
            spt2500d,
            sptecs,
            spt100,
            act_mcmf,
        ],
        ignore_index=True,
    )

    return clusters


def get_cluster_catalog() -> coord.SkyCoord:
    """Creates a SkyCoord catalog of known clusters.

    Returns:
        coord.SkyCoord: Catalog of clusters as a SkyCoord object.
    """
    clusters = datasets_collection.load_clusters()
    test_sample = collect_clusters.read_test_sample()

    clusters = pd.concat([clusters, test_sample], ignore_index=True)

    # The catalog of known found galaxies
    catalog = coord.SkyCoord(
        ra=clusters["ra_deg"] * u.degree, dec=clusters["dec_deg"] * u.degree, unit="deg"
    )

    return catalog


def expanded_positive_class() -> pd.DataFrame:
    """Generates an expanded dataset of positive class by applying random shifts.

    Returns:
        pd.DataFrame: Expanded dataset of positive class with shifted coordinates.
    """

    # Для того чтобы выборка была сбалансирована, координаты подтверждённых скоплений
    # немного шатаются => выборка увеличивается

    clusters = datasets_collection.load_clusters()

    more_clusters = []

    # initialized not for loop to avoid adressing __init__ each time and simply generates random shifts
    transform_positive = ShiftAndMirrorPadTransform()

    for _, row in clusters.iterrows():
        for _ in range(1):  # + новых картинок подтверждённых скоплений
            transform_positive()
            new_ra, new_dec = transform_positive.apply_shift(
                row["ra_deg"], row["dec_deg"]
            )
            more_clusters.append(
                {
                    "name": row["name"],
                    "ra_deg": new_ra,
                    "dec_deg": new_dec,
                    "red_shift": row["red_shift"],
                    "red_shift_type": row["red_shift_type"],
                    "target": IsCluster.IS_CLUSTER.value,
                    "source": row["source"],
                }
            )

    more_clusters_df = pd.DataFrame(more_clusters)

    extended_clusters = pd.concat(
        [
            clusters,
            more_clusters_df,
        ],
        ignore_index=True,
    )

    return extended_clusters


def get_negative_class() -> pd.DataFrame:
    """Combines datasets to create the negative class.

    Returns:
        pd.DataFrame: Combined dataset of negative class.
    """
    sga = collect_not_clusters.read_sga()
    gaia = collect_not_clusters.read_gaia()

    # drop columns "red_shift" and "red_shift_type", as tables with NaN columns cannot be concatenated properly(?)
    columns_to_drop = ["red_shift", "red_shift_type"]
    sga = sga.drop(
        columns=[col for col in columns_to_drop if col in sga.columns], errors="ignore"
    )
    gaia = gaia.drop(
        columns=[col for col in columns_to_drop if col in gaia.columns], errors="ignore"
    )

    negative_class = pd.concat([sga, gaia], axis=0, ignore_index=True)
    negative_class = inherit_columns(negative_class)
    return negative_class


def get_non_cluster_catalog() -> coord.SkyCoord:
    """Creates a SkyCoord catalog of non-cluster objects.

    Returns:
        coord.SkyCoord: Catalog of non-cluster objects as a SkyCoord object.
    """
    non_clusters = datasets_collection.load_non_clusters()

    # The catalog of known found galaxies
    catalog = coord.SkyCoord(
        ra=non_clusters["ra_deg"] * u.degree,
        dec=non_clusters["dec_deg"] * u.degree,
        unit="deg",
    )

    return catalog


def update_bright_stars(bright_stars: pd.DataFrame | None) -> pd.DataFrame:
    """Updates or loads the bright stars dataset.

    Args:
        bright_stars (pd.DataFrame | None): Existing bright stars dataset or None.

    Returns:
        pd.DataFrame: Updated bright stars dataset.
    """
    if bright_stars is None:
        bright_stars = collect_not_clusters.read_tyc2()
    return bright_stars


class DatasetsInfo:
    """Class to manage and cache datasets for training and evaluation."""

    def __init__(self):
        self._clusters = None
        self._expanded_clusters = None
        self._bright_stars = None
        self._non_clusters = None
        self._test_sample = None

    def load_clusters(self) -> pd.DataFrame:
        """Loads the positive class dataset.

        Returns:
            pd.DataFrame: Positive class dataset.
        """
        if self._clusters is None:
            self._clusters = get_positive_class()
        return self._clusters

    def load_expanded_clusters(self) -> pd.DataFrame:
        """Loads the expanded positive class dataset.

        Returns:
            pd.DataFrame: Expanded positive class dataset.
        """
        if self._expanded_clusters is None:
            self._expanded_clusters = expanded_positive_class()
        return self._expanded_clusters

    def load_test_sample(self) -> pd.DataFrame:
        """Loads the test sample dataset.

        Returns:
            pd.DataFrame: Test sample dataset.
        """
        if self._test_sample is None:
            self._test_sample = collect_clusters.read_test_sample()
        return self._test_sample

    # done separately for segmentation

    def load_bright_stars(self) -> pd.DataFrame:
        """Loads the bright stars dataset.

        Returns:
            pd.DataFrame: Bright stars dataset.
        """
        return update_bright_stars(self._bright_stars)

    def load_non_clusters(self) -> pd.DataFrame:
        """Loads the negative class dataset.

        Returns:
            pd.DataFrame: Negative class dataset.
        """
        if self._non_clusters is None:
            self._non_clusters = get_negative_class()
            self._bright_stars = update_bright_stars(self._bright_stars)

            # drop columns "red_shift" and "red_shift_type", as tables with NaN columns cannot be concatenated properly(?)
            columns_to_drop = ["red_shift", "red_shift_type"]

            self._bright_stars = self._bright_stars.drop(
                columns=[
                    col for col in columns_to_drop if col in self._bright_stars.columns
                ],
                errors="ignore",
            )
            self._non_clusters = self._non_clusters.drop(
                columns=[
                    col for col in columns_to_drop if col in self._non_clusters.columns
                ],
                errors="ignore",
            )

            self._non_clusters = pd.concat(
                [self._bright_stars, self._non_clusters], ignore_index=True
            )

            self._non_clusters = inherit_columns(self._non_clusters)

        return self._non_clusters


datasets_collection = DatasetsInfo()

"""Generate object that are not present in any of currently included datasets"""


def filter_candidates(candidates: coord.SkyCoord, max_len: int) -> coord.SkyCoord:
    """Filters candidate objects based on angular distance and galactic latitude.

    Args:
        candidates (coord.SkyCoord): SkyCoord object containing candidate objects.
        max_len (int): Maximum number of filtered candidates to return.

    Returns:
        coord.SkyCoord: Filtered SkyCoord object.
    """
    clusters_catalog = get_cluster_catalog()
    non_clusters_catalog = get_non_cluster_catalog()

    catalog = coord.SkyCoord(
        ra=np.concatenate([clusters_catalog.ra.deg, non_clusters_catalog.ra.deg])
        * u.degree,
        dec=np.concatenate([clusters_catalog.dec.deg, non_clusters_catalog.dec.deg])
        * u.degree,
        unit="deg",
    )

    idx, d2d, _ = candidates.match_to_catalog_sky(catalog)

    MIN_ANGLE, MAX_ANGLE = 10, 20

    candidates_filter = (d2d.arcmin > MIN_ANGLE) & (
        candidates.galactic.b.degree > MAX_ANGLE
    )

    filtered_candidates = candidates[candidates_filter][:max_len]

    return filtered_candidates


# TODO: ДОРАБОТАТЬ generate_random_sample
def generate_random_candidates(len: int = 7500) -> coord.SkyCoord:
    """Generates random sky coordinates for candidates.

    Args:
        len (int, optional): Number of candidates to generate. Defaults to 7500.

    Returns:
        coord.SkyCoord: SkyCoord object containing generated candidates.
    """
    n_sim = 20_000
    required_num = 7500

    np.random.seed(settings.SEED)

    ras = np.random.uniform(0, 360, n_sim)
    decs = np.random.uniform(-90, 90, n_sim)

    frame = pd.DataFrame({"ra_deg": ras, "dec_deg": decs})

    valid_idx = (
        (0 <= frame["ra_deg"])
        & (frame["ra_deg"] <= 360)
        & (-90 <= frame["dec_deg"])
        & (frame["dec_deg"] <= 90)
    )

    frame = frame[valid_idx].reset_index(drop=True)

    # Just points from our sky map
    candidates = coord.SkyCoord(
        ra=frame["ra_deg"] * u.degree, dec=frame["dec_deg"] * u.degree, unit="deg"
    )
    filtered_candidates = filter_candidates(candidates, max_len=required_num)
    return filtered_candidates


def generate_random_based(required_num: int = 7500) -> coord.SkyCoord:
    """Generates random candidates based on existing datasets.

    Args:
        required_num (int, optional): Number of candidates to generate. Defaults to 7500.

    Returns:
        coord.SkyCoord: SkyCoord object containing generated candidates.

    опытным путём было выяснено, что generate_random_candidates не даёт 7500 даже если n_sim выкрутить до 50к,
    предлагается брать собранные датасеты, брать ra_deg и dec_deg, перемешивать в независимости друг от друга
    и затем в filter_candidates подавать len = required_num - len(<собранного из generate_random_candidates>)

    #     ТЕХНИЧЕСКИЕ ШОКОЛАДКИ???
    # /usr/local/lib/python3.10/dist-packages/astropy/coordinates/angles/core.py in _validate_angles(self, angles)
    #     647                     f"<= 90 deg, got {angles.to(u.degree)}"
    #     648                 )
    # --> 649             raise ValueError(
    #     650                 "Latitude angle(s) must be within -90 deg <= angle "
    #     651                 f"<= 90 deg, got {angles.min().to(u.degree)} <= "

    ValueError: Latitude angle(s) must be within -90 deg <= angle <= 90 deg, got -100.8107 deg <= angle <= 106.2399 deg
    """
    clusters = datasets_collection.load_expanded_clusters()
    non_clusters = datasets_collection.load_non_clusters()
    frame = pd.concat(
        [
            clusters,
            non_clusters,
        ],
        ignore_index=True,
    )

    ras = frame["ra_deg"].tolist()
    decs = frame["dec_deg"].tolist()

    # Shuffling candidates, imitating samples
    np.random.seed(settings.SEED)
    np.random.shuffle(ras)
    np.random.shuffle(decs)

    frame = pd.DataFrame({"ra_deg": ras, "dec_deg": decs})

    valid_idx = (
        (0 <= frame["ra_deg"])
        & (frame["ra_deg"] <= 360)
        & (-90 <= frame["dec_deg"])
        & (frame["dec_deg"] <= 90)
    )

    frame = frame[valid_idx].reset_index(drop=True)

    # Just points from our sky map
    candidates = coord.SkyCoord(
        ra=frame["ra_deg"] * u.degree, dec=frame["dec_deg"] * u.degree, unit="deg"
    )
    filtered_candidates = filter_candidates(candidates, max_len=required_num // 2)
    return filtered_candidates


def generate_random_sample() -> pd.DataFrame:
    """Combines random candidates from `generate_random_candidates` and `generate_random_based`.

    Ensures the required number of candidates is generated by combining both methods.

    Returns:
        pd.DataFrame: DataFrame containing generated random candidates.
    """
    # for example: see create_negative_class_dr5 and create_negative_class_mc from last commit

    # 24195 - objects in extended positive class, 16700 - negative class from galaxies and stars
    required_num = 7500

    filtered_candidates1 = generate_random_candidates(required_num)
    filtered_candidates2 = generate_random_based(
        required_num - len(filtered_candidates1)
    )
    filtered_candidates = coord.SkyCoord(
        ra=np.concatenate([filtered_candidates1.ra, filtered_candidates2.ra]),
        dec=np.concatenate([filtered_candidates1.dec, filtered_candidates2.dec]),
        unit="deg",
    )

    names = [
        f"Rand {l:.3f}{b:+.3f}"
        for l, b in zip(
            filtered_candidates.galactic.l.degree, filtered_candidates.galactic.b.degree
        )
    ]

    # frame = pd.DataFrame(
    #     np.array([
    #         names,
    #         filtered_candidates.ra.deg,
    #         filtered_candidates.dec.deg]).T,
    #     columns=["name", "ra_deg", "dec_deg"],
    # )

    # frame["source"] = DataSource.RANDOM.value
    # frame["target"] = IsCluster.NOT_CLUSTER.value

    frame = pd.DataFrame(
        {
            "name": names,
            "ra_deg": filtered_candidates.ra.deg,
            "dec_deg": filtered_candidates.dec.deg,
            "source": DataSource.RANDOM.value,
            "target": IsCluster.NOT_CLUSTER.value,
        }
    )

    return inherit_columns(frame)


"""Split samples into train, validation and tests and get pictures from legacy survey"""


# MadCows пока что при обучении моделей не использовать, не забыть вывести вероятности для test_sample при тесте
def train_val_test_split() -> dict[DataPart, pd.DataFrame]:
    """Splits the dataset into training, validation, and test sets.

    Returns:
        dict[DataPart, pd.DataFrame]: Dictionary mapping data parts to DataFrames.
    """
    clusters = datasets_collection.load_expanded_clusters()
    non_clusters = datasets_collection.load_non_clusters()
    random = generate_random_sample()

    frame = pd.concat([clusters, non_clusters, random], ignore_index=True)

    frame = frame.sample(frac=1).reset_index(drop=True)

    for part in list(DataPart):
        path = os.path.join(settings.DATA_PATH, part.value)
        os.makedirs(path, exist_ok=True)

    train, validate, test = np.split(
        frame, [int(0.6 * len(frame)), int(0.8 * len(frame))]
    )

    validate = validate.reset_index(drop=True)
    validate.index.name = "idx"

    test = test.reset_index(drop=True)
    test.index.name = "idx"

    train_counts = {
        "is_cluster_1": train[train["target"] == 1].shape[0],
        "source_rand": train[train["source"] == "rand"].shape[0],
        "source_sga": train[train["source"] == "sga"].shape[0],
        "is_cluster_0_stars": train[
            (train["target"] == 0)
            & (train["source"] != "sga")
            & (train["source"] != "rand")
        ].shape[0],
    }

    test_counts = {
        "is_cluster_1": test[test["target"] == 1].shape[0],
        "source_rand": test[test["source"] == "rand"].shape[0],
        "source_sga": test[test["source"] == "sga"].shape[0],
        "is_cluster_0_stars": train[
            (train["target"] == 0)
            & (train["source"] != "sga")
            & (train["source"] != "rand")
        ].shape[0],
    }

    print("Train Counts:", train_counts)
    print("Test Counts:", test_counts)

    test_sample = datasets_collection.load_test_sample()
    bright_stars = datasets_collection.load_bright_stars()

    pairs = [
        (DataPart.TRAIN, train),
        (DataPart.VALIDATE, validate),
        (DataPart.TEST, test),
        (DataPart.TEST_SAMPLE, test_sample),
        (DataPart.BRIGHT_STARS, bright_stars),
    ]

    return dict(pairs)


def ddos() -> None:
    """Generates cutouts for all data splits and saves them."""

    description_path = os.path.join(settings.DATA_PATH, "description")
    os.makedirs(description_path, exist_ok=True)

    pairs = train_val_test_split()
    for part, description in pairs.items():

        description_file_path = os.path.join(description_path, f"{part.value}.csv")

        description.to_csv(description_file_path, index=True)

        path = os.path.join(settings.DATA_PATH, part.value)

        g = grabber.Grabber()
        g.grab_cutouts(targets=description, output_dir=path)


def create_dataloaders() -> tuple[dict[DataPart, Dataset], dict[DataPart, DataLoader]]:
    """Creates datasets and dataloaders for each data part.

    Returns:
        tuple[dict[DataPart, Dataset], dict[DataPart, DataLoader]]:
            Dictionary of datasets and corresponding dataloaders.
    """

    download_data()
    ddos()

    data_transforms = defaultdict(lambda: transforms.Compose(main_transforms))

    data_transforms[DataPart.TRAIN] = transforms.Compose(
        [
            *main_transforms,
            transforms.RandomRotation(
                15,
            ),
            transforms.RandomHorizontalFlip(),
        ]
    )

    custom_datasets = {}
    dataloaders = {}
    for part in list(DataPart):
        if part == DataPart.MC:
            continue

        cluster_dataset = ClusterDataset(
            os.path.join(settings.DATA_PATH, part.value),
            os.path.join(settings.DESCRIPTION_PATH, f"{part.value}.csv"),
        )

        custom_datasets[part] = cluster_dataset
        dataloaders[part] = DataLoader(cluster_dataset, batch_size=settings.BATCH_SIZE)

    # ? Not working for non-image data
    # # Get a batch of training data
    # batch = next(iter(dataloaders[DataPart.TRAIN]))

    # # Make a grid from batch
    # out = utils.make_grid(batch["image"])
    # show_original(out)

    return custom_datasets, dataloaders
