"""Script to get dataloaders"""

"""To create dataloaders you need to adress only function create_dataloaders()"""

import os
from collections import defaultdict
from enum import Enum
from pathlib import Path

import astropy.coordinates as coord
import astropy.table as atpy
import astropy.units as u
import numpy as np
import pandas as pd
import torch
import wget
from astropy.coordinates import Angle
from astropy.io import fits
from astroquery.gaia import Gaia
from astroquery.vizier import Vizier
from pixell import enmap
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from galaxy import collect_clusters, collect_not_clusters, grabber, util
from galaxy.config import settings

from galaxy.util import inherit_columns, DataSource, IsCluster


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
    GAIA = "gaia"

# class SurveyLayer(str, Enum):
#     UNWISE_NEO7 = "unwise-neo7"
#     VLASS1_2 = "vlass1.2"


import torch
import torch.nn.functional as F

def shift_and_mirror_pad(image, shift_x, shift_y):
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
    padded_image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')

    # Calculate cropping indices
    crop_left = pad_right
    crop_right = crop_left + W
    crop_top = pad_bottom
    crop_bottom = crop_top + H

    # Crop the image to original size
    shifted_image = padded_image[:, crop_top:crop_bottom, crop_left:crop_right]

    return shifted_image

class ShiftAndMirrorPadTransform:
    def __init__(self, max_shift_x: int = 20, max_shift_y: int = 20):
        self.max_shift_x = max_shift_x
        self.max_shift_y = max_shift_y

    def __call__(self, image):
        # Random shift values within the specified range
        shift_x = torch.randint(-self.max_shift_x, self.max_shift_x + 1, (1,)).item()
        shift_y = torch.randint(-self.max_shift_y, self.max_shift_y + 1, (1,)).item()
        return shift_and_mirror_pad(image, shift_x, shift_y)


class ClusterDataset(Dataset):
    def __init__(self, images_dir_path: str, description_csv_path: str, transform=None):
        super().__init__()

        self.images_dir_path = images_dir_path
        self.description_df = pd.read_csv(
            description_csv_path, index_col=0, dtype={"target": int}
        )

        self.description_df.index = self.description_df.index.astype(str)
        self.description_df.loc[:, "red_shift"] = self.description_df[
            "red_shift"
        ].astype(float)
        self.description_df.index.name = "idx"

        self.transform = transform

    def __len__(self):
        return len(self.description_df)

    def __getitem__(self, idx):

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
    def _read_img(fits_path: Path):
        with fits.open(fits_path) as hdul:
            img = torch.Tensor(hdul[0].data.astype(np.float64))

        return img


"""Collecting clusters of galaxies for positive class"""

def download_data():

    for config in [settings.MAP_ACT_CONFIG, 
                   settings.DR5_CONFIG, 
                   settings.SGA_CONFIG,
                   settings.SPT100_CONFIG
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

def get_positive_class():
    dr5 = collect_clusters.read_dr5()
    upc_sz = collect_clusters.read_upc_sz()
    spt_sz = collect_clusters.read_spt_sz()
    pszspt = collect_clusters.read_pszspt()
    comprass = collect_clusters.read_comprass()
    spt2500d = collect_clusters.read_spt2500d()
    sptecs = collect_clusters.read_sptecs()
    spt100 = collect_clusters.read_spt100()

    clusters = pd.concat([
        dr5, 
        upc_sz,
        spt_sz,
        pszspt,
        comprass,
        spt2500d,
        sptecs,
        spt100
        ], ignore_index=True)

    return clusters


def get_cluster_catalog() -> coord.SkyCoord:
    clusters = get_positive_class()
    test_sample = collect_clusters.read_test_sample()

    clusters = pd.concat([
        clusters,
        test_sample
        ], ignore_index=True)

    # The catalog of known found galaxies
    catalog = coord.SkyCoord(
        ra=clusters["ra_deg"] * u.degree, dec=clusters["dec_deg"] * u.degree, unit="deg"
    )

    return catalog


def get_negative_class():
    sga = collect_not_clusters.read_sga()
    tyc2 = collect_not_clusters.read_tyc2()
    gaia = collect_not_clusters.read_gaia()

# drop columns "red_shift" and "red_shift_type", as tables with NaN columns cannot be concatenated properly(?)
    columns_to_drop = ["red_shift", "red_shift_type"]
    sga = sga.drop(
        columns=[col for col in columns_to_drop if col in sga.columns],
          errors='ignore'
          )
    tyc2 = tyc2.drop(
        columns=[col for col in columns_to_drop if col in tyc2.columns], 
        errors='ignore'
        )
    gaia = gaia.drop(
        columns=[col for col in columns_to_drop if col in gaia.columns], 
        errors='ignore'
        )

    negative_class = pd.concat([sga, tyc2, gaia], axis=0, ignore_index=True)
    negative_class = inherit_columns(negative_class)
    return negative_class


def get_non_cluster_catalog() -> coord.SkyCoord:
    non_clusters = get_negative_class()

    # The catalog of known found galaxies
    catalog = coord.SkyCoord(
        ra=non_clusters["ra_deg"] * u.degree, dec=clusnon_clustersters["dec_deg"] * u.degree, unit="deg"
    )

    return catalog


"""Для отбора участков неба, удалённых от объектов из собранных датасетов"""
def filter_candiates(candidates: coord.SkyCoord, max_len: int) -> coord.SkyCoord:
    clusters_catalog = get_cluster_catalog()
    non_clusters_catalog = get_non_cluster_catalog()

    catalog = coord.SkyCoord(
        ra=np.concatenate([clusters_catalog.ra.deg, non_clusters_catalog.ra.deg]) * u.degree,
        dec=np.concatenate([clusters_catalog.dec.deg, non_clusters_catalog.dec.deg]) * u.degree,
        unit="deg"
    )

    _, d2d, _ = candidates.match_to_catalog_sky(catalog)

    MIN_ANGLE = 10
    MAX_ANGLE = 20

    candidates_filter = (d2d.arcmin > MIN_ANGLE) & (
        candidates.galactic.b.degree > MAX_ANGLE
    )

    filtered_candidates = candidates[candidates_filter][:max_len]

    return filtered_candidates


def generate_candidates_dr5() -> coord.SkyCoord:

    # Needed only for reading metadata and map generation?
    imap_98 = enmap.read_fits(settings.MAP_ACT_PATH)[0]

    # Generating positions of every pixel of telescope's sky zone
    positions = np.array(np.rad2deg(imap_98.posmap()))

    ras, decs = positions[1].ravel(), positions[0].ravel()

    # Shuffling candidates, imitating samples
    np.random.seed(settings.SEED)
    np.random.shuffle(ras)
    np.random.shuffle(decs)

    # Just points from our sky map
    candidates = coord.SkyCoord(ra=ras * u.degree, dec=decs * u.degree, unit="deg")

    return candidates


def generate_candidates_mc() -> coord.SkyCoord:

    n_sim = 10_000

    np.random.seed(settings.SEED)

    ras = np.random.uniform(0, 360, n_sim)
    decs = np.random.uniform(-90, 90, n_sim)

    # Just points from our sky map
    candidates = coord.SkyCoord(ra=ras * u.degree, dec=decs * u.degree, unit="deg")

    return candidates


def create_negative_class_dr5():
    """Create sample from dr5 clsuter catalogue"""

    dr5 = collect_clusters.read_dr5()

    candidates = generate_candidates_dr5()

    filtered_candidates = filter_candiates(candidates, max_len=len(dr5))
    names = [f"Rand {l:.3f}{b:+.3f}" for l, b in zip(
                filtered_candidates.galactic.l.degree, 
                filtered_candidates.galactic.b.degree
                )
            ]

    frame = pd.DataFrame(
        np.array([
            names, 
            filtered_candidates.ra.deg, 
            filtered_candidates.dec.deg]).T,
        columns=["name", "ra_deg", "dec_deg"],
    )

    frame["source"] = DataSource.RANDOM.value
    frame["is_cluster"] = IsCluster.NOT_CLUSTER.value

    frame = inherit_columns(frame)

    return frame


def create_negative_class_mc():
    """Create sample of negative class to compensate MadCows catalogue"""

    mc =  collect_clusters.read_mc()

    candidates = generate_candidates_mc()

    filtered_candidates = filter_candiates(candidates, max_len=len(mc))
    names = [f"Rand {l:.3f}{b:+.3f}" for l, b in zip(
            filtered_candidates.galactic.l.degree, 
            filtered_candidates.galactic.b.degree
            )
        ]

    frame = pd.DataFrame(
        np.array([
            names, 
            filtered_candidates.ra.deg, 
            filtered_candidates.dec.deg]).T,
        columns=["name", "ra_deg", "dec_deg"],
    )

    frame["source"] = DataSource.RANDOM.value
    frame["is_cluster"] = IsCluster.NOT_CLUSTER.value

    frame = inherit_columns(frame)

    return frame


"""Split samples into train, validation and tests and get pictures from legacy survey"""

def expand_positive_class():
    """
    Для того чтобы выборка была сбалансирована, координаты подтверждённых скоплений 
    немного шатаются => выборка увеличивается 
    """
    clusters = get_positive_class()

    transform = ShiftAndMirrorPadTransform()

    more_clusters = []

# TODO: нужно ли переименовывать "пошатанные" скопления + сохранять для них тот же red_shit?
    for _, row in clusters.iterrows():
        for _ in range(5):  
            new_ra, new_dec = transform(row["ra_deg"], row["dec_deg"])
            more_clusters.append({
                "name": row["name"],
                "ra_deg": new_ra,
                "dec_deg": new_dec,
                "red_shift": row["red_shift"], 
                "red_shift_type": row["red_shift_type"],
                "is_cluster": IsCluster.IS_CLUSTER.value,
                "source": row["source"]
            })

    more_clusters_df = pd.DataFrame(more_clusters)

    extended_clusters = pd.concat([
        clusters, 
        more_clusters_df,
        ], ignore_index=True)

    print(extended_clusters)


# TODO: свалить всё в один датасет, перемешать и нарезать train/val/test в соотношении 60/20/20
# MadCows пока что при обучении моделей не использовать, не забыть вывести вероятности для test_sample при тесте

def train_val_test_split():
    clusters = get_positive_class()
    non_clusters = get_negative_class()

    # for part in list(DataPart):
    #     path = os.path.join(settings.DATA_PATH, part.value)
    #     os.makedirs(path, exist_ok=True)

    # train, validate, test_dr5 = np.split(
    #     dr5, [int(0.6 * len(dr5)), int(0.8 * len(dr5))]
    # )

    # validate = validate.reset_index(drop=True)
    # validate.index.name = "idx"

    # test_dr5 = test_dr5.reset_index(drop=True)
    # test_dr5.index.name = "idx"

    # gaia = create_data_gaia()

    # pairs = [
    #     (DataPart.TRAIN, train),
    #     (DataPart.VALIDATE, validate),
    #     (DataPart.TEST, test_dr5),
    #     (DataPart.MC, test_mc),
    #     (DataPart.GAIA, gaia),
    # ]

    # return dict(pairs)


def ddos():

    description_path = os.path.join(settings.DATA_PATH, "description")
    os.makedirs(description_path, exist_ok=True)

    pairs = train_val_test_split()
    for part, description in pairs.items():

        description_file_path = os.path.join(description_path, f"{part.value}.csv")

        description.to_csv(description_file_path, index=True)

        path = os.path.join(settings.DATA_PATH, part.value)

        g = grabber.Grabber()
        g.grab_cutouts(targets=description, output_dir=path)


def create_dataloaders():

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

        cluster_dataset = ClusterDataset(
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
