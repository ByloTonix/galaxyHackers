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

from galaxy import grabber, util
from galaxy.config import settings


np.random.seed(settings.SEED)

TORCHVISION_MEAN = [23.19058950345032, 22.780995295792817]
TORCHVISION_STD = [106.89880134344101, 100.32284196853638]

main_transforms = [
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=TORCHVISION_MEAN, std=TORCHVISION_STD),
]


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

    CSV = "csv"
    RANDOM = "rand"


class DataPart(str, Enum):

    TRAIN = "train"
    VALIDATE = "validate"
    TEST = "test"
    MC = "mc"
    GAIA = "gaia"


class IsCluster(int, Enum):

    IS_CLUSTER = 1
    NOT_CLUSTER = 0

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


"""Obtain ACT_DR5, clusters identified there and in MaDCoWS"""


def download_data():

    for config in [settings.MAP_ACT_CONFIG, settings.DR5_CONFIG, settings.SGA_CONFIG]:

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


def read_dr5():

    table: atpy.Table = atpy.Table().read(settings.DR5_CLUSTERS_PATH)

    frame = table.to_pandas().reset_index(drop=True)

    frame["name"] = frame["name"].astype(str)

    frame = frame.rename(
        columns={
            "RADeg": "ra_deg",
            "decDeg": "dec_deg",
            "redshift": "red_shift",
            "redshiftType": "red_shift_type",
        }
    )


    frame = frame.loc[:, ["ra_deg", "dec_deg", "name", "red_shift", "red_shift_type"]]
    frame['source'] = DataSource.DR5.value
    frame["is_cluster"] = IsCluster.IS_CLUSTER.value

    frame = inherit_columns(frame)
    return frame


def read_mc():
    # the catalogue of MaDCoWS in VizieR
    CATALOGUE = "J/ApJS/240/33/"

    catalog_list = Vizier.find_catalogs(CATALOGUE)

    Vizier.ROW_LIMIT = -1
    catalogs = Vizier.get_catalogs(catalog_list.keys())

    interesting_table: atpy.Table = catalogs[os.path.join(CATALOGUE, "table3")]
    frame = interesting_table.to_pandas().reset_index(drop=True)

    frame["ra_deg"] = frame["RAJ2000"].apply(
        lambda x: Angle(util.to_hms_format(x)).degree
    )
    frame["dec_deg"] = frame["DEJ2000"].apply(
        lambda x: Angle(util.to_dms_format(x)).degree
    )

    frame = frame.rename(columns={"Name": "name"})

    frame["red_shift"] = np.where(
        frame["Specz"].notna(), frame["Specz"], frame["Photz"]
    )
    frame["red_shift_type"] = np.where(frame["Specz"].notna(), "spec", pd.NA)
    frame["red_shift_type"] = np.where(
        frame["Photz"].notna() & frame["red_shift_type"].isna(), "phot", pd.NA
    )

    frame = frame.loc[:, ["ra_deg", "dec_deg", "name", "red_shift", "red_shift_type"]]

    frame['source'] = DataSource.MC.value
    frame["is_cluster"] = IsCluster.IS_CLUSTER.value

    frame = inherit_columns(frame)

    return frame


def read_upc_sz():
    CATALOGUE = "J/ApJS/272/7/table2"
    frame = util.read_vizier(CATALOGUE)

    frame = frame.rename(columns={"PSZ2": "name",
                              "RAJ2000": "ra_deg",
                              "DEJ2000": "dec_deg",
                              "z": "red_shift",
                              "f_z": "red_shift_type"}
                     )
    # 'spec', '', 'phot', 'unct' - values in red_shift_type column. unct = uncertainty => skip?
    frame = frame[
    frame["red_shift"].notna() & frame["red_shift_type"].notna() & (frame["red_shift_type"] != "unct")
    ]

    frame = frame.loc[:, ["ra_deg", "dec_deg", "name", "red_shift", "red_shift_type"]]

    frame["source"] = DataSource.UPC_SZ.value
    frame["is_cluster"] = IsCluster.IS_CLUSTER.value

    frame = inherit_columns(frame)

    return frame


def read_spt_sz():
    CATALOGUE = "J/ApJS/216/27/table4"
    frame = util.read_vizier(CATALOGUE)

    frame = frame.rename(columns={"SPT-CL": "name",
                          "RAJ2000": "ra_deg",
                          "DEJ2000": "dec_deg",
                          "z": "red_shift",
                          "f_z": "red_shift_type"}
                  )
    
    # TODO: for red shift lower bounds are considered, adapt for script
    frame = frame[frame["red_shift"].notna()]

    frame = frame.loc[:, ["ra_deg", "dec_deg", "name", "red_shift", "red_shift_type"]]
    frame["source"] = DataSource.SPT_SZ.value
    frame["is_cluster"] = IsCluster.IS_CLUSTER.value

    frame = inherit_columns(frame)

    return frame


def read_pszspt():
    CATALOGUE = "J/A+A/647/A106"
    frame = util.read_vizier(CATALOGUE)

    # TODO: red shift is not specified in the table, adapt
    frame = frame.rename(columns={"Name": "name",
                            "RAJ2000": "ra_deg",
                            "DEJ2000": "dec_deg",
                            "z": "red_shift"}
                    )
    
    frame = frame[frame["red_shift"].notna()]

    frame = frame.loc[:, ["ra_deg", "dec_deg", "name", "red_shift"]]

    frame["source"] = DataSource.PSZSPT.value
    frame["is_cluster"] = IsCluster.IS_CLUSTER.value

    frame = inherit_columns(frame)

    return frame


def read_comprass():
    CATALOGUE = "J/A+A/626/A7/comprass"
    frame = util.read_vizier(CATALOGUE)

    # TODO: red shift is not specified in the table, adapt
    frame = frame.rename(columns={"Name": "name",
                            "RAJ2000": "ra_deg",
                            "DEJ2000": "dec_deg",
                            "z": "red_shift"}
                    )

    frame = frame[frame["red_shift"].notna()]

    frame = frame.loc[:, ["ra_deg", "dec_deg", "name", "red_shift"]]

    frame["source"] = DataSource.CCOMPRASS.value
    frame["is_cluster"] = IsCluster.IS_CLUSTER.value

    frame = inherit_columns(frame)

    return frame


def read_spt2500d():
    CATALOGUE = "J/ApJ/878/55/table5"
    frame = util.read_vizier(CATALOGUE)

    # TODO: red shift is not specified in the table, adapt
    frame = frame.rename(columns={"SPT-CL": "name",
                            "RAJ2000": "ra_deg",
                            "DEJ2000": "dec_deg",
                            "z": "red_shift"}
                    )

    frame = frame[frame["red_shift"].notna()]

    frame = frame.loc[:, ["ra_deg", "dec_deg", "name", "red_shift"]]

    frame["source"] = DataSource.SPT2500D.value
    frame["is_cluster"] = IsCluster.IS_CLUSTER.value

    frame = inherit_columns(frame)

    return frame


def collect_sptecs(catalogue):
    frame = util.read_vizier(catalogue)

    frame = frame.rename(columns={"SPT-CL": "name",
                        "RAJ2000": "ra_deg",
                        "DEJ2000": "dec_deg",
                        "z": "red_shift"}
                )

    # TODO: red shift is not specified in the table, adapt
    frame = frame[frame["red_shift"].notna()]

    frame = frame.loc[:, ["ra_deg", "dec_deg", "name", "red_shift"]]

    frame["source"] = DataSource.SPTECS.value
    frame["is_cluster"] = IsCluster.IS_CLUSTER.value

    frame = inherit_columns(frame)

    return frame


def read_sptecs():
    frame_certified = collect_sptecs("J/ApJS/247/25/table10")
    frame_candidates = collect_sptecs("J/ApJS/247/25/cand")

    frame = pd.concat([frame_certified, 
                       frame_candidates])
    return frame


def read_sga(sample_size=10_000):

    table: atpy.Table = atpy.Table().read(settings.SGA_PATH)
    frame = table.to_pandas().reset_index(drop=True)


    frame = frame.rename(
        columns={
            "SGA_ID": "name",
            "RA": "ra_deg",
            "DEC": "dec_deg",
            "Z_LEDA": "red_shift",
        }
    )

    # Нет точного указания, Светлана сказала скорее всего так.
    frame['red_shift_type'] = 'phot'


    frame = frame.loc[:, ["ra_deg", "dec_deg", "name", "red_shift", "red_shift_type"]]

    # Убираем выбросы слева
    frame = frame[frame.red_shift>0]

    # Убираем выбросы справа
    q_hi  = frame["red_shift"].quantile(0.995)
    frame = frame[(frame["red_shift"] < q_hi)]

    # Сэмплируем по бинам относительно red_shift. Хочется чтобы объекты в выборке были распределены равномерно
    n_bins = 10
    seps = np.linspace(0, frame["red_shift"].max(), num=n_bins+1)
    bins = zip(seps[:-1], seps[1:])

    # Вычисляем размер бина
    sub_sample_size = sample_size // n_bins
    sub_samples = []

    # Сэмплируем в каждом бине равномерно
    for low, high in bins:
        sub_frame = frame.loc[frame["red_shift"].between(low, high)]
        sub_samples.append(sub_frame.sample(n=sub_sample_size, random_state=settings.SEED))
    
    # Склеиваем
    sample = pd.concat(sub_samples, axis=0).sort_index()
    sample.index = np.arange(len(sample))

    frame['source'] = DataSource.SGA.value
    frame["is_cluster"] = IsCluster.NOT_CLUSTER.value

    frame = inherit_columns(frame)

    return frame


def read_tyc2(sample_size=5_000):

    frame = Vizier(row_limit=-1).get_catalogs(catalog='I/259/tyc2')
    frame: pd.DataFrame = frame[frame.keys()[0]].to_pandas().reset_index(drop=True)

    frame = frame.drop_duplicates("TYC2")


    frame = frame.rename(
            columns={
                "TYC2": "name",
                "RA_ICRS_": "ra_deg",
                "DE_ICRS_": "dec_deg",
            }
        )

    frame = frame.loc[:, ["ra_deg", "dec_deg", "name"]]

    frame = frame.sample(n=sample_size, random_state=settings.SEED)

    frame['source'] = DataSource.TYC2.value
    frame["is_cluster"] = IsCluster.NOT_CLUSTER.value

    frame = inherit_columns(frame)

    return frame


"""Obtain GAIA stars catalogue"""


def read_gaia():
    job = Gaia.launch_job_async(
        "select DESIGNATION, ra, dec from gaiadr3.gaia_source "
        "where random_index between 0 and 1000000 and phot_g_mean_mag < 12 and parallax is not null"
    )
    gaiaResponse = job.get_results().to_pandas()
    frame = (
        gaiaResponse.sample(frac=1, random_state=settings.SEED)
        .reset_index(drop=True)
        .rename(columns={
            "DESIGNATION": "name",
            "ra": "ra_deg", 
            "dec": "dec_deg"
            }
        )
    )

    frame["source"] = DataSource.GAIA.value
    frame["is_cluster"] = IsCluster.NOT_CLUSTER.value

    frame = inherit_columns(frame)

    return frame


def get_cluster_catalog() -> coord.SkyCoord:

    mc = read_mc()
    dr5 = read_dr5()

    clusters = pd.concat([dr5, mc], ignore_index=True)

    # The catalog of known found galaxies
    catalog = coord.SkyCoord(
        ra=clusters["ra_deg"] * u.degree, dec=clusters["dec_deg"] * u.degree, unit="deg"
    )

    return catalog


def filter_candiates(candidates: coord.SkyCoord, max_len: int) -> coord.SkyCoord:

    # TODO Написать не только относительно скоплений, но и относительно галактик и звёзд
    catalog = get_cluster_catalog()

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
    """Create sample of negative class to compensate MadCows catalogue"""

    n_sim = 10_000

    np.random.seed(settings.SEED)

    ras = np.random.uniform(0, 360, n_sim)
    decs = np.random.uniform(-90, 90, n_sim)

    # Just points from our sky map
    candidates = coord.SkyCoord(ra=ras * u.degree, dec=decs * u.degree, unit="deg")

    return candidates


def create_negative_class_dr5():
    """Create sample from dr5 clsuter catalogue"""

    dr5 = read_dr5()

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

    mc = read_mc()

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


# TODO: уточнить какие датасеты куда относить и как разделять
def create_data_train():
    dr5 = read_dr5()
    dr5_negative = create_negative_class_dr5()

    frame = pd.concat([
        dr5, 
        dr5_negative,
        ]
    ).reset_index(drop=True)

    return frame


def create_data_test():
    mc = read_gaia()
    mc_negative = create_negative_class_mc()

    frame = pd.concat([
        mc, 
        mc_negative,
        ]
    ).reset_index(drop=True)
    pass


"""Split samples into train, validation and tests and get pictures from legacy survey"""


def train_val_test_split():
    dr5 = create_data_dr5()
    test_mc = create_data_mc()

    for part in list(DataPart):
        path = os.path.join(settings.DATA_PATH, part.value)
        os.makedirs(path, exist_ok=True)

    train, validate, test_dr5 = np.split(
        dr5, [int(0.6 * len(dr5)), int(0.8 * len(dr5))]
    )

    validate = validate.reset_index(drop=True)
    validate.index.name = "idx"

    test_dr5 = test_dr5.reset_index(drop=True)
    test_dr5.index.name = "idx"

    gaia = create_data_gaia()

    pairs = [
        (DataPart.TRAIN, train),
        (DataPart.VALIDATE, validate),
        (DataPart.TEST, test_dr5),
        (DataPart.MC, test_mc),
        (DataPart.GAIA, gaia),
    ]

    return dict(pairs)


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
