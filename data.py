"""Script to get dataloaders"""

"""To create dataloaders you need to adress only function create_dataloaders()"""

import os
import sys
from enum import Enum
from pathlib import Path
from zipfile import ZipFile

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

import legacy_for_img
from config import settings


np.random.seed(settings.SEED)

TORCHVISION_MEAN = [23.19058950345032, 22.780995295792817]
TORCHVISION_STD = [106.89880134344101, 100.32284196853638]

main_transforms = [
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=TORCHVISION_MEAN, std=TORCHVISION_STD),
]


class DataSource(str, Enum):
    ACT_MCMF = "act_mcmf"


class DataPart(str, Enum):
    TRAIN = "train"
    VALIDATE = "validate"
    TEST_DR5 = "test_dr5"
    TEST_MC = "test_mc"
    GAIA = "gaia"
    ACT_MCMF = "act_mcmf"


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


def bar_progress(current, total, width=80):
    progress_message = "Downloading: %d%% [%d / %d] bytes" % (
        current / total * 100,
        current,
        total,
    )
    # Don't use print() as it will print in new line every time.
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


"""Obtain GAIA stars catalogue"""


def read_gaia():
    job = Gaia.launch_job_async(
        "select DESIGNATION, ra, dec from gaiadr3.gaia_source "
        "where random_index between 0 and 1000000 and phot_g_mean_mag < 12 and parallax is not null"
    )
    gaiaResponse = job.get_results().to_pandas()
    gaia_frame = (
        gaiaResponse.sample(frac=1, random_state=settings.SEED)
        .reset_index(drop=True)
        .rename(columns={"DESIGNATION": "name", "ra": "ra_deg", "dec": "dec_deg"})
    )

    return gaia_frame


"""Obtain ACT_DR5, clusters identified there and in MaDCoWS"""


def download_data():

    for config in [settings.MAP_ACT_CONFIG, settings.DR5_CONFIG]:

        if not os.path.exists(config.OUTPUT_PATH):
            try:
                wget.download(url=config.URL, out=settings.DATA_PATH, bar=bar_progress)
                with ZipFile(config.ZIPPED_OUTPUT_PATH, "r") as zObject:
                    zObject.extractall(path=settings.DATA_PATH)
                rename_dict = config.RENAME_DICT

                os.rename(rename_dict.SOURCE, rename_dict.TARGET)
                os.remove(config.ZIPPED_OUTPUT_PATH)
            except Exception:
                # Getting 403, what credentials needed?
                wget.download(
                    url=config.FALLBACK_URL,
                    out=config.OUTPUT_PATH,
                    bar=bar_progress,
                )


def read_dr5():

    dr5: atpy.Table = atpy.Table().read(settings.DR5_CLUSTERS_PATH)
    dr5_frame = dr5.to_pandas().reset_index(drop=True)

    dr5_frame["name"] = dr5_frame["name"].astype(str)

    dr5_frame = dr5_frame.rename(
        columns={
            "RADeg": "ra_deg",
            "decDeg": "dec_deg",
            "redshift": "red_shift",
            "redshiftType": "red_shift_type",
        }
    )
    dr5_frame = dr5_frame.reset_index(drop=True)
    dr5_frame.index.name = "idx"
    dr5_frame = dr5_frame.reset_index(drop=False)

    return dr5_frame


def to_hms_format(time_str):
    parts = time_str.split()
    return f"{parts[0]}h{parts[1]}m{parts[2]}s"


def to_dms_format(time_str):
    parts = time_str.split()
    return f"{parts[0]}d{parts[1]}m{parts[2]}s"


required_columns = set(["idx", "ra_deg", "dec_deg", "name", "source"])
optional_columns = set(["red_shift", "red_shift_type"])


def inherit_columns(frame: pd.DataFrame):
    frame["idx"] = np.arange(len(frame))
    frame_columns = set(frame.columns)
    assert required_columns.issubset(frame_columns), "Some required columns are missed"
    missed_optional = optional_columns.difference(frame_columns)
    if missed_optional:
        for col in missed_optional:
            frame[col] = pd.NA
    frame = frame.reset_index(drop=True)
    return frame


def read_mc():
    # the catalogue of MaDCoWS in VizieR
    CATALOGUE = "J/ApJS/240/33/"

    catalog_list = Vizier.find_catalogs(CATALOGUE)

    Vizier.ROW_LIMIT = -1
    catalogs = Vizier.get_catalogs(catalog_list.keys())

    interesting_table: atpy.Table = catalogs[os.path.join(CATALOGUE, "table3")]
    mc_frame = interesting_table.to_pandas().reset_index(drop=True)

    mc_frame["ra_deg"] = mc_frame["RAJ2000"].apply(
        lambda x: Angle(to_hms_format(x)).degree
    )
    mc_frame["dec_deg"] = mc_frame["DEJ2000"].apply(
        lambda x: Angle(to_dms_format(x)).degree
    )

    mc_frame = mc_frame.rename(columns={"Name": "name"})

    mc_frame["red_shift"] = np.where(
        mc_frame["Specz"].notna(), mc_frame["Specz"], mc_frame["Photz"]
    )
    mc_frame["red_shift_type"] = np.where(mc_frame["Specz"].notna(), "spec", np.nan)
    mc_frame["red_shift_type"] = np.where(
        mc_frame["Photz"].notna() & mc_frame["red_shift_type"].isna(), "phot", np.nan
    )

    return mc_frame


def read_act_mcmf():
    """
    Obtain Vizier ACT_MCMF Catalogue.
    """
    CATALOGUE = "J/A+A/690/A322/"

    catalog_list = Vizier.find_catalogs(CATALOGUE)
    Vizier.ROW_LIMIT = -1
    catalogs = Vizier.get_catalogs(catalog_list.keys())

    interesting_table: atpy.Table = catalogs[os.path.join(CATALOGUE, "catalog")]
    mc_frame = interesting_table.to_pandas().reset_index(drop=True)

    mc_frame = mc_frame.rename(
        columns={"Name": "name", "RAJ2000": "ra_deg", "DEJ2000": "dec_deg"}
    )

    mc_frame["red_shift"] = np.where(
        mc_frame["zsp1"].notna(),
        mc_frame["zsp1"],
        np.where(mc_frame["z1C"].notna(), mc_frame["z1C"], mc_frame["z2C"]),
    )

    mc_frame["red_shift_type"] = np.where(
        mc_frame["zsp1"].notna(),
        "spec",
        np.where(mc_frame["z1C"].notna(), "z1C", "z2C"),
    )

    return mc_frame


def get_all_clusters():
    """Concat clusters from act_dr5 and madcows to create negative classes in samples"""

    mc = read_mc()
    dr5 = read_dr5()

    needed_cols = ["name", "ra_deg", "dec_deg"]
    clusters_dr5_mc = pd.concat([dr5[needed_cols], mc[needed_cols]], ignore_index=True)

    return clusters_dr5_mc


def get_cluster_catalog() -> coord.SkyCoord:

    clusters = get_all_clusters()

    # The catalog of known found galaxies
    catalog = coord.SkyCoord(
        ra=clusters["ra_deg"] * u.degree, dec=clusters["dec_deg"] * u.degree, unit="deg"
    )

    return catalog


def filter_candiates(candidates: coord.SkyCoord, max_len: int) -> coord.SkyCoord:

    catalog = get_cluster_catalog()

    _, d2d, _ = candidates.match_to_catalog_sky(catalog)

    MIN_ANGLE = 10
    MAX_ANGLE = 20

    candidates_filter = (d2d.arcmin > MIN_ANGLE) & (
        candidates.galactic.b.degree > MAX_ANGLE
    )

    filtered_candidates = candidates[candidates_filter][:max_len]

    return filtered_candidates


def candidates_to_df(candidates: coord.SkyCoord) -> pd.DataFrame:

    b_values = candidates.galactic.b.degree
    l_values = candidates.galactic.l.degree

    names = [f"Rand {l:.3f}{b:+.3f}" for l, b in zip(l_values, b_values)]

    data = pd.DataFrame(
        np.array([names, candidates.ra.deg, candidates.dec.deg]).T,
        columns=["name", "ra_deg", "dec_deg"],
    )

    return data


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


def generate_candidates_mc():
    """Create sample from MadCows catalogue"""

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

    frame = candidates_to_df(filtered_candidates)

    return frame


def create_negative_class_mc():

    mc = read_mc()

    candidates = generate_candidates_mc()

    filtered_candidates = filter_candiates(candidates, max_len=len(mc))

    frame = candidates_to_df(filtered_candidates)

    return frame


def create_data_dr5():
    clusters = read_dr5()
    clusters = clusters[["name", "ra_deg", "dec_deg", "red_shift", "red_shift_type"]]
    clusters["target"] = 1
    random = create_negative_class_dr5()
    random["target"] = 0
    data_dr5 = pd.concat([clusters, random]).reset_index(drop=True)
    data_dr5[["ra_deg", "dec_deg"]] = data_dr5[["ra_deg", "dec_deg"]].astype(float)

    data_dr5 = data_dr5.sample(frac=1, random_state=1)

    data_dr5.loc[:, "red_shift_type"] = data_dr5["red_shift_type"].astype(str)

    data_dr5 = data_dr5.reset_index(drop=True)
    data_dr5.index.name = "idx"

    return data_dr5


def create_data_mc():
    clusters = read_mc()
    clusters = clusters[["name", "ra_deg", "dec_deg", "red_shift", "red_shift_type"]]
    clusters["target"] = 1
    random = create_negative_class_mc()
    random["target"] = 0
    data_mc = pd.concat([clusters, random]).reset_index(drop=True)

    data_mc[["ra_deg", "dec_deg"]] = data_mc[["ra_deg", "dec_deg"]].astype(float)

    data_mc.loc[:, "red_shift_type"] = data_mc["red_shift_type"].astype(str)

    data_mc = data_mc.reset_index(drop=True)
    data_mc.index.name = "idx"

    return data_mc


def create_data_gaia():

    clusters = read_gaia()

    clusters["red_shift"] = np.nan
    clusters["red_shift_type"] = "nan"
    clusters["target"] = 0
    clusters.index.name = "idx"

    return clusters


def create_data_act_mcmf():
    clusters = read_act_mcmf()
    clusters = clusters[["name", "ra_deg", "dec_deg", "red_shift", "red_shift_type"]]
    clusters["target"] = 1
    random = create_negative_class_mc()
    random["target"] = 0
    data_mc = pd.concat([clusters, random]).reset_index(drop=True)

    data_mc[["ra_deg", "dec_deg"]] = data_mc[["ra_deg", "dec_deg"]].astype(float)

    data_mc.loc[:, "red_shift_type"] = data_mc["red_shift_type"].astype(str)

    data_mc = data_mc.reset_index(drop=True)
    data_mc.index.name = "idx"

    return data_mc


"""Split samples into train, validation and tests and get pictures from legacy survey"""


def train_val_test_split():
    dr5 = create_data_dr5()
    test_mc = create_data_mc()
    act_mcmf = create_data_act_mcmf()

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
        (DataPart.TEST_DR5, test_dr5),
        (DataPart.TEST_MC, test_mc),
        (DataPart.GAIA, gaia),
        (DataPart.ACT_MCMF, act_mcmf),
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
        legacy_for_img.grab_cutouts(
            target_file=description,
            name_col="name",
            ra_col="ra_deg",
            dec_col="dec_deg",
            output_dir=path,
            survey="unwise-neo7",
            imgsize_pix=224,
        )


"""Create dataloaders"""


# def show_original(img):
#     denormalized_img = img.clone()
#     for channel, m, s in zip(denormalized_img, TORCHVISION_MEAN, TORCHVISION_STD):
#         channel.mul_(s).add_(m)

#     denormalized_img = denormalized_img.numpy()
#     plt.imshow(np.transpose(denormalized_img, (1, 2, 0)))


def check_catalogs():

    is_map = os.path.exists(settings.MAP_ACT_PATH)
    is_dr5 = os.path.exists(settings.DR5_CLUSTERS_PATH)

    if not is_map or not is_dr5:
        download_data()


def create_dataloaders():

    check_catalogs()

    ddos()

    from collections import defaultdict

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
