"""Script to collect data for negative class"""

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

from galaxy.util import inherit_columns, DataSource, IsCluster

"""Galaxies in short distances"""
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
    seps = np.linspace(0, frame["red_shift"].max(), num=n_bins + 1)
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

    sample['source'] = DataSource.SGA.value
    sample["is_cluster"] = IsCluster.NOT_CLUSTER.value

    sample = inherit_columns(sample)

    return sample



"""Brigt stars"""
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


"""Stars"""
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