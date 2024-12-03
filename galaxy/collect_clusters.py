"""Script to collect cluster datasets"""

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
from astroquery.vizier import Vizier
from pixell import enmap
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from galaxy import grabber, util
from galaxy.config import settings
from galaxy.util import inherit_columns, DataSource, IsCluster


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


'''Пока что не использовать в обучении модели'''
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
    
    frame = frame[frame["red_shift"].notna() & frame["n_z"].str.contains(r'\+', na=False)]
    frame["red_shift_type"] = "spec"


    frame = frame.loc[:, ["ra_deg", "dec_deg", "name", "red_shift", "red_shift_type"]]
    frame["source"] = DataSource.SPT_SZ.value
    frame["is_cluster"] = IsCluster.IS_CLUSTER.value

    frame = inherit_columns(frame)

    return frame


def read_pszspt():
    CATALOGUE = "J/A+A/647/A106"
    frame = util.read_vizier(CATALOGUE)

    frame = frame.rename(columns={"Name": "name",
                            "RAJ2000": "ra_deg",
                            "DEJ2000": "dec_deg",
                            "z": "red_shift"}
                    )
    
    frame = frame[frame["red_shift"].notna()]

    frame = frame.loc[:, ["ra_deg", "dec_deg", "name", "red_shift"]]

    # данные берутся из psz и spt (предлагается ставить phot)
    frame["red_shift_type"] = "phot"

    frame["source"] = DataSource.PSZSPT.value
    frame["is_cluster"] = IsCluster.IS_CLUSTER.value

    frame = inherit_columns(frame)

    return frame


def read_comprass():
    CATALOGUE = "J/A+A/626/A7/comprass"
    frame = util.read_vizier(CATALOGUE)

    frame = frame.rename(columns={"Name": "name",
                            "RAJ2000": "ra_deg",
                            "DEJ2000": "dec_deg",
                            "z": "red_shift"}
                    )

    frame = frame[frame["red_shift"].notna()]

    frame = frame.loc[:, ["ra_deg", "dec_deg", "name", "red_shift"]]

    # данные берутся из других каталогов (пока что предлагается ставить phot)
    frame["red_shift_type"] = "phot"

    frame["source"] = DataSource.CCOMPRASS.value
    frame["is_cluster"] = IsCluster.IS_CLUSTER.value

    frame = inherit_columns(frame)

    return frame


def read_spt2500d():
    CATALOGUE = "J/ApJ/878/55/table5"
    frame = util.read_vizier(CATALOGUE)

    frame = frame.rename(columns={"SPT-CL": "name",
                            "RAJ2000": "ra_deg",
                            "DEJ2000": "dec_deg",
                            "z": "red_shift"}
                    )

    frame = frame[frame["red_shift"].notna() & frame["n_z"].str.contains(r'\+', na=False)]

    frame = frame.loc[:, ["ra_deg", "dec_deg", "name", "red_shift"]]

    frame["red_shift_type"] = "spec"
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

    frame = frame[frame["red_shift"].notna()]

    frame = frame.loc[:, ["ra_deg", "dec_deg", "name", "red_shift"]]

    frame["red_shift_type"] = "phot"

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


def read_spt100():
    table: atpy.Table = atpy.Table().read(settings.SPT100_PATH)

    frame = table.to_pandas().reset_index(drop=True)

    frame = frame.rename(
        columns={
            "SPT_ID": "name", 
            "RA": "ra_deg", 
            "Dec": "dec_deg", 
            "redshift": "red_shift"
            }
    )

    frame["name"] = frame["name"].astype(str)
    frame = frame[frame["redshift_unc"] == 0]

    frame = frame.loc[:, ["ra_deg", "dec_deg", "name", "red_shift"]]

    frame["red_shift_type"] = "spec"

    frame["source"] = DataSource.SPT100.value
    frame["is_cluster"] = IsCluster.IS_CLUSTER.value

    frame = inherit_columns(frame)

    return frame


'''
TODO: Для test_sample выдавать при тесте полученные вероятности, можно просто добавлять колонку
Можно попробовать визуализировать на графике масса - красное смещение с колорбаром в виде вероятностей
'''
def read_test_sample():
    frame = pd.read_csv(settings.TEST_SAMPLE_PATH)

    frame = frame.rename(
        columns={
            "RADeg": "ra_deg",
            "decDeg": "dec_deg",
            "redshift": "red_shift",
        }
    )

    frame = frame.loc[:, ["ra_deg", "dec_deg", "name", "red_shift"]]

    frame["source"] = DataSource.TEST_SAMPLE.value
    frame["is_cluster"] = IsCluster.IS_CLUSTER.value

    frame = inherit_columns(frame)

    return frame