"""Script to collect cluster datasets"""

import os

import astropy.table as atpy
import numpy as np
import pandas as pd
from astropy.coordinates import Angle
from astroquery.vizier import Vizier

from galaxy import util
from galaxy.config import settings
from galaxy.util import DataSource, IsCluster, inherit_columns


def read_dr5() -> pd.DataFrame:
    """Reads the DR5 catalog.

    Returns:
        pd.DataFrame: DataFrame containing DR5 cluster data.
    """

    table: atpy.Table = atpy.Table().read(settings.DR5_CLUSTERS_PATH)

    frame = table.to_pandas().reset_index(drop=True)

    frame["name"] = frame["name"].astype(str)
    frame["redshiftType"] = frame["redshiftType"].astype(str)
    frame = frame.rename(
        columns={
            "RADeg": "ra_deg",
            "decDeg": "dec_deg",
            "redshift": "red_shift",
            "redshiftType": "red_shift_type",
        }
    )

    frame = frame.loc[:, ["ra_deg", "dec_deg", "name", "red_shift", "red_shift_type"]]
    frame["source"] = DataSource.DR5.value
    frame["target"] = IsCluster.IS_CLUSTER.value

    frame = inherit_columns(frame)

    return frame


"""Пока что не использовать в обучении модели"""


def read_mc() -> pd.DataFrame:
    """Reads the MaDCoWS catalog from VizieR.

    Returns:
        pd.DataFrame: DataFrame containing MaDCoWS cluster data.
    """
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

    frame["source"] = DataSource.MC.value
    frame["target"] = IsCluster.IS_CLUSTER.value

    frame = inherit_columns(frame)

    return frame





def read_spt100() -> pd.DataFrame:
    """Reads the SPT100 catalog.

    Returns:
        pd.DataFrame: DataFrame containing SPT100 cluster data.
    """
    table: atpy.Table = atpy.Table().read(settings.SPT100_PATH)

    frame = table.to_pandas().reset_index(drop=True)

    frame = frame.rename(
        columns={
            "SPT_ID": "name",
            "RA": "ra_deg",
            "Dec": "dec_deg",
            "redshift": "red_shift",
        }
    )

    frame["name"] = frame["name"].astype(str)
    frame = frame[frame["redshift_unc"] == 0]

    frame = frame.loc[:, ["ra_deg", "dec_deg", "name", "red_shift"]]

    frame["red_shift_type"] = "spec"

    frame["source"] = DataSource.SPT100.value
    frame["target"] = IsCluster.IS_CLUSTER.value

    frame = inherit_columns(frame)

    return frame


"""
TODO: Для test_sample выдавать при тесте полученные вероятности, можно просто добавлять колонку
Можно попробовать визуализировать на графике масса - красное смещение c колорбаром в виде вероятностей
"""


def read_test_sample() -> pd.DataFrame:
    """Reads the test sample dataset.

    Returns:
        pd.DataFrame: DataFrame containing test sample cluster data.
    """
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
    frame["target"] = IsCluster.IS_CLUSTER.value

    frame = inherit_columns(frame)

    return frame


def read_act_mcmf(row_limit = 1000) -> pd.DataFrame:
    """Reads the ACT_MCMF catalog from VizieR.

    Returns:
        pd.DataFrame: DataFrame containing ACT_MCMF cluster data.
    """
    CATALOGUE = "J/A+A/690/A322/"

    catalog_list = Vizier.find_catalogs(CATALOGUE)
    Vizier.ROW_LIMIT = row_limit
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

    mc_frame = mc_frame.loc[
        :, ["ra_deg", "dec_deg", "name", "red_shift", "red_shift_type"]
    ]
    mc_frame["source"] = DataSource.ACT_MCMF.value
    mc_frame["target"] = IsCluster.IS_CLUSTER.value
    mc_frame = inherit_columns(mc_frame)

    return mc_frame
