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


def read_upc_sz() -> pd.DataFrame:
    """Reads the UPC_SZ catalog from VizieR.

    Returns:
        pd.DataFrame: DataFrame containing UPC_SZ cluster data.
    """
    CATALOGUE = "J/ApJS/272/7/table2"
    frame = util.read_vizier(CATALOGUE)

    frame = frame.rename(
        columns={
            "PSZ2": "name",
            "RAJ2000": "ra_deg",
            "DEJ2000": "dec_deg",
            "z": "red_shift",
            "f_z": "red_shift_type",
        }
    )
    # 'spec', '', 'phot', 'unct' - values in red_shift_type column. unct = uncertainty => skip?
    frame = frame[
        frame["red_shift"].notna()
        & frame["red_shift_type"].notna()
        & (frame["red_shift_type"] != "unct")
    ]

    frame = frame.loc[:, ["ra_deg", "dec_deg", "name", "red_shift", "red_shift_type"]]

    frame["source"] = DataSource.UPC_SZ.value
    frame["target"] = IsCluster.IS_CLUSTER.value

    frame = inherit_columns(frame)

    return frame


def read_spt_sz() -> pd.DataFrame:
    """Reads the SPT_SZ catalog from VizieR.

    Returns:
        pd.DataFrame: DataFrame containing SPT_SZ cluster data.
    """
    CATALOGUE = "J/ApJS/216/27/table4"
    frame = util.read_vizier(CATALOGUE)

    frame = frame.rename(
        columns={
            "SPT-CL": "name",
            "RAJ2000": "ra_deg",
            "DEJ2000": "dec_deg",
            "z": "red_shift",
            "f_z": "red_shift_type",
        }
    )

    frame = frame[
        frame["red_shift"].notna() & frame["n_z"].str.contains(r"\+", na=False)
    ]
    frame["red_shift_type"] = "spec"

    frame = frame.loc[:, ["ra_deg", "dec_deg", "name", "red_shift", "red_shift_type"]]
    frame["source"] = DataSource.SPT_SZ.value
    frame["target"] = IsCluster.IS_CLUSTER.value

    frame = inherit_columns(frame)

    return frame


def read_pszspt() -> pd.DataFrame:
    """Reads the PSZSPT catalog from VizieR.

    Returns:
        pd.DataFrame: DataFrame containing PSZSPT cluster data.
    """
    CATALOGUE = "J/A+A/647/A106"
    frame = util.read_vizier(CATALOGUE)

    frame = frame.rename(
        columns={
            "Name": "name",
            "RAJ2000": "ra_deg",
            "DEJ2000": "dec_deg",
            "z": "red_shift",
        }
    )

    frame = frame[frame["red_shift"].notna()]

    frame = frame.loc[:, ["ra_deg", "dec_deg", "name", "red_shift"]]

    # данные берутся из psz и spt (предлагается ставить phot)
    frame["red_shift_type"] = "phot"

    frame["source"] = DataSource.PSZSPT.value
    frame["target"] = IsCluster.IS_CLUSTER.value

    frame = inherit_columns(frame)

    return frame


def read_comprass() -> pd.DataFrame:
    """Reads the COMPRASS catalog from VizieR.

    Returns:
        pd.DataFrame: DataFrame containing COMPRASS cluster data.
    """
    CATALOGUE = "J/A+A/626/A7/comprass"
    frame = util.read_vizier(CATALOGUE)

    frame = frame.rename(
        columns={
            "Name": "name",
            "RAJ2000": "ra_deg",
            "DEJ2000": "dec_deg",
            "z": "red_shift",
        }
    )

    frame = frame[frame["red_shift"].notna()]

    frame = frame.loc[:, ["ra_deg", "dec_deg", "name", "red_shift"]]

    # данные берутся из других каталогов (пока что предлагается ставить phot)
    frame["red_shift_type"] = "phot"

    frame["source"] = DataSource.CCOMPRASS.value
    frame["target"] = IsCluster.IS_CLUSTER.value

    frame = inherit_columns(frame)

    return frame


def read_spt2500d() -> pd.DataFrame:
    """Reads the SPT2500D catalog from VizieR.

    Returns:
        pd.DataFrame: DataFrame containing SPT2500D cluster data.
    """
    CATALOGUE = "J/ApJ/878/55/table5"
    frame = util.read_vizier(CATALOGUE)

    frame = frame.rename(
        columns={
            "SPT-CL": "name",
            "RAJ2000": "ra_deg",
            "DEJ2000": "dec_deg",
            "z": "red_shift",
        }
    )

    frame = frame[
        frame["red_shift"].notna() & frame["n_z"].str.contains(r"\+", na=False)
    ]

    frame = frame.loc[:, ["ra_deg", "dec_deg", "name", "red_shift"]]

    frame["red_shift_type"] = "spec"
    frame["source"] = DataSource.SPT2500D.value
    frame["target"] = IsCluster.IS_CLUSTER.value

    frame = inherit_columns(frame)

    return frame


def collect_sptecs(catalogue: str) -> pd.DataFrame:
    """Reads the SPTECS catalog from VizieR.

    Args:
        catalogue (str): VizieR catalog identifier.

    Returns:
        pd.DataFrame: DataFrame containing SPTECS cluster data.
    """
    frame = util.read_vizier(catalogue)

    frame = frame.rename(
        columns={
            "SPT-CL": "name",
            "RAJ2000": "ra_deg",
            "DEJ2000": "dec_deg",
            "z": "red_shift",
        }
    )

    frame = frame[frame["red_shift"].notna()]

    frame = frame.loc[:, ["ra_deg", "dec_deg", "name", "red_shift"]]

    frame["red_shift_type"] = "phot"

    frame["source"] = DataSource.SPTECS.value
    frame["target"] = IsCluster.IS_CLUSTER.value

    frame = inherit_columns(frame)

    return frame


def read_sptecs() -> pd.DataFrame:
    """Reads certified and candidate SPTECS clusters and combines them.

    Returns:
        pd.DataFrame: DataFrame containing combined SPTECS cluster data.
    """
    frame_certified = collect_sptecs("J/ApJS/247/25/table10")
    frame_candidates = collect_sptecs("J/ApJS/247/25/cand")

    frame = pd.concat([frame_certified, frame_candidates])

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


def read_act_mcmf() -> pd.DataFrame:
    """Reads the ACT_MCMF catalog from VizieR.

    Returns:
        pd.DataFrame: DataFrame containing ACT_MCMF cluster data.
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

    mc_frame = mc_frame.loc[
        :, ["ra_deg", "dec_deg", "name", "red_shift", "red_shift_type"]
    ]
    mc_frame["source"] = DataSource.ACT_MCMF.value
    mc_frame["target"] = IsCluster.IS_CLUSTER.value
    mc_frame = inherit_columns(mc_frame)

    return mc_frame
