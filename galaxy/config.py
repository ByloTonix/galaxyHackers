import os
from pathlib import Path

from dynaconf import Dynaconf


settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=["settings.toml", ".secrets.toml"],
)

settings.STORAGE_PATH = Path(settings.WORKDIR, "storage/")
settings.METRICS_PATH = Path(settings.STORAGE_PATH, "metrics/")
settings.DATA_PATH = Path(settings.STORAGE_PATH, "data/")
settings.DESCRIPTION_PATH = Path(settings.DATA_PATH, "description/")
settings.BEST_MODELS_PATH = Path(settings.STORAGE_PATH, "best_models/")
settings.PREDICTIONS_PATH = Path(settings.STORAGE_PATH, "predictions/")

settings.MAP_ACT_PATH = Path(settings.DATA_PATH, settings.MAP_ACT_FILENAME)
settings.DR5_CLUSTERS_PATH = Path(settings.DATA_PATH, settings.DR5_CLUSTERS_FILENAME)
settings.SGA_PATH =  Path(settings.DATA_PATH, settings.SGA_FILENAME)

settings.SEGMENTATION_PATH = Path(settings.STORAGE_PATH, "segmentation/")
settings.SEGMENTATION_SAMPLES_PATH = Path(settings.SEGMENTATION_PATH, "samples/")
settings.SEGMENTATION_SAMPLES_DESCRIPTION_PATH = Path(
    settings.SEGMENTATION_SAMPLES_PATH, "description/"
)
settings.SEGMENTATION_MAPS_PATH = Path(settings.SEGMENTATION_PATH, "maps/")


# TODO: fix FALLBACK_URL in MAP_ACT_CONFIG and DR5_CONFIG
settings.MAP_ACT_CONFIG = {
    "RENAME_DICT": {
        "SOURCE": Path(settings.DATA_PATH, settings.MAP_ACT_ROUTE),
        "TARGET": settings.MAP_ACT_PATH,
    },
    "URL": "https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr5/maps/act_planck_dr5.01_s08s18_AA_f220_daynight_fullivar.fits",
    "ZIPPED_OUTPUT_PATH": Path(settings.DATA_PATH, "ACT.zip"),
    "FALLBACK_URL": "https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr5/maps/act_planck_dr5.01_s08s18_AA_f220_daynight_fullivar.fits",
    "OUTPUT_PATH": str(settings.MAP_ACT_PATH),
}

settings.DR5_CONFIG = {
    "RENAME_DICT": {
        "SOURCE": Path(settings.DATA_PATH, settings.DR5_CLUSTERS_ROUTE),
        "TARGET": settings.DR5_CLUSTERS_PATH,
    },
    "URL": "https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr5/DR5_cluster-catalog_v1.1.fits",
    "ZIPPED_OUTPUT_PATH": Path(settings.DATA_PATH, "dr5.zip"),
    "FALLBACK_URL": "https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr5/DR5_cluster-catalog_v1.1.fits",
    "OUTPUT_PATH": str(settings.DR5_CLUSTERS_PATH),
}

settings.SGA_CONFIG = {
     "RENAME_DICT" : {
         "SOURCE": Path(settings.DATA_PATH, settings.SGA_ROUTE), 
         "TARGET":  settings.SGA_PATH 
         },
    "URL": "https://portal.nersc.gov/project/cosmo/data/sga/2020/SGA-2020.fits",
    "ZIPPED_OUTPUT_PATH": None,
    "FALLBACK_URL" : 'https://portal.nersc.gov/project/cosmo/data/sga/2020/SGA-2020.fits',
    "OUTPUT_PATH": str(settings.SGA_PATH) 
}


required_paths = [
    settings.STORAGE_PATH,
    settings.METRICS_PATH,
    settings.DATA_PATH,
    settings.DESCRIPTION_PATH,
    settings.BEST_MODELS_PATH,
    settings.SEGMENTATION_PATH,
    settings.SEGMENTATION_SAMPLES_PATH,
    settings.SEGMENTATION_SAMPLES_DESCRIPTION_PATH,
    settings.SEGMENTATION_MAPS_PATH,
    settings.PREDICTIONS_PATH,
]


for path in required_paths:
    os.makedirs(path, exist_ok=True)


# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load these files in the order.
