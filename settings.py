import os


SEED = 1
WORKDIR = "./"
STORAGE_PATH = os.path.join(WORKDIR, "storage/")

data_out = os.path.join(WORKDIR, "data/")

zipped_data_out = data_out + "DATA.zip"
zipped_example_out = data_out + "example.zip"

segmentation_maps_pics = data_out + "example/"
samples_location = STORAGE_PATH
dr5_sample_location = samples_location + "test_dr5/"
madcows_sample_location = samples_location + "test_madcows/"
gaia_sample_location = samples_location + "test_gaia/"

clusters_out = segmentation_maps_pics + "cl5.csv"
randoms_out = segmentation_maps_pics + "r5.csv"
stars_out = segmentation_maps_pics + "gaia5.csv"

seg_maps = "segmentation_maps/"
bigSegMapLocation = data_out + "example_big/"
bigSegMap_pics = bigSegMapLocation + "Big/"
cl_bigSegMap_out = bigSegMapLocation + "cluster_bigSegMap.csv"
zipped_bigSegMap_out = data_out + "example_big.zip"
