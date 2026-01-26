import os

# Base Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input Paths
IMG1_PATH = os.path.join(BASE_DIR, "data/input/2023-02-19_clip.tif")
IMG2_PATH = os.path.join(BASE_DIR, "data/input/2023-03-21_clip.tif")

VECTOR_MASK_PATH = os.path.join(BASE_DIR, "data/stable_masks/stable_areas.shp")
RASTER_MASK_PATH = os.path.join(BASE_DIR, "data/stable_masks/stable_mask_raster.tif")

# Output Path
OUTPUT_DIR = os.path.join(BASE_DIR, "data/output")

# Parameters
MIN_LANDSLIDE_AREA = 500  # pixels
BLUR_KERNEL_SIZE = (9, 9)