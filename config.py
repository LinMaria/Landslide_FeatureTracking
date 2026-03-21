import os
import glob
import re

# Base Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input folder and reference image
INPUT_FOLDER = os.path.join(BASE_DIR, "data/input")

# Image file extensions to look for
IMAGE_EXTENSIONS = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]

VECTOR_MASK_PATH = os.path.join(BASE_DIR, "data/stable_masks/stable_area_c.shp")
RASTER_MASK_PATH = os.path.join(BASE_DIR, "data/stable_masks/stable_mask_raster.tif")

# Output Paths
OUTPUT_DIR = os.path.join(BASE_DIR, "data/output")
ALIGNED_IMAGES_DIR = os.path.join(OUTPUT_DIR, "aligned")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "img")

# Create output directories if they don't exist
os.makedirs(ALIGNED_IMAGES_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Parameters
MIN_LANDSLIDE_AREA = 500  # pixels
BLUR_KERNEL_SIZE = (9, 9)
CHANGE_THRESHOLD = 30  # Difference threshold for change detection

# Function to get sorted image paths by date
def get_sorted_image_paths():
    """
    Scans INPUT_FOLDER for images and returns them sorted by date in filename.
    Date format expected: YYYY-MM-DD (e.g., clip_2023-01-17.tif)
    The first image (oldest date) will be used as reference for alignment.
    """
    all_images = []
    for ext in IMAGE_EXTENSIONS:
        pattern = os.path.join(INPUT_FOLDER, ext)
        all_images.extend(glob.glob(pattern))
    
    if not all_images:
        return []
    
    # Extract date from filename and sort
    def extract_date(path):
        fname = os.path.basename(path)
        match = re.search(r'(\d{4}-\d{2}-\d{2})', fname)
        return match.group(1) if match else ''
    
    all_images.sort(key=extract_date)
    return all_images