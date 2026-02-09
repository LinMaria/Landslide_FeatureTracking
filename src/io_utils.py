import cv2
import os
import tifffile as tiff
import numpy as np
import rasterio

def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path not found: {path}")
    
    # Use rasterio to load with georeference
    with rasterio.open(path) as src:
        img = src.read()
        profile = src.profile
        
    # Rasterio reads as (bands, height, width), transpose to (height, width, bands)
    if img.shape[0] == 3:  # RGB
        img = np.transpose(img, (1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif img.shape[0] == 4:  # RGBA
        img = np.transpose(img, (1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img = img[0]  # Single band, grayscale
    
    return img, profile

def save_result(filename, image, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    path = os.path.join(output_dir, filename)
    cv2.imwrite(path, image)
    print(f"Saved: {path}")