import cv2
import os
import tifffile as tiff
import numpy as np

def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path not found: {path}")
    
    # Try tifffile first (better for drone orthos)
    try:
        img = tiff.imread(path)
        # Convert RGB/RGBA to BGR for OpenCV
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        return img
    except:
        # Fallback to OpenCV standard
        return cv2.imread(path)

def save_result(filename, image, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    path = os.path.join(output_dir, filename)
    cv2.imwrite(path, image)
    print(f"Saved: {path}")