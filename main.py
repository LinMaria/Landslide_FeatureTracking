import os
import cv2
from matplotlib import pyplot as plt
import rasterio
from rasterio.transform import Affine
import numpy as np
import re
import config
from src.geo_utils import rasterize_vector_mask
from src.io_utils import load_image, save_result
from src.alignment import align_images_constrained
from src.processing import (
    get_common_mask, crop_to_content, match_histograms, 
    detect_landslide_changes, compute_dense_displacement,
    process_image_pair
)
from src.visualization import plot_image_with_mask, plot_image_with_change_mask, preview_images, visualize_displacement_field

def extract_date(path):
    """Extract date from filename in format YYYY-MM-DD"""
    fname = os.path.basename(path)
    match = re.search(r'(\d{4}-\d{2}-\d{2})', fname)
    return match.group(1) if match else fname


def main():
    # --- 1.1 Preparation ---
    print("--- Preparation ---")

    # Check: Do we need to generate the raster mask?
    if not os.path.exists(config.RASTER_MASK_PATH):
        if os.path.exists(config.VECTOR_MASK_PATH):
            print("Vector mask found. Converting to raster...")
            # Get sorted image paths
            image_paths = config.get_sorted_image_paths()
            if not image_paths:
                print("Error: No images found")
                return
            ref_img_path = image_paths[0]
            rasterize_vector_mask(
                vector_path=config.VECTOR_MASK_PATH,
                reference_tif_path=ref_img_path,
                output_path=config.RASTER_MASK_PATH
            )
        else:
            print("Error: No vector file found")
            return
    
    # --- 1.2 Loading Images (sorted by date) ---
    print(f"\nLoading images from folder: {config.INPUT_FOLDER}")
    image_paths = config.get_sorted_image_paths()
    
    if len(image_paths) < 2:
        print("Error: Need at least 2 images to process")
        return

    # Load stable mask
    if os.path.exists(config.RASTER_MASK_PATH):
        mask = cv2.imread(config.RASTER_MASK_PATH, cv2.IMREAD_GRAYSCALE)
    else:
        mask = None

    # Load all images and profiles
    print(f"\nLoading {len(image_paths)} images (sorted by date)...")
    images = []
    profiles = []
    for i, img_path in enumerate(image_paths):
        img, profile = load_image(img_path)
        images.append(img)
        profiles.append(profile)
        img_name = os.path.basename(img_path)
        ref_marker = " (REFERENCE - Oldest)" if i == 0 else ""
        print(f"  [{i}] {img_name}{ref_marker}")

    # --- 2. Alignment (Each to Previous) ---
    print(f"\nAligning {len(images)} images to previous image using mask...")
    aligned_images = [images[0]]
    print(f"  Image 0: {extract_date(image_paths[0])} (Reference - no alignment needed)")
    for i in range(1, len(images)):
        prev_img = aligned_images[i-1]
        curr_img = images[i]
        prev_date = extract_date(image_paths[i-1])
        curr_date = extract_date(image_paths[i])
        print(f"  Aligning image {i}: {curr_date} to previous: {prev_date} ...")
        img_aligned, kp1_used, gray_ref = align_images_constrained(prev_img, curr_img, config.RASTER_MASK_PATH)
        aligned_images.append(img_aligned)
        if i == 1 and mask is not None and kp1_used is not None and gray_ref is not None:
            plot_image_with_mask(prev_img, mask, title=f"Reference Image {prev_date} with Stable Mask Overlay", save_path=os.path.join(config.PLOTS_DIR, "reference_with_mask_overlay.png"))
            img_with_keypoints = cv2.drawKeypoints(gray_ref, kp1_used, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            plt.imshow(img_with_keypoints, cmap='gray')
            plt.title(f"Keypoints on Grayscale Reference Image {prev_date}")
            plt.axis('off')
            plt.savefig(os.path.join(config.PLOTS_DIR, "keypoints_on_grayscale.png"))
            plt.show()

    # --- 3. Save aligned images as georeferenced TIFFs ---
    print("\nSaving aligned images...")
    for i, (img_aligned, profile, img_path) in enumerate(zip(aligned_images, profiles, image_paths)):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_filename = f"{base_name}_aligned.tif"
        output_path = os.path.join(config.ALIGNED_IMAGES_DIR, output_filename)
        profile_copy = profile.copy()
        profile_copy.update(width=img_aligned.shape[1], height=img_aligned.shape[0], count=3, dtype=img_aligned.dtype)
        with rasterio.open(output_path, 'w', **profile_copy) as dst:
            dst.write(np.transpose(img_aligned, (2, 0, 1)))
        print(f"Saved aligned image {i}: {output_path}")

    # --- 4-5. Process consecutive image pairs ---
    print("\nProcessing consecutive image pairs...")
    for i in range(1, len(aligned_images)):
        prev_img = aligned_images[i-1]
        curr_img = aligned_images[i]
        prev_profile = profiles[i-1]
        curr_profile = profiles[i]
        prev_date = extract_date(image_paths[i-1])
        curr_date = extract_date(image_paths[i])
        print(f"\n--- Processing pair: {prev_date} vs {curr_date} ---")
        process_image_pair(
            prev_img,
            prev_profile,
            curr_img,
            curr_profile,
            mask,
            i,
            config.PLOTS_DIR,
            prev_date=prev_date,
            curr_date=curr_date
        )

if __name__ == "__main__":
    main()