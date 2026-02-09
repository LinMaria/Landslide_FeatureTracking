import os
import cv2
import tifffile
import rasterio
from rasterio.transform import Affine
import numpy as np
import config
from src.geo_utils import rasterize_vector_mask
from src.io_utils import load_image, save_result
from src.alignment import align_images_constrained
from src.processing import get_common_mask, crop_to_content, match_histograms, detect_landslide_changes
from src.visualization import preview_images

def main():
    # --- 1.1 Preparation ---
    print("--- Preparation ---")

    # Check: Do we need to generate the raster mask?
    if not os.path.exists(config.RASTER_MASK_PATH):
        if os.path.exists(config.VECTOR_MASK_PATH):
            print("Vector mask found. Converting to raster...")
            rasterize_vector_mask(
                vector_path=config.VECTOR_MASK_PATH,
                reference_tif_path=config.IMG1_PATH,
                output_path=config.RASTER_MASK_PATH
            )
        else:
            print("Error: No vector file found")
            return
    
    # --- 1.2 Loading ---
    print(f"Loading images from: {config.BASE_DIR}")
    img1, profile1 = load_image(config.IMG1_PATH)
    img2, profile2 = load_image(config.IMG2_PATH)

    # Load stable mask
    if os.path.exists(config.RASTER_MASK_PATH):
        mask = cv2.imread(config.RASTER_MASK_PATH, cv2.IMREAD_GRAYSCALE)
    else:
        mask = None

    # --- 2. Alignment ---
    # This step warps img2 to match img1 perfectly based on the stable ground
    print(f"Aligning images using mask: {config.RASTER_MASK_PATH}")
    
    if os.path.exists(config.RASTER_MASK_PATH):
        # We update 'img2' to be the new, aligned version
        img2 = align_images_constrained(img1, img2, config.RASTER_MASK_PATH)
    else:
        print("WARNING: Stable mask not found at path. Skipping alignment!")
        # If you don't have a mask yet, you can fallback to the auto-alignment 
        # from the previous step, or just proceed if they are already aligned.

    # --- 3. Pre-Processing (Intersection & Crop) ---
    print("Calculating common area...")
    # Now that they are aligned, we find the overlapping pixels
    common_mask = get_common_mask(img1, img2)
    
    # Crop both images to the valid data area (removes black borders)
    img1_crop, rect = crop_to_content(img1, common_mask)
    
    # Apply exactly the same crop to the aligned img2
    x, y, w, h = rect
    img2_crop = img2[y:y+h, x:x+w]
    
    # Crop the mask to the same extent
    if mask is not None:
        mask_crop = mask[y:y+h, x:x+w]
    else:
        mask_crop = None
    
    # Save the aligned and intersected image as georeferenced TIFF
    base_name = os.path.splitext(os.path.basename(config.IMG2_PATH))[0]
    output_filename = f"{base_name}_aligned.tif"
    output_path = os.path.join(config.OUTPUT_DIR, output_filename)
    new_transform = profile2['transform'] * Affine.translation(x, y)
    profile2.update(width=w, height=h, transform=new_transform, count=3, dtype=img2_crop.dtype)
    with rasterio.open(output_path, 'w', **profile2) as dst:
        # Transpose to (bands, height, width)
        dst.write(np.transpose(img2_crop, (2, 0, 1)))
    print(f"Saved georeferenced aligned and intersected image to: {output_path}")
    
    #-- 4. Lighting Correction ---
    print("Harmonizing lighting...")
    # Match the brightness of Date 2 to Date 1
    img2_corrected = match_histograms(img2_crop, img1_crop)

    # --- 5. Detection ---
    print("Detecting changes...")
    change_mask = detect_landslide_changes(
        img1_crop, 
        img2_corrected, 
        blur_k=config.BLUR_KERNEL_SIZE,
        min_area=config.MIN_LANDSLIDE_AREA
    )

    # Plot reference image with translucent change mask
    from src.visualization import plot_image_with_change_mask
    plot_image_with_change_mask(img1_crop, change_mask, title="Reference Image with Detected Changes", save_path=os.path.join(config.OUTPUT_DIR, "change_detection_overlay.jpg"))

    # --- 5.5 Dense Displacement Field ---
    print("Computing dense displacement field...")
    from src.processing import compute_dense_displacement, visualize_displacement_field
    flow = compute_dense_displacement(img1_crop, img2_corrected, mask_crop)
    visualize_displacement_field(img1_crop, flow, save_path=os.path.join(config.OUTPUT_DIR, "displacement_field.jpg"))

    # --- 6. Visualization & Save ---
    # print("Visualizing results...")
    
    # # Create a red overlay for the changes
    # result_visual = img2_crop.copy()
    # # Paint the mask area Red (BGR format: 0, 0, 255)
    # result_visual[change_mask == 255] = [0, 0, 255]

    # # Save to disk
    # save_result("change_mask.jpg", change_mask, config.OUTPUT_DIR)
    # save_result("final_overlay.jpg", result_visual, config.OUTPUT_DIR)

    # # Display safely (Matplotlib)
    # preview_images(
    #     [img1_crop, img2_corrected, change_mask, result_visual], 
    #     ["Date 1", "Date 2 (Aligned & Corrected)", "Change Mask", "Landslide Detection"]
    # )

if __name__ == "__main__":
    main()