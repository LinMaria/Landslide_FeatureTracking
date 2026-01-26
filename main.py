import os
#import cv2
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
    img1 = load_image(config.IMG1_PATH)
    img2 = load_image(config.IMG2_PATH)

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
    
    # --- 4. Lighting Correction ---
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

    # --- 6. Visualization & Save ---
    print("Visualizing results...")
    
    # Create a red overlay for the changes
    result_visual = img2_crop.copy()
    # Paint the mask area Red (BGR format: 0, 0, 255)
    result_visual[change_mask == 255] = [0, 0, 255]

    # Save to disk
    save_result("change_mask.jpg", change_mask, config.OUTPUT_DIR)
    save_result("final_overlay.jpg", result_visual, config.OUTPUT_DIR)

    # Display safely (Matplotlib)
    preview_images(
        [img1_crop, img2_corrected, change_mask, result_visual], 
        ["Date 1", "Date 2 (Aligned & Corrected)", "Change Mask", "Landslide Detection"]
    )

if __name__ == "__main__":
    main()