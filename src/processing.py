import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import config

def compute_dense_displacement(img1, img2, mask=None):
    """
    Computes dense optical flow (displacement field) between two images using Farneback method.
    If mask is provided, sets flow to 0 in stable areas (mask == 255).
    Returns the flow (dx, dy) as a 3D array (h, w, 2).
    """
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Compute optical flow
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None, 
        pyr_scale=0.5, levels=3, winsize=15, iterations=3, 
        poly_n=5, poly_sigma=1.2, flags=0
    )
    
    # If mask provided, zero out flow in stable areas
    if mask is not None:
        # Resize mask to match flow if needed
        if mask.shape != flow.shape[:2]:
            mask = cv2.resize(mask, (flow.shape[1], flow.shape[0]), interpolation=cv2.INTER_NEAREST)
        # Set flow to 0 where mask is 255 (stable)
        flow[mask == 255] = 0
    
    return flow

def visualize_displacement_field(img1, flow, save_path=None):
    """
    Visualizes the displacement field as a color-coded flow image and quiver plot.
    Saves to save_path if provided.
    """
    h, w = flow.shape[:2]
    
    # Create HSV flow visualization
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2  # Hue: direction
    hsv[..., 1] = 255  # Saturation: full
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value: magnitude
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Overlay on original image
    overlay = cv2.addWeighted(img1, 0.5, flow_rgb, 0.5, 0)
    
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(flow_rgb, cv2.COLOR_BGR2RGB))
    plt.title("Displacement Field (Color-Coded)")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Overlay on Reference Image")
    plt.axis('off')
    
    # Quiver plot for displacement vectors (subsampled)
    plt.subplot(1, 3, 3)
    step = 20  # Subsample every 20 pixels
    y, x = np.mgrid[step//2:h:step, step//2:w:step]
    u = flow[y, x, 0]
    v = flow[y, x, 1]
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), alpha=0.5)
    plt.quiver(x, y, u, v, color='red', scale=1, scale_units='xy', angles='xy')
    plt.title("Displacement Vectors (Quiver Plot)")
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def get_common_mask(img1, img2):
    """
    Creates a mask where BOTH images have valid data.
    Assumes (0,0,0) or Alpha=0 is 'No Data'.
    """
    # Helper to find valid pixels
    def get_valid_pixels(img):
        if img.shape[2] == 4: # RGBA
            return img[:, :, 3] > 0
        else: # BGR
            return np.any(img > 0, axis=-1)

    mask1 = get_valid_pixels(img1)
    mask2 = get_valid_pixels(img2)
    
    # Logical AND to find intersection
    common_mask = np.logical_and(mask1, mask2)
    return common_mask.astype(np.uint8) * 255

def crop_to_content(img, mask):
    """
    Crops an image (and its mask) to the bounding box of the valid data.
    Reduces file size and processing time.
    """
    points = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(points)
    
    cropped_img = img[y:y+h, x:x+w]
    return cropped_img, (x, y, w, h)

def match_histograms(source, reference):
    """
    Adjusts the lighting of 'source' to match 'reference'.
    Crucial for comparing sunny vs cloudy days.
    """
    # Convert to LAB color space (L = Lightness)
    src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)
    
    src_l, src_a, src_b = cv2.split(src_lab)
    ref_l, ref_a, ref_b = cv2.split(ref_lab)
    
    # Calculate CDF (Cumulative Distribution Function)
    src_hist, _ = np.histogram(src_l.flatten(), 256, [0, 256])
    ref_hist, _ = np.histogram(ref_l.flatten(), 256, [0, 256])
    
    src_cdf = src_hist.cumsum()
    ref_cdf = ref_hist.cumsum()
    
    # Normalize
    src_cdf_norm = src_cdf / src_cdf.max()
    ref_cdf_norm = ref_cdf / ref_cdf.max()
    
    # Create Lookup Table
    lookup_table = np.zeros(256, dtype=np.uint8)
    g_j = 0
    for g_i in range(256):
        while g_j < 255 and ref_cdf_norm[g_j] < src_cdf_norm[g_i]:
            g_j += 1
        lookup_table[g_i] = g_j
        
    # Apply mapping to Lightness channel only
    src_l_matched = cv2.LUT(src_l, lookup_table)
    
    # Merge back
    merged_lab = cv2.merge((src_l_matched, src_a, src_b))
    matched_source = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    
    # Plot both images one on top of the other
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.imshow(cv2.cvtColor(reference, cv2.COLOR_BGR2RGB))
    plt.title("Reference Image")
    plt.axis('off')
    plt.subplot(2, 1, 2)
    plt.imshow(cv2.cvtColor(matched_source, cv2.COLOR_BGR2RGB))
    plt.title("Source Image (Histogram Matched)")
    plt.axis('off')
    plt.tight_layout()
    
    # Save the plot before showing
    plt.savefig(os.path.join(config.OUTPUT_DIR, "histogram_matching.jpg"))
    plt.show()
    
    return matched_source

def detect_landslide_changes(img1, img2, blur_k=(9,9), threshold=30, min_area=500):
    """
    The Core Pipeline:
    1. Grayscale -> 2. Blur -> 3. AbsDiff -> 4. Threshold -> 5. Morphology
    Returns: A clean binary mask of changes.
    """
    # 1. Convert to Gray
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # 2. Blur (Remove grass texture noise)
    blur1 = cv2.GaussianBlur(gray1, blur_k, 0)
    blur2 = cv2.GaussianBlur(gray2, blur_k, 0)
    
    # 3. Absolute Difference
    diff = cv2.absdiff(blur1, blur2)
    
    # 4. Binary Threshold
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # 5. Morphology (Clean up noise)
    # Open: Removes small dots (leaves/noise)
    # Close: Connects gaps inside the landslide
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    clean_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # 6. Area Filtering (Optional but recommended)
    # Remove blobs smaller than min_area
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(clean_mask)
    
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            cv2.drawContours(final_mask, [cnt], -1, 255, -1)
            
    return final_mask