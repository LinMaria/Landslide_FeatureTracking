import cv2
import numpy as np

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
    return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

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