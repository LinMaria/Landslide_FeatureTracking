import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import config
from src.visualization import plot_image_with_mask

def align_images_constrained(img_ref, img_to_align, mask=None):
    """
    Aligns images using SIFT, but ONLY calculates features 
    inside the 'stable_mask' area.
    """
    print("Loading stable area mask...")
    # Load mask as grayscale (0 = Ignore, 255 = Include)
    if isinstance(mask, str):
        mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
    # Else assume it's already the array
    
    if mask is None:
        print("Error: Could not load mask.")
        return img_to_align
        
    # Ensure mask is same size as reference image
    ## Is this really necessary or even correct?
    if mask.shape != img_ref.shape[:2]:
        print("Resizing mask to match image dimensions...")
        mask = cv2.resize(mask, (img_ref.shape[1], img_ref.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        print("No resizing needed")

    # Plot the reference image with the mask overlay
    plot_image_with_mask(img_ref, mask, title="Reference Image with Stable Mask Overlay", save_path=os.path.join(config.OUTPUT_DIR, "reference_with_mask_overlay.png"))

    print("Detecting features in STABLE areas only...")
    
    # 1. Convert images to grayscale
    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    gray_align = cv2.cvtColor(img_to_align, cv2.COLOR_BGR2GRAY)
    
    # 2. Initialize SIFT
    sift = cv2.SIFT_create()
    
    # 3. Detect Keypoints
    # CRITICAL CHANGE: We pass the 'mask' to the reference detection.
    # OpenCV will only look for points where mask != 0.
    kp1, des1 = sift.detectAndCompute(gray_ref, mask=mask)
    print(f"Key points detected in image 1 {len(kp1)}")
    
    # We search the whole second image for matches corresponding to those stable points
    kp2, des2 = sift.detectAndCompute(gray_align, None)
    print(f"Key points detected in image 2 {len(kp2)}")
    
    # 4. Match features (FLANN)
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    print(f" Found {len(matches)} total matches")
    
    # 5. Filter matches (Lowe's Ratio Test)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
            
    print(f"Found {len(good_matches)} matches in stable terrain.")

    idx1 = sorted({m.queryIdx for m in good_matches})
    kp1_used = [kp1[i] for i in idx1]
    
    # Plot reference image with keypoints
    img_with_keypoints = cv2.drawKeypoints(gray_ref, kp1_used, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(img_with_keypoints, cmap='gray')
    plt.title("Keypoints on Grayscale Reference Image")
    plt.axis('off')
    plt.savefig(os.path.join(config.OUTPUT_DIR, "keypoints_on_grayscale.png"))
    plt.show()
    
    if len(good_matches) < 10:
        print("Error: Not enough stable matches found!")
        return img_to_align
        
    # 6. Extract points and find Homography
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # We still use RANSAC to handle any small errors in your manual masking
    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    print("Computed Homography Matrix with RANSAC.")
    
    # 7. Warp
    h, w = img_ref.shape[:2]
    aligned_img = cv2.warpPerspective(img_to_align, H, (w, h))
    
    return aligned_img