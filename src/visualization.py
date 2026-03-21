import matplotlib.pyplot as plt
import numpy as np
import cv2

def preview_images(img_list, titles):
    """
    Displays images using Matplotlib.
    Safe for Ubuntu/Jupyter/VS Code.
    """
    # Create a figure
    plt.figure(figsize=(20, 10))
    
    for i, (img, title) in enumerate(zip(img_list, titles)):
        plt.subplot(1, len(img_list), i+1)
        
        # Check if the image is color (3 channels) or grayscale (2 dimensions)
        if len(img.shape) == 3:
            # OpenCV loads as Blue-Green-Red (BGR). 
            # Matplotlib expects Red-Green-Blue (RGB). We must convert.
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(rgb_img)
        else:
            # Grayscale image
            plt.imshow(img, cmap='gray')
            
        plt.title(title)
        plt.axis('off') # Hide axis numbers
        
    plt.show()

def plot_image_with_mask(image, mask, title="Image with Mask Overlay", save_path=None, opacity=0.5, color=(0, 0, 255)):
    """
    Overlays a translucent mask on the image where mask == 255.
    Useful for visualizing masks on reference images during processing stages.
    If save_path is provided, saves the plot to that path.
    opacity: transparency level (0-1)
    color: BGR color for the overlay
    """
    if image.shape[:2] != mask.shape:
        raise ValueError("Image and mask dimensions do not match.")
    
    # Create a copy of the image
    overlay = image.copy()
    
    # Create overlay for mask
    mask_overlay = np.zeros_like(image)
    mask_overlay[mask == 255] = color  # Color in BGR
    
    # Blend with specified opacity
    overlay = cv2.addWeighted(overlay, 1.0, mask_overlay, opacity, 0)
    
    # Convert to RGB for Matplotlib
    if len(overlay.shape) == 3:
        rgb_overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_overlay)
    else:
        plt.imshow(overlay, cmap='gray')
    
    plt.title(title)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_image_with_change_mask(image, change_mask, title="Reference Image with Change Mask Overlay", save_path=None):
    """
    Overlays a translucent red mask on the image where changes are detected (change_mask == 255).
    """
    plot_image_with_mask(image, change_mask, title, save_path, opacity=0.5, color=(0, 0, 255))

def visualize_displacement_field(img1, flow, title="Displacement Field (Color-Coded)", save_path=None):
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
    plt.title(title)
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