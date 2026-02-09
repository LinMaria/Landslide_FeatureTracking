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