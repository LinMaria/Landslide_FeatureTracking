import matplotlib.pyplot as plt
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