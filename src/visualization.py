import matplotlib.pyplot as plt
import numpy as np
import cv2


def _save_figure(fig, save_path, dpi=300):
    if save_path:
        fig.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            pil_kwargs={"quality": 95},
        )
    plt.close(fig)

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


def save_uncertainty_diagnostics_plot(
    rgb_ref,
    flow_uncertainty_px,
    valid_bool,
    stable_bool_flow,
    low_uncertainty_mask,
    reliable_motion_mask,
    coarse_field,
    uncertainty_threshold_px,
    save_path,
):
    uncertainty_display = np.ma.masked_invalid(flow_uncertainty_px)
    uncertainty_vmax = np.nanpercentile(flow_uncertainty_px, 99) if np.isfinite(flow_uncertainty_px).any() else 1.0
    stable_uncertainty_overlay = np.ma.masked_where(~(stable_bool_flow & valid_bool), stable_bool_flow & valid_bool)
    reliable_only = reliable_motion_mask.astype(np.uint8)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)

    im_unc = axes[0, 0].imshow(uncertainty_display, cmap="magma", vmin=0, vmax=uncertainty_vmax)
    axes[0, 0].set_title("(a) Forward-backward uncertainty map [px]")
    axes[0, 0].axis("off")
    fig.colorbar(
        im_unc,
        ax=axes[0, 0],
        fraction=0.046,
        pad=0.08,
        orientation="horizontal",
        label="Uncertainty [px]",
    )

    axes[0, 1].imshow(uncertainty_display, cmap="magma", vmin=0, vmax=uncertainty_vmax)
    axes[0, 1].imshow(stable_uncertainty_overlay, cmap="Greens", alpha=0.28)
    axes[0, 1].set_title("(b) Stable terrain overlay on uncertainty")
    axes[0, 1].axis("off")
    fig.colorbar(
        im_unc,
        ax=axes[0, 1],
        fraction=0.046,
        pad=0.08,
        orientation="horizontal",
        label="Uncertainty [px]",
    )

    axes[1, 0].imshow(np.ma.masked_where(~low_uncertainty_mask, low_uncertainty_mask), cmap="Greys", alpha=0.8)
    axes[1, 0].imshow(np.ma.masked_where(~reliable_motion_mask, reliable_only), cmap="Reds", alpha=0.55)
    axes[1, 0].set_title("(c) Low-uncertainty mask (gray) and reliable motion (red)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(rgb_ref, alpha=0.78)
    quiver_keep = np.isfinite(coarse_field["qu"]) & np.isfinite(coarse_field["qv"])
    axes[1, 1].quiver(
        coarse_field["x"][quiver_keep],
        coarse_field["y"][quiver_keep],
        coarse_field["qu"][quiver_keep],
        coarse_field["qv"][quiver_keep],
        color="cyan",
        angles="xy",
        scale_units="xy",
        scale=0.5,
        width=0.003,
    )
    axes[1, 1].set_title("(d) Coarse quiver vectors from reliable motion only")
    axes[1, 1].axis("off")

    fig.text(
        0.5,
        0.01,
        f"Stable-area uncertainty values are used to derive the 95th-percentile threshold ({uncertainty_threshold_px:.2f} px).",
        ha="center",
        fontsize=11,
    )
    _save_figure(fig, save_path)


def save_change_and_displacement_summary_plot(
    rgb_ref,
    ref_gray,
    change_mask,
    coarse_mag_m_masked,
    low_uncertainty_mask,
    flow_uncertainty_px,
    coarse_dir_deg_masked,
    coarse_mag_m,
    coarse_valid,
    coarse_field,
    save_path,
):
    fig, axes = plt.subplots(3, 2, figsize=(18, 18), constrained_layout=True)

    axes[0, 0].imshow(rgb_ref)
    axes[0, 0].imshow(change_mask > 0, cmap="Reds", alpha=0.4)
    axes[0, 0].set_title("Binary change mask over the reference image")
    axes[0, 0].axis("off")

    im1 = axes[0, 1].imshow(coarse_mag_m_masked, cmap="viridis")
    axes[0, 1].imshow(np.ma.masked_where(~low_uncertainty_mask, flow_uncertainty_px), cmap="gray", alpha=0.08)
    axes[0, 1].set_title("Low-uncertainty coarse displacement magnitude")
    axes[0, 1].axis("off")
    fig.colorbar(
        im1,
        ax=axes[0, 1],
        fraction=0.046,
        pad=0.08,
        orientation="horizontal",
        label="Displacement [m]",
    )

    if np.any(coarse_valid):
        alpha_scale = max(np.nanpercentile(coarse_mag_m[coarse_valid], 99), 1e-9)
        alpha_map = np.clip(np.nan_to_num(coarse_mag_m / alpha_scale, nan=0.0, posinf=0.0, neginf=0.0), 0, 1)
    else:
        alpha_map = np.zeros_like(coarse_mag_m, dtype=float)
    alpha_display = 0.75 * alpha_map
    alpha_display[~coarse_valid] = 0.0

    axes[1, 0].imshow(cv2.cvtColor(ref_gray, cv2.COLOR_GRAY2RGB))
    im2 = axes[1, 0].imshow(coarse_mag_m_masked, cmap="viridis", alpha=alpha_display)
    axes[1, 0].set_title("Coarse low-uncertainty magnitude over grayscale")
    axes[1, 0].axis("off")
    fig.colorbar(
        im2,
        ax=axes[1, 0],
        fraction=0.046,
        pad=0.08,
        orientation="horizontal",
        label="Displacement [m]",
    )

    direction_cmap = plt.get_cmap("twilight_shifted").copy()
    direction_cmap.set_bad(alpha=0)
    im_dir = axes[1, 1].imshow(coarse_dir_deg_masked, cmap=direction_cmap, vmin=0, vmax=360)
    axes[1, 1].imshow(cv2.cvtColor(ref_gray, cv2.COLOR_GRAY2RGB), alpha=0.18)
    axes[1, 1].set_title("Coarse low-uncertainty movement direction")
    axes[1, 1].axis("off")
    cbar_dir = fig.colorbar(
        im_dir,
        ax=axes[1, 1],
        fraction=0.046,
        pad=0.08,
        orientation="horizontal",
        ticks=[0, 90, 180, 270, 360],
    )
    cbar_dir.set_label("Direction")
    cbar_dir.ax.set_xticklabels(["E", "N", "W", "S", "E"])

    axes[2, 0].imshow(rgb_ref, alpha=0.75)
    quiver_keep = np.isfinite(coarse_field["qu"]) & np.isfinite(coarse_field["qv"])
    axes[2, 0].quiver(
        coarse_field["x"][quiver_keep],
        coarse_field["y"][quiver_keep],
        coarse_field["qu"][quiver_keep],
        coarse_field["qv"][quiver_keep],
        color="crimson",
        angles="xy",
        scale_units="xy",
        scale=0.5,
        width=0.003,
    )
    axes[2, 0].set_title("Coarse quiver from low-uncertainty vectors")
    axes[2, 0].axis("off")

    axes[2, 1].imshow(cv2.cvtColor(ref_gray, cv2.COLOR_GRAY2RGB))
    axes[2, 1].imshow(coarse_dir_deg_masked, cmap=direction_cmap, vmin=0, vmax=360, alpha=alpha_display)
    axes[2, 1].set_title("Coarse direction overlay on grayscale")
    axes[2, 1].axis("off")
    cbar_dir_overlay = fig.colorbar(
        im_dir,
        ax=axes[2, 1],
        fraction=0.046,
        pad=0.08,
        orientation="horizontal",
        ticks=[0, 90, 180, 270, 360],
    )
    cbar_dir_overlay.set_label("Direction")
    cbar_dir_overlay.ax.set_xticklabels(["E", "N", "W", "S", "E"])

    _save_figure(fig, save_path)
