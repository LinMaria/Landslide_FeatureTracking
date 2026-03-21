import json
from pathlib import Path
import textwrap


def md_cell(text: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [textwrap.dedent(text).strip() + "\n"],
    }


def code_cell(text: str):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [textwrap.dedent(text).strip() + "\n"],
    }


nb = {
    "cells": [
    md_cell(
        """
        # Landslide Monitoring Notebook

        This notebook is organized in two parts:

        1. **Method walkthrough**: how the images are finely aligned on stable terrain, how radiometric differences are reduced, and how change masks and dense displacement fields are computed.
        2. **Result analysis**: how the detected changes and dense-field products behave in space and through time, including empirical uncertainty estimates derived from the stable terrain.

        The uncertainty values reported here are **measurement-quality indicators**, not formal probabilistic confidence intervals. They are estimated from places that should remain stable and therefore provide a practical reference for residual misalignment and false detections.
        """
    ),
    code_cell(
        """
        import os
        import re
        import sys
        from datetime import datetime
        from pathlib import Path

        import cv2
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import rasterio
        from IPython.display import display

        PROJECT_ROOT = Path("..").resolve()
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))

        import config
        from src.alignment import align_images_constrained
        from src.io_utils import load_image
        from src.processing import (
            compute_dense_displacement,
            detect_landslide_changes,
            get_common_mask,
            match_histograms,
        )

        %matplotlib inline
        plt.style.use("seaborn-v0_8-whitegrid")
        """
    ),
    code_cell(
        """
        EXAMPLE_PREV_DATE = "2023-02-03"
        EXAMPLE_CURR_DATE = "2023-02-06"
        CHANGE_THRESHOLD = config.CHANGE_THRESHOLD
        PIXEL_TO_METER_OVERRIDE = None


        def extract_date(path):
            match = re.search(r"(\\d{4}-\\d{2}-\\d{2})", Path(path).name)
            if not match:
                raise ValueError(f"Could not extract a date from {path}")
            return match.group(1)


        def parse_date(date_str):
            return datetime.strptime(date_str, "%Y-%m-%d")


        def image_path_for_date(date_str, image_paths):
            for path in image_paths:
                if date_str in Path(path).name:
                    return path
            raise FileNotFoundError(f"No input image found for {date_str}")


        def aligned_path_for_date(date_str):
            path = Path(config.ALIGNED_IMAGES_DIR) / f"clip_{date_str}_aligned.tif"
            return path if path.exists() else None


        def pair_output_paths(prev_date, curr_date):
            base = Path(config.OUTPUT_DIR) / "results"
            return (
                base / f"change_mask_{prev_date}_to_{curr_date}.npy",
                base / f"dense_field_{prev_date}_to_{curr_date}.npy",
            )


        image_paths = config.get_sorted_image_paths()
        if len(image_paths) < 2:
            raise RuntimeError("At least two dated input images are required.")

        with rasterio.open(image_paths[0]) as src:
            pixel_size_x, pixel_size_y = src.res
            pixel_size_m = PIXEL_TO_METER_OVERRIDE or float(np.mean(np.abs(src.res)))
            pixel_area_m2 = float(abs(src.transform.a * src.transform.e))

        stable_mask = cv2.imread(config.RASTER_MASK_PATH, cv2.IMREAD_GRAYSCALE)
        if stable_mask is None:
            raise FileNotFoundError(f"Stable raster mask not found: {config.RASTER_MASK_PATH}")

        example_prev_path = image_path_for_date(EXAMPLE_PREV_DATE, image_paths)
        example_curr_path = image_path_for_date(EXAMPLE_CURR_DATE, image_paths)

        print(f"Loaded {len(image_paths)} input images")
        print(f"Example pair: {EXAMPLE_PREV_DATE} -> {EXAMPLE_CURR_DATE}")
        print(f"Pixel size: {pixel_size_m:.4f} m")
        print(f"Pixel area: {pixel_area_m2:.4f} m²")
        """
    ),
    md_cell(
        """
        ## 1. Fine Alignment of the Images

        The alignment step uses SIFT features restricted to the **stable terrain mask**. That is important because landslide-affected terrain should not drive the geometric transformation.

        The practical sequence is:

        1. detect keypoints only in stable areas of the reference image,
        2. match them to the target image,
        3. robustly estimate a homography with RANSAC,
        4. warp the target image onto the reference geometry.

        Below, the residual difference over the stable terrain is compared before and after alignment. A successful fine alignment should reduce those residuals.
        """
    ),
    code_cell(
        """
        img_ref_raw, ref_profile = load_image(example_prev_path)
        img_curr_raw, curr_profile = load_image(example_curr_path)

        img_curr_aligned, kp_ref_used, gray_ref = align_images_constrained(
            img_ref_raw,
            img_curr_raw,
            config.RASTER_MASK_PATH,
        )

        stable_bool = stable_mask == 255
        ref_gray = cv2.cvtColor(img_ref_raw, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(img_curr_raw, cv2.COLOR_BGR2GRAY)
        aligned_gray = cv2.cvtColor(img_curr_aligned, cv2.COLOR_BGR2GRAY)

        common_h = min(ref_gray.shape[0], curr_gray.shape[0])
        common_w = min(ref_gray.shape[1], curr_gray.shape[1])
        ref_gray_before = ref_gray[:common_h, :common_w]
        curr_gray_before = curr_gray[:common_h, :common_w]
        stable_bool_before = stable_bool[:common_h, :common_w]

        diff_before = cv2.absdiff(ref_gray_before, curr_gray_before)
        diff_after = cv2.absdiff(ref_gray, aligned_gray)

        before_stable_mae = float(diff_before[stable_bool_before].mean())
        after_stable_mae = float(diff_after[stable_bool].mean())
        improvement = before_stable_mae - after_stable_mae

        print(f"Stable-terrain MAE before alignment: {before_stable_mae:.2f}")
        print(f"Stable-terrain MAE after alignment:  {after_stable_mae:.2f}")
        print(f"Residual improvement:                {improvement:.2f}")
        """
    ),
    code_cell(
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)

        axes[0, 0].imshow(cv2.cvtColor(img_ref_raw, cv2.COLOR_BGR2RGB))
        axes[0, 0].imshow(stable_bool, cmap="Greens", alpha=0.22)
        axes[0, 0].set_title(f"Reference image ({EXAMPLE_PREV_DATE}) and stable mask")
        axes[0, 0].axis("off")

        keypoints_preview = cv2.drawKeypoints(
            gray_ref,
            kp_ref_used,
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        axes[0, 1].imshow(keypoints_preview, cmap="gray")
        axes[0, 1].set_title("Stable-terrain keypoints used for alignment")
        axes[0, 1].axis("off")

        im0 = axes[1, 0].imshow(diff_before, cmap="magma", vmin=0, vmax=np.percentile(diff_before, 99))
        axes[1, 0].imshow(stable_bool_before, cmap="Greens", alpha=0.12)
        axes[1, 0].set_title("Absolute difference before alignment (shared raw extent)")
        axes[1, 0].axis("off")

        im1 = axes[1, 1].imshow(diff_after, cmap="magma", vmin=0, vmax=np.percentile(diff_before, 99))
        axes[1, 1].imshow(stable_bool, cmap="Greens", alpha=0.12)
        axes[1, 1].set_title("Absolute difference after alignment")
        axes[1, 1].axis("off")

        fig.colorbar(im1, ax=axes[1, :], shrink=0.75, label="Gray-level residual")
        plt.show()
        """
    ),
    md_cell(
        """
        ## 2. Calculating the Change Mask and the Dense Field

        After geometric alignment, the notebook computes the analysis products in three steps:

        1. **Common valid extent**: restrict the analysis to overlapping pixels.
        2. **Radiometric harmonization**: match the target-image histogram to the reference image so illumination differences contribute less to false change.
        3. **Product generation**:
           - a binary **change mask** from filtered image differences,
           - a dense **optical-flow displacement field** from Farneback flow.

        In the dense field, the magnitude indicates how much the surface appears to move between the two dates, while the vector direction indicates the motion direction.
        """
    ),
    code_cell(
        """
        common_mask = get_common_mask(img_ref_raw, img_curr_aligned)
        valid_bool = common_mask > 0

        img_curr_corrected = match_histograms(img_curr_aligned, img_ref_raw)

        change_mask_raw = detect_landslide_changes(
            img_ref_raw,
            img_curr_corrected,
            blur_k=config.BLUR_KERNEL_SIZE,
            threshold=CHANGE_THRESHOLD,
            min_area=config.MIN_LANDSLIDE_AREA,
        )
        change_mask = change_mask_raw.copy()
        change_mask[stable_bool] = 0
        change_mask[~valid_bool] = 0

        flow_raw = compute_dense_displacement(img_ref_raw, img_curr_corrected, mask=None)
        flow = flow_raw.copy()
        flow[stable_bool] = 0
        flow[~valid_bool] = 0
        flow_magnitude_px = np.linalg.norm(flow, axis=-1)
        flow_magnitude_m = flow_magnitude_px * pixel_size_m
        flow_direction_deg = (np.degrees(np.arctan2(flow[..., 1], flow[..., 0])) + 360) % 360
        direction_valid = valid_bool & (flow_magnitude_px > 0.25)

        change_area_m2 = float((change_mask > 0).sum() * pixel_area_m2)
        print(f"Detected change area: {change_area_m2:,.1f} m²")
        print(f"Maximum displacement: {flow_magnitude_m.max():.2f} m")
        """
    ),
    code_cell(
        """
        rgb_ref = cv2.cvtColor(img_ref_raw, cv2.COLOR_BGR2RGB)
        step = 50
        y, x = np.mgrid[step // 2 : flow.shape[0] : step, step // 2 : flow.shape[1] : step]
        u = flow[y, x, 0]
        v = flow[y, x, 1]
        keep = np.linalg.norm(np.stack([u, v], axis=-1), axis=-1) > 0.5

        fig, axes = plt.subplots(3, 2, figsize=(18, 18), constrained_layout=True)

        axes[0, 0].imshow(rgb_ref)
        axes[0, 0].imshow(change_mask > 0, cmap="Reds", alpha=0.4)
        axes[0, 0].set_title("Binary change mask over the reference image")
        axes[0, 0].axis("off")

        im1 = axes[0, 1].imshow(flow_magnitude_m, cmap="viridis")
        axes[0, 1].set_title("Dense-field displacement magnitude")
        axes[0, 1].axis("off")
        fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04, label="Displacement [m]")

        direction_cmap = plt.get_cmap("twilight_shifted").copy()
        direction_cmap.set_bad(alpha=0)
        direction_display = np.ma.masked_where(~direction_valid, flow_direction_deg)
        alpha_map = np.clip(flow_magnitude_px / np.percentile(flow_magnitude_px[valid_bool], 99), 0, 1)

        axes[1, 0].imshow(cv2.cvtColor(ref_gray, cv2.COLOR_GRAY2RGB))
        im2 = axes[1, 0].imshow(flow_magnitude_m, cmap="viridis", alpha=0.75 * alpha_map)
        axes[1, 0].set_title("Dense-field magnitude over grayscale context")
        axes[1, 0].axis("off")
        fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04, label="Displacement [m]")

        im_dir = axes[1, 1].imshow(direction_display, cmap=direction_cmap, vmin=0, vmax=360)
        axes[1, 1].imshow(cv2.cvtColor(ref_gray, cv2.COLOR_GRAY2RGB), alpha=0.18)
        axes[1, 1].set_title("Movement direction (cyclic colormap)")
        axes[1, 1].axis("off")
        cbar_dir = fig.colorbar(
            im_dir,
            ax=axes[1, 1],
            fraction=0.046,
            pad=0.04,
            ticks=[0, 90, 180, 270, 360],
        )
        cbar_dir.set_label("Direction [degrees]")

        axes[2, 0].imshow(rgb_ref, alpha=0.75)
        axes[2, 0].quiver(
            x[keep],
            y[keep],
            u[keep],
            v[keep],
            color="crimson",
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.002,
        )
        axes[2, 0].set_title("Subsampled dense-field vectors")
        axes[2, 0].axis("off")

        axes[2, 1].imshow(cv2.cvtColor(ref_gray, cv2.COLOR_GRAY2RGB))
        axes[2, 1].imshow(direction_display, cmap=direction_cmap, vmin=0, vmax=360, alpha=0.75 * alpha_map)
        axes[2, 1].set_title("Direction overlay on grayscale context")
        axes[2, 1].axis("off")

        plt.show()
        """
    ),
    md_cell(
        """
        ## 3. Pairwise Uncertainty Estimation

        The notebook estimates uncertainty empirically from the **stable terrain**, where true motion should be negligible.

        Two practical indicators are used:

        - **Change-mask uncertainty**: area falsely flagged as change inside the stable mask.
        - **Dense-field uncertainty**: residual motion measured inside the stable mask, summarized with RMSE and the 95th percentile.

        These indicators are especially useful for judging whether a weak signal is likely meaningful or comparable to the background processing noise.
        """
    ),
    code_cell(
        """
        def summarize_pair_uncertainty(change_mask, flow, stable_bool, valid_bool, pixel_size_m, pixel_area_m2):
            change_bool = (change_mask > 0) & valid_bool
            stable_valid = stable_bool & valid_bool
            unstable_valid = (~stable_bool) & valid_bool
            reported_change_bool = change_bool & (~stable_bool)

            mag_px = np.linalg.norm(flow, axis=-1)
            mag_m = mag_px * pixel_size_m

            stable_change_pixels = int((change_bool & stable_valid).sum())
            stable_change_area_m2 = stable_change_pixels * pixel_area_m2
            stable_false_positive_rate = stable_change_pixels / max(int(stable_valid.sum()), 1)

            stable_mag = mag_m[stable_valid]
            unstable_mag = mag_m[unstable_valid]
            change_zone_mag = mag_m[change_bool]

            summary = {
                "detected_change_area_m2": float(reported_change_bool.sum() * pixel_area_m2),
                "stable_false_change_area_m2": float(stable_change_area_m2),
                "stable_false_change_rate_pct": float(stable_false_positive_rate * 100),
                "stable_flow_rmse_m": float(np.sqrt(np.mean(stable_mag ** 2))),
                "stable_flow_median_m": float(np.median(stable_mag)),
                "stable_flow_p95_m": float(np.percentile(stable_mag, 95)),
                "change_zone_mean_disp_m": float(np.mean(change_zone_mag)) if change_zone_mag.size else np.nan,
                "change_zone_p95_disp_m": float(np.percentile(change_zone_mag, 95)) if change_zone_mag.size else np.nan,
                "unstable_area_mean_disp_m": float(np.mean(unstable_mag)) if unstable_mag.size else np.nan,
                "unstable_area_p95_disp_m": float(np.percentile(unstable_mag, 95)) if unstable_mag.size else np.nan,
            }
            return pd.Series(summary)


        pair_uncertainty = summarize_pair_uncertainty(
            change_mask=change_mask_raw,
            flow=flow_raw,
            stable_bool=stable_bool,
            valid_bool=valid_bool,
            pixel_size_m=pixel_size_m,
            pixel_area_m2=pixel_area_m2,
        )

        display(pair_uncertainty.to_frame("value").style.format("{:,.4f}"))
        """
    ),
    md_cell(
        """
        ## 4. Temporal Analysis Across All Consecutive Pairs

        The next section builds a time series across all consecutive image pairs.

        For each pair, the notebook reports:

        - detected change area,
        - false-change area inside stable terrain,
        - mean and 95th percentile displacement in the detected change zone,
        - stable-terrain flow residuals as an uncertainty reference.

        When aligned images are already available in `data/output/aligned`, they are reused. The change and flow products are then recomputed from those aligned pairs so the uncertainty can still be measured on the stable terrain before any masking is applied.
        """
    ),
    code_cell(
        """
        def load_pair_images(prev_date, curr_date):
            prev_aligned = aligned_path_for_date(prev_date)
            curr_aligned = aligned_path_for_date(curr_date)

            if prev_aligned is not None and curr_aligned is not None:
                img_prev, _ = load_image(prev_aligned)
                img_curr, _ = load_image(curr_aligned)
                return img_prev, img_curr

            prev_path = image_path_for_date(prev_date, image_paths)
            curr_path = image_path_for_date(curr_date, image_paths)
            img_prev, _ = load_image(prev_path)
            img_curr_raw, _ = load_image(curr_path)
            img_curr, _, _ = align_images_constrained(img_prev, img_curr_raw, config.RASTER_MASK_PATH)
            return img_prev, img_curr


        def compute_pair_products(prev_date, curr_date):
            img_prev, img_curr = load_pair_images(prev_date, curr_date)
            valid_local = get_common_mask(img_prev, img_curr) > 0
            corrected = match_histograms(img_curr, img_prev)
            change_raw = detect_landslide_changes(
                img_prev,
                corrected,
                blur_k=config.BLUR_KERNEL_SIZE,
                threshold=CHANGE_THRESHOLD,
                min_area=config.MIN_LANDSLIDE_AREA,
            )
            flow_raw_local = compute_dense_displacement(img_prev, corrected, mask=None)
            return img_prev, img_curr, valid_local, change_raw, flow_raw_local


        records = []
        for prev_path, curr_path in zip(image_paths[:-1], image_paths[1:]):
            prev_date = extract_date(prev_path)
            curr_date = extract_date(curr_path)
            _, _, valid_local, change_arr, flow_arr = compute_pair_products(prev_date, curr_date)
            pair_summary = summarize_pair_uncertainty(
                change_mask=change_arr,
                flow=flow_arr,
                stable_bool=stable_mask == 255,
                valid_bool=valid_local,
                pixel_size_m=pixel_size_m,
                pixel_area_m2=pixel_area_m2,
            )

            records.append(
                {
                    "prev_date": parse_date(prev_date),
                    "curr_date": parse_date(curr_date),
                    "pair_label": f"{prev_date} -> {curr_date}",
                    "dt_days": (parse_date(curr_date) - parse_date(prev_date)).days,
                    **pair_summary.to_dict(),
                }
            )

        summary_df = pd.DataFrame(records).sort_values("curr_date").reset_index(drop=True)
        summary_df["change_area_ha"] = summary_df["detected_change_area_m2"] / 10000
        summary_df["change_uncertainty_ha"] = summary_df["stable_false_change_area_m2"] / 10000

        display(
            summary_df[
                [
                    "pair_label",
                    "dt_days",
                    "detected_change_area_m2",
                    "stable_false_change_area_m2",
                    "stable_flow_rmse_m",
                    "change_zone_mean_disp_m",
                    "change_zone_p95_disp_m",
                ]
            ].style.format(
                {
                    "detected_change_area_m2": "{:,.1f}",
                    "stable_false_change_area_m2": "{:,.1f}",
                    "stable_flow_rmse_m": "{:.3f}",
                    "change_zone_mean_disp_m": "{:.3f}",
                    "change_zone_p95_disp_m": "{:.3f}",
                }
            )
        )
        """
    ),
    code_cell(
        """
        fig, axes = plt.subplots(2, 1, figsize=(16, 11), sharex=True, constrained_layout=True)

        axes[0].errorbar(
            summary_df["curr_date"],
            summary_df["detected_change_area_m2"],
            yerr=summary_df["stable_false_change_area_m2"],
            fmt="-o",
            color="firebrick",
            ecolor="salmon",
            capsize=4,
            linewidth=2,
        )
        axes[0].set_title("Detected change area through time")
        axes[0].set_ylabel("Area [m²]")
        axes[0].text(
            0.01,
            0.92,
            "Error bars = false-change area measured on stable terrain",
            transform=axes[0].transAxes,
            fontsize=10,
        )

        axes[1].errorbar(
            summary_df["curr_date"],
            summary_df["change_zone_mean_disp_m"],
            yerr=summary_df["stable_flow_rmse_m"],
            fmt="-o",
            color="navy",
            ecolor="skyblue",
            capsize=4,
            linewidth=2,
            label="Mean displacement inside detected change zone",
        )
        axes[1].plot(
            summary_df["curr_date"],
            summary_df["change_zone_p95_disp_m"],
            "--s",
            color="teal",
            label="95th percentile displacement inside change zone",
        )
        axes[1].set_title("Dense-field displacement through time")
        axes[1].set_ylabel("Displacement [m]")
        axes[1].set_xlabel("Observation date")
        axes[1].text(
            0.01,
            0.92,
            "Error bars = stable-terrain flow RMSE",
            transform=axes[1].transAxes,
            fontsize=10,
        )
        axes[1].legend()

        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.xticks(rotation=45)
        plt.show()
        """
    ),
    md_cell(
        """
        ## 5. Spatial Aggregation Through Time

        Time-series plots show *when* activity increases. The maps below show *where* activity repeatedly concentrates:

        - **change recurrence**: fraction of pairwise comparisons in which a pixel was marked as changed,
        - **mean displacement magnitude**: average dense-field magnitude across all pairs,
        - **displacement variability**: temporal standard deviation of the dense-field magnitude.
        """
    ),
    code_cell(
        """
        pair_count = 0
        recurrence_sum = None
        mag_sum = None
        mag_sq_sum = None

        for prev_path, curr_path in zip(image_paths[:-1], image_paths[1:]):
            prev_date = extract_date(prev_path)
            curr_date = extract_date(curr_path)
            _, _, valid_local, change_arr_raw, flow_arr_raw = compute_pair_products(prev_date, curr_date)
            change_arr = change_arr_raw.copy()
            change_arr[stable_mask == 255] = 0
            change_arr[~valid_local] = 0
            flow_arr = flow_arr_raw.copy()
            flow_arr[stable_mask == 255] = 0
            flow_arr[~valid_local] = 0
            mag_arr = np.linalg.norm(flow_arr, axis=-1) * pixel_size_m

            if recurrence_sum is None:
                recurrence_sum = np.zeros_like(change_arr, dtype=np.float32)
                mag_sum = np.zeros_like(mag_arr, dtype=np.float64)
                mag_sq_sum = np.zeros_like(mag_arr, dtype=np.float64)

            recurrence_sum += (change_arr > 0).astype(np.float32)
            mag_sum += mag_arr
            mag_sq_sum += mag_arr ** 2
            pair_count += 1

        change_recurrence = recurrence_sum / max(pair_count, 1)
        mean_mag = mag_sum / max(pair_count, 1)
        std_mag = np.sqrt(np.maximum(mag_sq_sum / max(pair_count, 1) - mean_mag ** 2, 0))

        fig, axes = plt.subplots(1, 3, figsize=(20, 7), constrained_layout=True)

        axes[0].imshow(rgb_ref)
        im0 = axes[0].imshow(change_recurrence, cmap="Reds", alpha=0.75, vmin=0, vmax=1)
        axes[0].set_title("Change recurrence")
        axes[0].axis("off")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Fraction of pairs")

        axes[1].imshow(rgb_ref)
        im1 = axes[1].imshow(mean_mag, cmap="viridis", alpha=0.75)
        axes[1].set_title("Mean dense-field magnitude")
        axes[1].axis("off")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Mean displacement [m]")

        axes[2].imshow(rgb_ref)
        im2 = axes[2].imshow(std_mag, cmap="magma", alpha=0.75)
        axes[2].set_title("Temporal variability of displacement")
        axes[2].axis("off")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="Std. dev. [m]")

        plt.show()
        """
    ),
    md_cell(
        """
        ## Interpretation Notes

        A few reading rules can help when you discuss the results:

        - If the **change area** is close to the stable-area false positive area, the mapped change should be treated cautiously.
        - If the **mean displacement** in the change zone is only as large as the stable-terrain RMSE, the motion signal is weak.
        - Pixels with both **high recurrence** and **high mean displacement** are the most likely locations of persistent activity.

        If you want, this notebook can be extended one step further with exported tables, animated maps, or conversion from image displacements to downslope velocity rates.
        """
    ),
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.x",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output_path = Path(__file__).with_name("Intership2.ipynb")
output_path.write_text(json.dumps(nb, indent=2))
print(f"Wrote {output_path}")
