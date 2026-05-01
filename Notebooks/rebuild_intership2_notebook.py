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
            2. **Area-based result analysis**: how the detected changes and dense-field products behave in space and through time, separately for each polygon in `analysis_areas.shp`, including empirical uncertainty estimates derived from the stable terrain.

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
            import geopandas as gpd
            import matplotlib.dates as mdates
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import rasterio
            from IPython.display import display
            from rasterio.features import rasterize

            try:
                import ipywidgets as widgets
            except ImportError:
                widgets = None


            def get_start_dir():
                try:
                    return Path.cwd()
                except FileNotFoundError:
                    pwd_env = os.environ.get("PWD")
                    if pwd_env:
                        return Path(pwd_env)
                    return Path("/Users/linamaria/Documents/University/3_Intership_Part2/Landslide_project")


            def find_project_root(start_dir):
                candidates = [start_dir, start_dir.parent]
                for candidate in candidates:
                    if (candidate / "src").exists() and (candidate / "config.py").exists():
                        return candidate
                    if candidate.name == "Notebooks" and (candidate.parent / "src").exists() and (candidate.parent / "config.py").exists():
                        return candidate.parent
                raise FileNotFoundError("Could not locate the project root containing 'src/' and 'config.py'.")


            START_DIR = get_start_dir()
            PROJECT_ROOT = find_project_root(START_DIR)

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
            EXAMPLE_PREV_DATE = "2023-02-11"
            EXAMPLE_CURR_DATE = "2023-02-16"
            CHANGE_THRESHOLD = config.CHANGE_THRESHOLD
            PIXEL_TO_METER_OVERRIDE = None
            ANALYSIS_AREAS_PATH = Path(config.INPUT_FOLDER) / "analysis_areas.shp"
            PRECIPITATION_CANDIDATES = [
                Path(config.INPUT_FOLDER) / "descargaDhime.csv",
                PROJECT_ROOT / "data" / "descargaDhime.csv",
            ]
            PRECIPITATION_PATH = next((path for path in PRECIPITATION_CANDIDATES if path.exists()), PRECIPITATION_CANDIDATES[0])


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


            image_paths = config.get_sorted_image_paths()
            if len(image_paths) < 2:
                raise RuntimeError("At least two dated input images are required.")

            with rasterio.open(image_paths[0]) as src:
                pixel_size_m = PIXEL_TO_METER_OVERRIDE or float(np.mean(np.abs(src.res)))
                pixel_area_m2 = float(abs(src.transform.a * src.transform.e))
                raster_shape = src.shape
                raster_transform = src.transform
                raster_crs = src.crs

            stable_mask = cv2.imread(config.RASTER_MASK_PATH, cv2.IMREAD_GRAYSCALE)
            if stable_mask is None:
                raise FileNotFoundError(f"Stable raster mask not found: {config.RASTER_MASK_PATH}")

            def get_stable_bool(shape_hw):
                shape_hw = tuple(shape_hw)
                if stable_mask.shape != shape_hw:
                    resized_mask = cv2.resize(
                        stable_mask,
                        (shape_hw[1], shape_hw[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                else:
                    resized_mask = stable_mask
                return resized_mask == 255

            if not ANALYSIS_AREAS_PATH.exists():
                raise FileNotFoundError(f"Analysis areas shapefile not found: {ANALYSIS_AREAS_PATH}")

            analysis_gdf = gpd.read_file(ANALYSIS_AREAS_PATH)
            if analysis_gdf.crs != raster_crs:
                analysis_gdf = analysis_gdf.to_crs(raster_crs)

            stable_bool_global = stable_mask == 255
            analysis_areas = []
            for idx, row in analysis_gdf.iterrows():
                area_name = str(row.get("Type", row.get("area", f"area_{idx + 1}"))).strip() or f"area_{idx + 1}"
                area_mask = rasterize(
                    [(row.geometry, 1)],
                    out_shape=raster_shape,
                    transform=raster_transform,
                    fill=0,
                    dtype="uint8",
                ).astype(bool)
                unstable_area_mask = area_mask & (~stable_bool_global)
                analysis_areas.append(
                    {
                        "name": area_name,
                        "geometry": row.geometry,
                        "mask": unstable_area_mask,
                        "pixel_count": int(unstable_area_mask.sum()),
                        "area_m2": float(unstable_area_mask.sum() * pixel_area_m2),
                    }
                )

            example_prev_path = image_path_for_date(EXAMPLE_PREV_DATE, image_paths)
            example_curr_path = image_path_for_date(EXAMPLE_CURR_DATE, image_paths)

            print(f"Loaded {len(image_paths)} input images")
            print(f"Example pair: {EXAMPLE_PREV_DATE} -> {EXAMPLE_CURR_DATE}")
            print(f"Pixel size: {pixel_size_m:.4f} m")
            print(f"Pixel area: {pixel_area_m2:.4f} m²")
            print(f"Analysis areas: {[area['name'] for area in analysis_areas]}")
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

            ref_gray = cv2.cvtColor(img_ref_raw, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(img_curr_raw, cv2.COLOR_BGR2GRAY)
            aligned_gray = cv2.cvtColor(img_curr_aligned, cv2.COLOR_BGR2GRAY)
            stable_bool_ref = get_stable_bool(ref_gray.shape)
            stable_bool_aligned = get_stable_bool(aligned_gray.shape)

            common_h = min(ref_gray.shape[0], curr_gray.shape[0])
            common_w = min(ref_gray.shape[1], curr_gray.shape[1])
            ref_gray_before = ref_gray[:common_h, :common_w]
            curr_gray_before = curr_gray[:common_h, :common_w]
            stable_bool_before = stable_bool_ref[:common_h, :common_w]

            diff_before = cv2.absdiff(ref_gray_before, curr_gray_before)
            diff_after = cv2.absdiff(ref_gray, aligned_gray)

            before_stable_mae = float(diff_before[stable_bool_before].mean())
            after_stable_mae = float(diff_after[stable_bool_aligned].mean())
            improvement = before_stable_mae - after_stable_mae

            print(f"Stable-terrain MAE before alignment: {before_stable_mae:.2f}")
            print(f"Stable-terrain MAE after alignment:  {after_stable_mae:.2f}")
            print(f"Residual improvement:                {improvement:.2f}")
            """
        ),
        md_cell(
            """
            ### Alignment Diagnostics Plot

            This 2x2 figure shows the reference image, the stable-terrain keypoints used for registration, and the residual image differences before and after alignment.
            """
        ),
        code_cell(
            """
            fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)

            axes[0, 0].imshow(cv2.cvtColor(img_ref_raw, cv2.COLOR_BGR2RGB))
            axes[0, 0].imshow(stable_bool_ref, cmap="Greens", alpha=0.22)
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
            axes[1, 1].imshow(stable_bool_aligned, cmap="Greens", alpha=0.12)
            axes[1, 1].set_title("Absolute difference after alignment")
            axes[1, 1].axis("off")

            fig.colorbar(
                im0,
                ax=axes[1, 0],
                fraction=0.046,
                pad=0.08,
                orientation="horizontal",
                label="Gray-level residual",
            )
            fig.colorbar(
                im1,
                ax=axes[1, 1],
                fraction=0.046,
                pad=0.08,
                orientation="horizontal",
                label="Gray-level residual",
            )
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

            For the visualization below, the notebook also estimates a **per-pixel flow uncertainty** using forward-backward consistency. Pixels with high uncertainty are filtered out, and the remaining reliable vectors are aggregated onto a coarser grid before plotting magnitude, direction, and quiver vectors.
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
            stable_bool_analysis = get_stable_bool(change_mask_raw.shape)
            change_mask = change_mask_raw.copy()
            change_mask[stable_bool_analysis] = 0
            change_mask[~valid_bool] = 0

            flow_raw = compute_dense_displacement(img_ref_raw, img_curr_corrected, mask=None)
            stable_bool_flow = get_stable_bool(flow_raw.shape[:2])
            flow = flow_raw.copy()
            flow[stable_bool_flow] = 0
            flow[~valid_bool] = 0
            flow_magnitude_px = np.linalg.norm(flow, axis=-1)
            flow_magnitude_m = flow_magnitude_px * pixel_size_m
            flow_direction_deg = (np.degrees(np.arctan2(flow[..., 1], flow[..., 0])) + 360) % 360
            direction_valid = valid_bool & (flow_magnitude_px > 0.25)

            backward_flow_raw = compute_dense_displacement(img_curr_corrected, img_ref_raw, mask=None)
            h, w = flow_raw.shape[:2]
            grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            mapped_x = grid_x + flow_raw[..., 0]
            mapped_y = grid_y + flow_raw[..., 1]
            backward_at_forward = np.dstack(
                [
                    cv2.remap(backward_flow_raw[..., 0], mapped_x, mapped_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan),
                    cv2.remap(backward_flow_raw[..., 1], mapped_x, mapped_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan),
                ]
            )
            flow_uncertainty_px = np.linalg.norm(flow_raw + backward_at_forward, axis=-1)
            flow_uncertainty_px[~valid_bool] = np.nan

            stable_uncertainty = flow_uncertainty_px[stable_bool_flow & valid_bool]
            uncertainty_threshold_px = float(np.nanpercentile(stable_uncertainty, 95)) if np.isfinite(stable_uncertainty).any() else 1.0
            low_uncertainty_mask = valid_bool & np.isfinite(flow_uncertainty_px) & (flow_uncertainty_px <= uncertainty_threshold_px)
            reliable_motion_mask = low_uncertainty_mask & (~stable_bool_flow) & (flow_magnitude_px > 0.25)

            def build_coarse_field(flow_field, valid_mask, uncertainty_field, step=40, min_count=12):
                h, w = valid_mask.shape
                coarse_u = np.full((h, w), np.nan, dtype=np.float32)
                coarse_v = np.full((h, w), np.nan, dtype=np.float32)
                coarse_unc = np.full((h, w), np.nan, dtype=np.float32)
                qx, qy, qu, qv, qunc = [], [], [], [], []

                for row_start in range(0, h, step):
                    for col_start in range(0, w, step):
                        row_end = min(row_start + step, h)
                        col_end = min(col_start + step, w)
                        block_mask = valid_mask[row_start:row_end, col_start:col_end]
                        if int(block_mask.sum()) < min_count:
                            continue

                        block_u = flow_field[row_start:row_end, col_start:col_end, 0][block_mask]
                        block_v = flow_field[row_start:row_end, col_start:col_end, 1][block_mask]
                        block_unc = uncertainty_field[row_start:row_end, col_start:col_end][block_mask]

                        u_med = float(np.nanmedian(block_u))
                        v_med = float(np.nanmedian(block_v))
                        unc_med = float(np.nanmedian(block_unc))

                        coarse_u[row_start:row_end, col_start:col_end] = u_med
                        coarse_v[row_start:row_end, col_start:col_end] = v_med
                        coarse_unc[row_start:row_end, col_start:col_end] = unc_med

                        qx.append((col_start + col_end - 1) / 2)
                        qy.append((row_start + row_end - 1) / 2)
                        qu.append(u_med)
                        qv.append(v_med)
                        qunc.append(unc_med)

                return {
                    "u": coarse_u,
                    "v": coarse_v,
                    "uncertainty": coarse_unc,
                    "x": np.array(qx),
                    "y": np.array(qy),
                    "qu": np.array(qu),
                    "qv": np.array(qv),
                    "qunc": np.array(qunc),
                }


            coarse_step = 40
            coarse_field = build_coarse_field(flow_raw, reliable_motion_mask, flow_uncertainty_px, step=coarse_step, min_count=10)
            coarse_flow = np.dstack([coarse_field["u"], coarse_field["v"]])
            coarse_mag_px = np.linalg.norm(coarse_flow, axis=-1)
            coarse_mag_m = coarse_mag_px * pixel_size_m
            coarse_dir_deg = (np.degrees(np.arctan2(coarse_field["v"], coarse_field["u"])) + 360) % 360
            coarse_valid = np.isfinite(coarse_mag_px)

            coarse_mag_m_masked = np.ma.masked_where(~coarse_valid, coarse_mag_m)
            coarse_dir_deg_masked = np.ma.masked_where(~coarse_valid, coarse_dir_deg)

            change_area_m2 = float((change_mask > 0).sum() * pixel_area_m2)
            print(f"Detected change area: {change_area_m2:,.1f} m²")
            print(f"Maximum displacement: {flow_magnitude_m.max():.2f} m")
            print(f"Low-uncertainty threshold: {uncertainty_threshold_px:.2f} px")
            print(f"Reliable dense-flow pixels: {int(reliable_motion_mask.sum()):,}")
            """
        ),
        md_cell(
            """
            ### Change Detection Overview Plot

            This 3-panel figure summarizes the example pair with the reference orthophoto, the absolute image difference, and the final binary change mask.
            """
        ),
        code_cell(
            """
            rgb_ref = cv2.cvtColor(img_ref_raw, cv2.COLOR_BGR2RGB)
            corrected_gray = cv2.cvtColor(img_curr_corrected, cv2.COLOR_BGR2GRAY)
            abs_difference = cv2.absdiff(ref_gray, corrected_gray).astype(np.float32)
            abs_difference[~valid_bool] = np.nan
            abs_difference_masked = np.ma.masked_invalid(abs_difference)
            diff_vmax = np.nanpercentile(abs_difference, 99) if np.isfinite(abs_difference).any() else 1.0

            fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

            axes[0].imshow(rgb_ref)
            axes[0].set_title(f"Reference orthophoto ({EXAMPLE_PREV_DATE})")
            axes[0].axis("off")

            im_diff = axes[1].imshow(abs_difference_masked, cmap="magma", vmin=0, vmax=diff_vmax)
            axes[1].set_title(f"Absolute difference ({EXAMPLE_PREV_DATE} vs {EXAMPLE_CURR_DATE})")
            axes[1].axis("off")
            fig.colorbar(
                im_diff,
                ax=axes[1],
                fraction=0.046,
                pad=0.08,
                orientation="horizontal",
                label="Gray-level residual",
            )

            axes[2].imshow(rgb_ref)
            axes[2].imshow(np.ma.masked_where(change_mask == 0, change_mask), cmap="Reds", alpha=0.4)
            axes[2].set_title("Final binary change mask overlay")
            axes[2].axis("off")

            plt.show()
            fig.savefig(
                PROJECT_ROOT / "data" / "output" / "change_detection_triptych.jpg",
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
                pil_kwargs={"quality": 95},
            )
            """
        ),
        md_cell(
            """
            ### Uncertainty Diagnostics Plot

            This 2x2 figure shows the forward-backward inconsistency, the stable-terrain overlay used to set the threshold, the low-uncertainty mask, and the reliable motion vectors.
            """
        ),
        code_cell(
            """
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
            plt.show()
            fig.savefig(
                PROJECT_ROOT / "data" / "output" / "uncertainty_diagnostics_4panel.jpg",
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
                pil_kwargs={"quality": 95},
            )
            """
        ),
        md_cell(
            """
            ### Change and Displacement Summary Plot

            This 3x2 figure combines the change mask, coarse displacement magnitude, movement direction, and quiver vectors for the example pair.
            """
        ),
        code_cell(
            """
            rgb_ref = cv2.cvtColor(img_ref_raw, cv2.COLOR_BGR2RGB)
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

            direction_cmap = plt.get_cmap("twilight_shifted").copy()
            direction_cmap.set_bad(alpha=0)
            direction_display = coarse_dir_deg_masked
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

            im_dir = axes[1, 1].imshow(direction_display, cmap=direction_cmap, vmin=0, vmax=360)
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
            axes[2, 1].imshow(direction_display, cmap=direction_cmap, vmin=0, vmax=360, alpha=alpha_display)
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
            def summarize_pair_uncertainty(
                change_mask,
                flow,
                stable_bool,
                valid_bool,
                pixel_size_m,
                pixel_area_m2,
                unstable_area_mask=None,
                reliable_mask=None,
                flow_uncertainty_px=None,
                low_uncertainty_mask=None,
            ):
                if unstable_area_mask is None:
                    unstable_area_valid = valid_bool & (~stable_bool)
                else:
                    unstable_area_valid = valid_bool & unstable_area_mask & (~stable_bool)

                change_bool = (change_mask > 0) & unstable_area_valid
                stable_valid = valid_bool & stable_bool
                if reliable_mask is None:
                    unstable_valid = unstable_area_valid
                    reported_change_bool = change_bool
                else:
                    unstable_valid = unstable_area_valid & reliable_mask
                    reported_change_bool = change_bool & reliable_mask

                mag_px = np.linalg.norm(flow, axis=-1)
                mag_m = mag_px * pixel_size_m
                flow_x = flow[..., 0]
                flow_y = flow[..., 1]

                stable_change_pixels = int(((change_mask > 0) & stable_valid).sum())
                stable_change_area_m2 = stable_change_pixels * pixel_area_m2
                stable_false_positive_rate = stable_change_pixels / max(int(stable_valid.sum()), 1)

                stable_mag = mag_m[stable_valid]
                unstable_mag = mag_m[unstable_valid]
                change_zone_mag = mag_m[reported_change_bool]
                change_zone_u = flow_x[reported_change_bool]
                change_zone_v = flow_y[reported_change_bool]
                stable_uncertainty = (
                    flow_uncertainty_px[stable_valid]
                    if flow_uncertainty_px is not None
                    else np.array([], dtype=float)
                )

                valid_pixel_count = int(valid_bool.sum())
                active_area_pixel_count = int(unstable_area_valid.sum())
                if low_uncertainty_mask is None:
                    low_uncertainty_valid_fraction = np.nan
                    low_uncertainty_active_fraction = np.nan
                else:
                    low_uncertainty_valid_fraction = float(
                        (low_uncertainty_mask & valid_bool).sum() / valid_pixel_count
                    ) if valid_pixel_count else np.nan
                    low_uncertainty_active_fraction = float(
                        (low_uncertainty_mask & unstable_area_valid).sum() / active_area_pixel_count
                    ) if active_area_pixel_count else np.nan

                if change_zone_u.size:
                    mean_u = float(np.mean(change_zone_u))
                    mean_v = float(np.mean(change_zone_v))
                    mean_direction_deg = float((np.degrees(np.arctan2(mean_v, mean_u)) + 360) % 360)
                    direction_strength = float(
                        np.hypot(mean_u, mean_v) / max(np.mean(np.hypot(change_zone_u, change_zone_v)), 1e-9)
                    )
                else:
                    mean_u = np.nan
                    mean_v = np.nan
                    mean_direction_deg = np.nan
                    direction_strength = np.nan

                summary = {
                    "analysis_area_m2": float(unstable_area_valid.sum() * pixel_area_m2),
                    "detected_change_area_m2": float(reported_change_bool.sum() * pixel_area_m2),
                    "stable_false_change_area_m2": float(stable_change_area_m2),
                    "stable_false_change_rate_pct": float(stable_false_positive_rate * 100),
                    "stable_flow_rmse_m": float(np.sqrt(np.mean(stable_mag ** 2))) if stable_mag.size else np.nan,
                    "stable_flow_median_m": float(np.median(stable_mag)) if stable_mag.size else np.nan,
                    "stable_flow_p95_m": float(np.percentile(stable_mag, 95)) if stable_mag.size else np.nan,
                    "stable_fb_median_inconsistency_px": float(np.nanmedian(stable_uncertainty)) if stable_uncertainty.size else np.nan,
                    "stable_fb_p95_inconsistency_px": float(np.nanpercentile(stable_uncertainty, 95)) if stable_uncertainty.size else np.nan,
                    "low_uncertainty_fraction_valid": low_uncertainty_valid_fraction,
                    "low_uncertainty_fraction_active_area": low_uncertainty_active_fraction,
                    "change_zone_mean_disp_m": float(np.mean(change_zone_mag)) if change_zone_mag.size else np.nan,
                    "change_zone_p95_disp_m": float(np.percentile(change_zone_mag, 95)) if change_zone_mag.size else np.nan,
                    "change_zone_mean_u_px": mean_u,
                    "change_zone_mean_v_px": mean_v,
                    "change_zone_mean_direction_deg": mean_direction_deg,
                    "change_zone_direction_strength": direction_strength,
                    "unstable_area_mean_disp_m": float(np.mean(unstable_mag)) if unstable_mag.size else np.nan,
                    "unstable_area_p95_disp_m": float(np.percentile(unstable_mag, 95)) if unstable_mag.size else np.nan,
                }
                return pd.Series(summary)


            pair_uncertainty = summarize_pair_uncertainty(
                change_mask=change_mask_raw,
                flow=flow_raw,
                stable_bool=stable_bool_flow,
                valid_bool=valid_bool,
                pixel_size_m=pixel_size_m,
                pixel_area_m2=pixel_area_m2,
                reliable_mask=reliable_motion_mask,
                flow_uncertainty_px=flow_uncertainty_px,
                low_uncertainty_mask=low_uncertainty_mask,
            )

            display(pair_uncertainty.to_frame("value").style.format("{:,.4f}"))
            """
        ),
        md_cell(
            """
            ## 4. Temporal Analysis by Analysis Area

            Up to this point, the workflow has been run on the **whole image**. From here onward, the final summaries are broken down by the unstable sub-areas from `analysis_areas.shp`.

            For each pair and for each analysis area, the notebook reports:

            - detected change area,
            - false-change area inside stable terrain,
            - mean and 95th percentile displacement in the detected change zone,
            - stable-terrain flow residuals as an uncertainty reference.

            When aligned images are already available in `data/output/aligned`, they are reused. The change and flow products are recomputed from those aligned pairs so the uncertainty can still be measured on the stable terrain before any masking is applied.

            Daily precipitation from `descargaDhime.csv` is also added here so the image-based observations can be read together with rainfall forcing.
            """
        ),
        md_cell(
            """
            ### Analysis Area Map

            This plot shows the unstable analysis polygons on top of the reference image together with the stable-terrain mask used throughout the workflow.
            """
        ),
        code_cell(
            """
            area_overview = pd.DataFrame(
                [
                    {
                        "area_name": area["name"],
                        "pixels": area["pixel_count"],
                        "area_m2": area["area_m2"],
                        "area_ha": area["area_m2"] / 10000,
                    }
                    for area in analysis_areas
                ]
            )
            display(area_overview.style.format({"area_m2": "{:,.1f}", "area_ha": "{:.3f}"}))

            fig, axes = plt.subplots(1, len(analysis_areas), figsize=(7 * len(analysis_areas), 6), constrained_layout=True)
            axes = np.atleast_1d(axes)
            for ax, area in zip(axes, analysis_areas):
                ax.imshow(rgb_ref)
                ax.imshow(np.ma.masked_where(~area["mask"], area["mask"]), cmap="autumn", alpha=0.35)
                ax.imshow(np.ma.masked_where(~stable_bool_global, stable_bool_global), cmap="Greens", alpha=0.18)
                ax.set_title(f"Unstable analysis area: {area['name']}")
                ax.axis("off")
            plt.show()
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
                stable_bool_local = get_stable_bool(valid_local.shape)
                corrected = match_histograms(img_curr, img_prev)
                change_raw = detect_landslide_changes(
                    img_prev,
                    corrected,
                    blur_k=config.BLUR_KERNEL_SIZE,
                    threshold=CHANGE_THRESHOLD,
                    min_area=config.MIN_LANDSLIDE_AREA,
                )
                flow_raw_local = compute_dense_displacement(img_prev, corrected, mask=None)
                backward_flow_local = compute_dense_displacement(corrected, img_prev, mask=None)
                h, w = flow_raw_local.shape[:2]
                grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
                mapped_x = grid_x + flow_raw_local[..., 0]
                mapped_y = grid_y + flow_raw_local[..., 1]
                backward_at_forward = np.dstack(
                    [
                        cv2.remap(backward_flow_local[..., 0], mapped_x, mapped_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan),
                        cv2.remap(backward_flow_local[..., 1], mapped_x, mapped_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan),
                    ]
                )
                flow_uncertainty_local = np.linalg.norm(flow_raw_local + backward_at_forward, axis=-1)
                flow_uncertainty_local[~valid_local] = np.nan

                stable_uncertainty_local = flow_uncertainty_local[stable_bool_local & valid_local]
                uncertainty_threshold_local = float(np.nanpercentile(stable_uncertainty_local, 95)) if np.isfinite(stable_uncertainty_local).any() else 1.0
                mag_local = np.linalg.norm(flow_raw_local, axis=-1)
                low_uncertainty_local = valid_local & np.isfinite(flow_uncertainty_local) & (flow_uncertainty_local <= uncertainty_threshold_local)
                reliable_motion_local = low_uncertainty_local & (~stable_bool_local) & (mag_local > 0.25)

                return (
                    img_prev,
                    img_curr,
                    valid_local,
                    stable_bool_local,
                    change_raw,
                    flow_raw_local,
                    flow_uncertainty_local,
                    low_uncertainty_local,
                    reliable_motion_local,
                    uncertainty_threshold_local,
                )


            if PRECIPITATION_PATH.exists():
                precipitation_df = pd.read_csv(PRECIPITATION_PATH)
                precipitation_df["Fecha"] = pd.to_datetime(precipitation_df["Fecha"])
                precipitation_df["date"] = precipitation_df["Fecha"].dt.normalize()
                precipitation_df["Valor"] = pd.to_numeric(precipitation_df["Valor"], errors="coerce").fillna(0.0)
                precipitation_daily = (
                    precipitation_df.groupby("date", as_index=False)["Valor"]
                    .sum()
                    .rename(columns={"Valor": "precip_mm"})
                )
                print(f"Loaded precipitation data from {PRECIPITATION_PATH.name}")
            else:
                precipitation_daily = pd.DataFrame(columns=["date", "precip_mm"])
                print(f"Precipitation file not found: {PRECIPITATION_PATH}. Rainfall columns will remain empty.")


            def precipitation_between(start_date, end_date):
                start_ts = pd.Timestamp(start_date)
                end_ts = pd.Timestamp(end_date)
                mask = (precipitation_daily["date"] > start_ts) & (precipitation_daily["date"] <= end_ts)
                if precipitation_daily.empty:
                    return np.nan
                return float(precipitation_daily.loc[mask, "precip_mm"].sum())


            def antecedent_precipitation(end_date, days=7):
                end_ts = pd.Timestamp(end_date)
                start_ts = end_ts - pd.Timedelta(days=days - 1)
                mask = (precipitation_daily["date"] >= start_ts) & (precipitation_daily["date"] <= end_ts)
                if precipitation_daily.empty:
                    return np.nan
                return float(precipitation_daily.loc[mask, "precip_mm"].sum())


            records = []
            for prev_path, curr_path in zip(image_paths[:-1], image_paths[1:]):
                prev_date = extract_date(prev_path)
                curr_date = extract_date(curr_path)
                (
                    _,
                    _,
                    valid_local,
                    stable_bool_local,
                    change_arr,
                    flow_arr,
                    flow_uncertainty_local,
                    low_uncertainty_local,
                    reliable_motion_local,
                    _,
                ) = compute_pair_products(prev_date, curr_date)

                for area in analysis_areas:
                    pair_summary = summarize_pair_uncertainty(
                        change_mask=change_arr,
                        flow=flow_arr,
                        stable_bool=stable_bool_local,
                        valid_bool=valid_local,
                        pixel_size_m=pixel_size_m,
                        pixel_area_m2=pixel_area_m2,
                        unstable_area_mask=area["mask"],
                        reliable_mask=reliable_motion_local,
                        flow_uncertainty_px=flow_uncertainty_local,
                        low_uncertainty_mask=low_uncertainty_local,
                    )
                    records.append(
                        {
                            "area_name": area["name"],
                            "prev_date": parse_date(prev_date),
                            "curr_date": parse_date(curr_date),
                            "pair_label": f"{prev_date} -> {curr_date}",
                            "dt_days": (parse_date(curr_date) - parse_date(prev_date)).days,
                            "pair_precip_mm": precipitation_between(prev_date, curr_date),
                            "antecedent_7d_precip_mm": antecedent_precipitation(curr_date, days=7),
                            **pair_summary.to_dict(),
                        }
                    )

            summary_df = pd.DataFrame(records).sort_values(["area_name", "curr_date"]).reset_index(drop=True)
            summary_df["change_area_ha"] = summary_df["detected_change_area_m2"] / 10000
            summary_df["change_uncertainty_ha"] = summary_df["stable_false_change_area_m2"] / 10000

            display(
                summary_df[
                    [
                        "area_name",
                        "pair_label",
                        "dt_days",
                        "analysis_area_m2",
                        "detected_change_area_m2",
                        "stable_false_change_area_m2",
                        "stable_flow_rmse_m",
                        "stable_fb_median_inconsistency_px",
                        "stable_fb_p95_inconsistency_px",
                        "low_uncertainty_fraction_valid",
                        "low_uncertainty_fraction_active_area",
                        "pair_precip_mm",
                        "antecedent_7d_precip_mm",
                        "change_zone_mean_disp_m",
                        "change_zone_p95_disp_m",
                        "change_zone_mean_direction_deg",
                    ]
                ].style.format(
                    {
                        "analysis_area_m2": "{:,.1f}",
                        "detected_change_area_m2": "{:,.1f}",
                        "stable_false_change_area_m2": "{:,.1f}",
                        "stable_flow_rmse_m": "{:.3f}",
                        "stable_fb_median_inconsistency_px": "{:.3f}",
                        "stable_fb_p95_inconsistency_px": "{:.3f}",
                        "low_uncertainty_fraction_valid": "{:.3f}",
                        "low_uncertainty_fraction_active_area": "{:.3f}",
                        "pair_precip_mm": "{:.1f}",
                        "antecedent_7d_precip_mm": "{:.1f}",
                        "change_zone_mean_disp_m": "{:.3f}",
                        "change_zone_p95_disp_m": "{:.3f}",
                        "change_zone_mean_direction_deg": "{:.1f}",
                    }
                )
            )
            """
        ),
        md_cell(
            """
            ### Rainfall, Change Area, and Displacement Time Series

            This 3-panel plot compares rainfall forcing with detected change area and dense-field displacement through time for each analysis area.
            """
        ),
        code_cell(
            """
            area_colors = {area["name"]: plt.cm.Set2(i) for i, area in enumerate(analysis_areas)}

            rain_start = summary_df["prev_date"].min() - pd.Timedelta(days=2)
            rain_end = summary_df["curr_date"].max() + pd.Timedelta(days=2)
            precipitation_plot_df = precipitation_daily[
                (precipitation_daily["date"] >= rain_start) & (precipitation_daily["date"] <= rain_end)
            ].copy() if not precipitation_daily.empty else pd.DataFrame(columns=["date", "precip_mm"])

            fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True, constrained_layout=True)

            if precipitation_plot_df.empty:
                axes[0].text(
                    0.5,
                    0.5,
                    "No precipitation file available",
                    ha="center",
                    va="center",
                    transform=axes[0].transAxes,
                    fontsize=12,
                )
                axes[0].set_title("Daily precipitation")
                axes[0].set_ylabel("Rainfall [mm/day]")
            else:
                axes[0].bar(
                    precipitation_plot_df["date"],
                    precipitation_plot_df["precip_mm"],
                    width=1.0,
                    color="steelblue",
                    alpha=0.85,
                )
                axes[0].set_title("Daily precipitation")
                axes[0].set_ylabel("Rainfall [mm/day]")
            for curr_date in sorted(summary_df["curr_date"].unique()):
                axes[0].axvline(curr_date, color="black", linestyle="--", alpha=0.15, linewidth=1)

            for area_name, area_df in summary_df.groupby("area_name"):
                axes[1].errorbar(
                    area_df["curr_date"],
                    area_df["detected_change_area_m2"],
                    yerr=area_df["stable_false_change_area_m2"],
                    fmt="-o",
                    capsize=4,
                    linewidth=2,
                    color=area_colors.get(area_name),
                    label=area_name,
                )
            axes[1].set_title("Detected change area through time by analysis area")
            axes[1].set_ylabel("Area [m²]")
            axes[1].text(
                0.01,
                0.92,
                "Error bars = false-change area measured on stable terrain inside each analysis polygon",
                transform=axes[1].transAxes,
                fontsize=10,
            )
            axes[1].legend(title="Area")

            for area_name, area_df in summary_df.groupby("area_name"):
                axes[2].errorbar(
                    area_df["curr_date"],
                    area_df["change_zone_mean_disp_m"],
                    yerr=area_df["stable_flow_rmse_m"],
                    fmt="-o",
                    capsize=4,
                    linewidth=2,
                    color=area_colors.get(area_name),
                    label=f"{area_name} mean",
                )
                axes[2].plot(
                    area_df["curr_date"],
                    area_df["change_zone_p95_disp_m"],
                    "--s",
                    color=area_colors.get(area_name),
                    alpha=0.75,
                    label=f"{area_name} p95",
                )
            axes[2].set_title("Dense-field displacement through time by analysis area")
            axes[2].set_ylabel("Displacement [m]")
            axes[2].set_xlabel("Observation date")
            axes[2].text(
                0.01,
                0.92,
                "Error bars = stable-terrain flow RMSE inside each analysis polygon",
                transform=axes[2].transAxes,
                fontsize=10,
            )
            axes[2].legend(ncol=2, fontsize=9)

            axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            plt.xticks(rotation=45)
            plt.show()
            """
        ),
        md_cell(
            """
            ### Low-Uncertainty and Forward-Backward Metrics Plot

            This 2x2 plot tracks stable-area forward-backward inconsistency and the fraction of pixels classified as low-uncertainty for each pair.
            """
        ),
        code_cell(
            """
            fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, constrained_layout=True)
            axes = axes.ravel()

            pair_level_df = (
                summary_df.sort_values(["curr_date", "area_name"])
                .drop_duplicates(subset=["curr_date"])
                .reset_index(drop=True)
            )

            pair_metric_specs = [
                ("stable_fb_p95_inconsistency_px", "Stable-area p95 forward-backward inconsistency", "Inconsistency [px]"),
                ("stable_fb_median_inconsistency_px", "Stable-area median forward-backward inconsistency", "Inconsistency [px]"),
                ("low_uncertainty_fraction_valid", "Fraction of valid pixels classified as low-uncertainty", "Fraction [-]"),
            ]

            for ax, (column, title, ylabel) in zip(axes[:3], pair_metric_specs):
                ax.plot(
                    pair_level_df["curr_date"],
                    pair_level_df[column],
                    "-o",
                    linewidth=2,
                    markersize=6,
                    color="black",
                    label="All pairs",
                )
                ax.set_title(title)
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.25)
                ax.legend()

            active_ax = axes[3]
            for area_name, area_df in summary_df.groupby("area_name"):
                active_ax.plot(
                    area_df["curr_date"],
                    area_df["low_uncertainty_fraction_active_area"],
                    "-o",
                    linewidth=2,
                    markersize=6,
                    color=area_colors.get(area_name),
                    label=area_name,
                )
            active_ax.set_title("Fraction of active-area pixels classified as low-uncertainty")
            active_ax.set_ylabel("Fraction [-]")
            active_ax.grid(True, alpha=0.25)
            active_ax.legend(title="Area")

            axes[2].set_xlabel("Observation date")
            axes[3].set_xlabel("Observation date")
            axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            for ax in axes[2:]:
                ax.tick_params(axis="x", rotation=45)

            plt.show()
            """
        ),
        md_cell(
            """
            ### Two-Panel Uncertainty Summary Plot

            This figure summarizes detected change area with stable false-change error bars, and change-zone displacement with stable-flow RMSE plus the p95 displacement line.
            """
        ),
        code_cell(
            """
            fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True, constrained_layout=True)

            for area_name, area_df in summary_df.groupby("area_name"):
                axes[0].errorbar(
                    area_df["curr_date"],
                    area_df["detected_change_area_m2"],
                    yerr=area_df["stable_false_change_area_m2"],
                    fmt="-o",
                    capsize=4,
                    linewidth=2,
                    color=area_colors.get(area_name),
                    label=area_name,
                )

            axes[0].set_title("Uncertainty Summary: detected change area")
            axes[0].set_ylabel("Area [m²]")
            axes[0].text(
                0.01,
                0.92,
                "Error bars = stable false-change area inside each analysis polygon",
                transform=axes[0].transAxes,
                fontsize=10,
            )
            axes[0].legend(title="Area")

            for area_name, area_df in summary_df.groupby("area_name"):
                axes[1].errorbar(
                    area_df["curr_date"],
                    area_df["change_zone_mean_disp_m"],
                    yerr=area_df["stable_flow_rmse_m"],
                    fmt="-o",
                    capsize=4,
                    linewidth=2,
                    color=area_colors.get(area_name),
                    label=f"{area_name} mean",
                )
                axes[1].plot(
                    area_df["curr_date"],
                    area_df["change_zone_p95_disp_m"],
                    "--s",
                    color=area_colors.get(area_name),
                    alpha=0.75,
                    label=f"{area_name} p95",
                )

            axes[1].set_title("Uncertainty Summary: change-zone displacement")
            axes[1].set_ylabel("Displacement [m]")
            axes[1].set_xlabel("Observation date")
            axes[1].text(
                0.01,
                0.92,
                "Error bars = stable-flow RMSE; dashed line = change-zone p95 displacement",
                transform=axes[1].transAxes,
                fontsize=10,
            )
            axes[1].legend(ncol=2, fontsize=9)
            axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            plt.xticks(rotation=45)
            plt.show()
            """
        ),
        md_cell(
            """
            ### Signal-to-Noise Ratio Plot

            This 2-panel figure plots area-based and displacement-based signal-to-noise ratios using the stable-terrain uncertainty references.
            """
        ),
        code_cell(
            """
            snr_df = summary_df.copy()
            snr_df["change_area_snr"] = np.where(
                snr_df["stable_false_change_area_m2"] > 0,
                snr_df["detected_change_area_m2"] / snr_df["stable_false_change_area_m2"],
                np.nan,
            )
            snr_df["displacement_p95_snr"] = np.where(
                snr_df["stable_flow_p95_m"] > 0,
                snr_df["change_zone_p95_disp_m"] / snr_df["stable_flow_p95_m"],
                np.nan,
            )

            fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True, constrained_layout=True)

            for area_name, area_df in snr_df.groupby("area_name"):
                axes[0].plot(
                    area_df["curr_date"],
                    area_df["change_area_snr"],
                    "-o",
                    linewidth=2,
                    markersize=6,
                    color=area_colors.get(area_name),
                    label=area_name,
                )
                axes[1].plot(
                    area_df["curr_date"],
                    area_df["displacement_p95_snr"],
                    "-s",
                    linewidth=2,
                    markersize=6,
                    color=area_colors.get(area_name),
                    label=area_name,
                )

            axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.7)
            axes[0].set_title("Signal-to-noise ratio: detected change area / stable false-change area")
            axes[0].set_ylabel("Area SNR")
            axes[0].legend(title="Area")

            axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.7)
            axes[1].set_title("Signal-to-noise ratio: change-zone p95 displacement / stable-flow p95")
            axes[1].set_ylabel("Displacement SNR")
            axes[1].set_xlabel("Observation date")
            axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            plt.xticks(rotation=45)
            plt.show()
            """
        ),
        md_cell(
            """
            ### Mean Movement Direction Time Series

            This plot shows the average displacement direction through time for each analysis area.
            """
        ),
        code_cell(
            """
            fig, ax = plt.subplots(figsize=(16, 5), constrained_layout=True)

            for area_name, area_df in summary_df.groupby("area_name"):
                valid_dir = area_df.dropna(subset=["change_zone_mean_direction_deg"])
                ax.plot(
                    valid_dir["curr_date"],
                    valid_dir["change_zone_mean_direction_deg"],
                    "-o",
                    linewidth=2,
                    color=area_colors.get(area_name),
                    label=area_name,
                )

            ax.set_title("Movement direction through time by analysis area")
            ax.set_ylabel("Direction [degrees]")
            ax.set_xlabel("Observation date")
            ax.set_ylim(0, 360)
            ax.set_yticks([0, 90, 180, 270, 360])
            ax.set_yticklabels(["E (0°)", "N (90°)", "W (180°)", "S (270°)", "E (360°)"])
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax.legend(title="Area")
            plt.xticks(rotation=45)
            plt.show()
            """
        ),
        md_cell(
            """
            ### Direction Rose Diagram Plot

            This rose diagram compares the distribution of displacement directions accumulated across all pairs for each analysis area.
            """
        ),
        code_cell(
            """
            fig, axes = plt.subplots(
                1,
                len(analysis_areas),
                figsize=(7 * len(analysis_areas), 6),
                subplot_kw={"projection": "polar"},
                constrained_layout=True,
            )
            axes = np.atleast_1d(axes)
            bins_deg = np.linspace(0, 360, 17)
            bins_rad = np.deg2rad(bins_deg)
            widths = np.diff(bins_rad)

            for ax, area in zip(axes, analysis_areas):
                direction_samples = []

                for prev_path, curr_path in zip(image_paths[:-1], image_paths[1:]):
                    prev_date = extract_date(prev_path)
                    curr_date = extract_date(curr_path)
                    (
                        _,
                        _,
                        valid_local,
                        _,
                        change_arr_raw,
                        flow_arr_raw,
                        _,
                        _,
                        reliable_motion_local,
                        _,
                    ) = compute_pair_products(prev_date, curr_date)
                    effective_mask = area["mask"] & valid_local & (change_arr_raw > 0) & reliable_motion_local

                    mag_local = np.linalg.norm(flow_arr_raw, axis=-1)
                    dir_local = (np.degrees(np.arctan2(flow_arr_raw[..., 1], flow_arr_raw[..., 0])) + 360) % 360
                    direction_samples.append(dir_local[effective_mask & (mag_local > 0.25)])

                direction_samples = np.concatenate([arr for arr in direction_samples if arr.size], axis=0) if any(arr.size for arr in direction_samples) else np.array([])
                counts, _ = np.histogram(direction_samples, bins=bins_deg)

                ax.bar(
                    bins_rad[:-1],
                    counts,
                    width=widths,
                    bottom=0.0,
                    align="edge",
                    color=area_colors.get(area["name"]),
                    alpha=0.8,
                    edgecolor="white",
                    linewidth=1.0,
                )
                ax.set_theta_zero_location("E")
                ax.set_theta_direction(1)
                ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
                ax.set_xticklabels(["E", "N", "W", "S"])
                ax.set_title(f"Direction rose: {area['name']}")
                ax.set_ylabel("Count")
                ax.grid(True, alpha=0.3)

            plt.show()
            """
        ),
        md_cell(
            """
            ### Direction Rose Diagrams by Date Pair

            This subplot grid shows one rose diagram per consecutive date pair and per analysis area, so directional patterns can be compared through time.
            """
        ),
        code_cell(
            """
            pair_sequence_local = [
                (extract_date(prev_path), extract_date(curr_path))
                for prev_path, curr_path in zip(image_paths[:-1], image_paths[1:])
            ]

            n_rows = len(analysis_areas)
            n_cols = len(pair_sequence_local)
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(4.8 * n_cols, 5.5 * n_rows),
                subplot_kw={"projection": "polar"},
                constrained_layout=True,
            )
            axes = np.atleast_2d(axes)
            bins_deg = np.linspace(0, 360, 17)
            bins_rad = np.deg2rad(bins_deg)
            widths = np.diff(bins_rad)

            for col_idx, (prev_date, curr_date) in enumerate(pair_sequence_local):
                (
                    _,
                    _,
                    valid_local,
                    _,
                    change_arr_raw,
                    flow_arr_raw,
                    _,
                    _,
                    reliable_motion_local,
                    _,
                ) = compute_pair_products(prev_date, curr_date)

                mag_local = np.linalg.norm(flow_arr_raw, axis=-1)
                dir_local = (np.degrees(np.arctan2(flow_arr_raw[..., 1], flow_arr_raw[..., 0])) + 360) % 360

                for row_idx, area in enumerate(analysis_areas):
                    ax = axes[row_idx, col_idx]
                    effective_mask = area["mask"] & valid_local & (change_arr_raw > 0) & reliable_motion_local
                    direction_samples = dir_local[effective_mask & (mag_local > 0.25)]
                    counts, _ = np.histogram(direction_samples, bins=bins_deg)

                    ax.bar(
                        bins_rad[:-1],
                        counts,
                        width=widths,
                        bottom=0.0,
                        align="edge",
                        color=area_colors.get(area["name"]),
                        alpha=0.8,
                        edgecolor="white",
                        linewidth=1.0,
                    )
                    ax.set_theta_zero_location("E")
                    ax.set_theta_direction(1)
                    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
                    ax.set_xticklabels(["E", "N", "W", "S"])
                    if row_idx == 0:
                        ax.set_title(f"{prev_date} -> {curr_date}")
                    if col_idx == 0:
                        ax.text(
                            -0.18,
                            0.5,
                            area["name"],
                            transform=ax.transAxes,
                            rotation=90,
                            va="center",
                            ha="center",
                            fontsize=10,
                        )
                    ax.grid(True, alpha=0.3)

            plt.show()
            """
        ),
        md_cell(
            """
            ## 5. Agreement Between Change Detection and Displacement

            This section builds the comparison requested for section 6.1. For each consecutive orthophoto pair and for each analysis area (`flux` and `flank`), the notebook prepares these base layers first:

            - `change_mask`
            - `mag`
            - `uncertainty`
            - `stable_mask`
            - `valid_mask`
            - `reliable_motion_mask`

            Then a pair-specific displacement threshold is estimated from the stable terrain:

            - `stable_mag = mag[(stable_mask) & (valid_mask)]`
            - `stable_p95 = 95th percentile of stable_mag`
            - `displacement_mask = reliable_motion_mask & (mag > stable_p95)`

            The resulting binary comparison uses:

            - `C`: change mask inside each analysis area
            - `D`: displacement mask inside each analysis area
            """
        ),
        code_cell(
            """
            MIN_RELIABLE_MOTION_PX = 0.25


            def classify_area_name(name):
                normalized = str(name).strip().lower()
                if "flux" in normalized:
                    return "flux"
                if "flank" in normalized:
                    return "flank"
                return normalized


            analysis_area_lookup = {classify_area_name(area["name"]): area for area in analysis_areas}
            required_area_names = ["flux", "flank"]
            missing_area_names = [name for name in required_area_names if name not in analysis_area_lookup]
            if missing_area_names:
                raise ValueError(f"Missing expected analysis areas: {missing_area_names}")


            pair_sequence = [(extract_date(prev_path), extract_date(curr_path)) for prev_path, curr_path in zip(image_paths[:-1], image_paths[1:])]


            def compute_pair_products(prev_date, curr_date):
                img_prev, img_curr = load_pair_images(prev_date, curr_date)
                valid_local = get_common_mask(img_prev, img_curr) > 0
                stable_bool_local = get_stable_bool(valid_local.shape)
                corrected = match_histograms(img_curr, img_prev)
                change_raw = detect_landslide_changes(
                    img_prev,
                    corrected,
                    blur_k=config.BLUR_KERNEL_SIZE,
                    threshold=CHANGE_THRESHOLD,
                    min_area=config.MIN_LANDSLIDE_AREA,
                )
                change_mask_local = (change_raw > 0) & valid_local & (~stable_bool_local)

                flow_raw_local = compute_dense_displacement(img_prev, corrected, mask=None)
                backward_flow_local = compute_dense_displacement(corrected, img_prev, mask=None)
                h, w = flow_raw_local.shape[:2]
                grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
                mapped_x = grid_x + flow_raw_local[..., 0]
                mapped_y = grid_y + flow_raw_local[..., 1]
                backward_at_forward = np.dstack(
                    [
                        cv2.remap(backward_flow_local[..., 0], mapped_x, mapped_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan),
                        cv2.remap(backward_flow_local[..., 1], mapped_x, mapped_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan),
                    ]
                )
                flow_uncertainty_local = np.linalg.norm(flow_raw_local + backward_at_forward, axis=-1)
                flow_uncertainty_local[~valid_local] = np.nan

                mag_px_local = np.linalg.norm(flow_raw_local, axis=-1)
                mag_m_local = mag_px_local * pixel_size_m
                stable_mag = mag_m_local[stable_bool_local & valid_local]
                stable_p95_m = float(np.nanpercentile(stable_mag, 95)) if np.isfinite(stable_mag).any() else np.nan

                stable_uncertainty_local = flow_uncertainty_local[stable_bool_local & valid_local]
                uncertainty_threshold_local = float(np.nanpercentile(stable_uncertainty_local, 95)) if np.isfinite(stable_uncertainty_local).any() else 1.0
                low_uncertainty_local = valid_local & np.isfinite(flow_uncertainty_local) & (flow_uncertainty_local <= uncertainty_threshold_local)
                reliable_motion_local = low_uncertainty_local & (~stable_bool_local) & (mag_px_local > MIN_RELIABLE_MOTION_PX)
                displacement_local = reliable_motion_local & np.isfinite(mag_m_local) & (mag_m_local > stable_p95_m)

                return {
                    "prev_date": prev_date,
                    "curr_date": curr_date,
                    "label": f"{prev_date} -> {curr_date}",
                    "img_prev": img_prev,
                    "img_curr": img_curr,
                    "img_prev_rgb": cv2.cvtColor(img_prev, cv2.COLOR_BGR2RGB),
                    "valid_mask": valid_local,
                    "stable_mask": stable_bool_local,
                    "change_mask": change_mask_local,
                    "flow": flow_raw_local,
                    "mag_m": mag_m_local,
                    "uncertainty": flow_uncertainty_local,
                    "reliable_motion_mask": reliable_motion_local,
                    "stable_p95_m": stable_p95_m,
                    "uncertainty_threshold_px": uncertainty_threshold_local,
                    "displacement_mask": displacement_local,
                }


            pair_products_cache = {}


            def get_pair_products(prev_date, curr_date):
                key = (prev_date, curr_date)
                if key not in pair_products_cache:
                    pair_products_cache[key] = compute_pair_products(prev_date, curr_date)
                return pair_products_cache[key]


            base_layer_records = []
            agreement_records = []
            agreement_class_records = []
            agreement_maps = {}

            for prev_date, curr_date in pair_sequence:
                pair_data = get_pair_products(prev_date, curr_date)
                base_layer_records.append(
                    {
                        "pair_label": pair_data["label"],
                        "prev_date": parse_date(prev_date),
                        "curr_date": parse_date(curr_date),
                        "valid_pixels": int(pair_data["valid_mask"].sum()),
                        "stable_pixels": int((pair_data["stable_mask"] & pair_data["valid_mask"]).sum()),
                        "change_pixels": int(pair_data["change_mask"].sum()),
                        "reliable_motion_pixels": int(pair_data["reliable_motion_mask"].sum()),
                        "displacement_pixels": int(pair_data["displacement_mask"].sum()),
                        "stable_p95_m": pair_data["stable_p95_m"],
                        "uncertainty_threshold_px": pair_data["uncertainty_threshold_px"],
                    }
                )

                for area_name in required_area_names:
                    area = analysis_area_lookup[area_name]
                    area_mask = area["mask"] & pair_data["valid_mask"] & (~pair_data["stable_mask"])
                    C = pair_data["change_mask"] & area_mask
                    D = pair_data["displacement_mask"] & area_mask

                    both = C & D
                    change_only = C & (~D)
                    displacement_only = (~C) & D & area_mask
                    neither = (~C) & (~D) & area_mask

                    class_masks = {
                        "changed_and_displacement": both,
                        "changed_not_displacement": change_only,
                        "unchanged_displacement": displacement_only,
                        "neither": neither,
                    }
                    class_labels = {
                        "changed_and_displacement": "C=1, D=1",
                        "changed_not_displacement": "C=1, D=0",
                        "unchanged_displacement": "C=0, D=1",
                        "neither": "C=0, D=0",
                    }

                    area_pixel_count = int(area_mask.sum())
                    change_pixels = int(C.sum())
                    displacement_pixels = int(D.sum())
                    intersection_pixels = int(both.sum())
                    union_pixels = int((C | D).sum())

                    p_d_given_c = intersection_pixels / change_pixels if change_pixels else np.nan
                    p_c_given_d = intersection_pixels / displacement_pixels if displacement_pixels else np.nan
                    jaccard = intersection_pixels / union_pixels if union_pixels else np.nan

                    mag_in_change = pair_data["mag_m"][C]
                    mag_in_change_and_displacement = pair_data["mag_m"][both]
                    mag_in_unchanged = pair_data["mag_m"][area_mask & (~C)]

                    agreement_records.append(
                        {
                            "area_name": area_name,
                            "prev_date": parse_date(prev_date),
                            "curr_date": parse_date(curr_date),
                            "pair_label": pair_data["label"],
                            "area_pixels": area_pixel_count,
                            "area_m2": area_pixel_count * pixel_area_m2,
                            "change_pixels": change_pixels,
                            "change_area_m2": change_pixels * pixel_area_m2,
                            "displacement_pixels": displacement_pixels,
                            "displacement_area_m2": displacement_pixels * pixel_area_m2,
                            "intersection_pixels": intersection_pixels,
                            "intersection_area_m2": intersection_pixels * pixel_area_m2,
                            "union_pixels": union_pixels,
                            "stable_p95_m": pair_data["stable_p95_m"],
                            "P_D_given_C": p_d_given_c,
                            "P_C_given_D": p_c_given_d,
                            "jaccard": jaccard,
                            "change_mag_mean_m": float(np.mean(mag_in_change)) if mag_in_change.size else np.nan,
                            "change_mag_median_m": float(np.median(mag_in_change)) if mag_in_change.size else np.nan,
                            "change_mag_p95_m": float(np.percentile(mag_in_change, 95)) if mag_in_change.size else np.nan,
                            "change_displacement_mag_mean_m": float(np.mean(mag_in_change_and_displacement)) if mag_in_change_and_displacement.size else np.nan,
                            "change_displacement_mag_median_m": float(np.median(mag_in_change_and_displacement)) if mag_in_change_and_displacement.size else np.nan,
                            "change_displacement_mag_p95_m": float(np.percentile(mag_in_change_and_displacement, 95)) if mag_in_change_and_displacement.size else np.nan,
                            "unchanged_mag_sample_count": int(mag_in_unchanged.size),
                        }
                    )

                    for class_key, class_mask in class_masks.items():
                        pixel_count = int(class_mask.sum())
                        agreement_class_records.append(
                            {
                                "area_name": area_name,
                                "prev_date": parse_date(prev_date),
                                "curr_date": parse_date(curr_date),
                                "pair_label": pair_data["label"],
                                "class_key": class_key,
                                "class_label": class_labels[class_key],
                                "pixels": pixel_count,
                                "area_m2": pixel_count * pixel_area_m2,
                                "area_pct": (pixel_count / area_pixel_count * 100) if area_pixel_count else np.nan,
                            }
                        )

                    agreement_map = np.full(pair_data["change_mask"].shape, -1, dtype=np.int8)
                    agreement_map[neither] = 0
                    agreement_map[displacement_only] = 1
                    agreement_map[change_only] = 2
                    agreement_map[both] = 3
                    agreement_maps[(pair_data["label"], area_name)] = agreement_map


            base_layers_df = pd.DataFrame(base_layer_records).sort_values("curr_date").reset_index(drop=True)
            agreement_df = pd.DataFrame(agreement_records).sort_values(["area_name", "curr_date"]).reset_index(drop=True)
            agreement_classes_df = pd.DataFrame(agreement_class_records).sort_values(["area_name", "curr_date", "class_key"]).reset_index(drop=True)

            display(
                base_layers_df.style.format(
                    {
                        "stable_p95_m": "{:.3f}",
                        "uncertainty_threshold_px": "{:.3f}",
                    }
                )
            )

            display(
                agreement_df[
                    [
                        "area_name",
                        "pair_label",
                        "area_m2",
                        "change_area_m2",
                        "displacement_area_m2",
                        "intersection_area_m2",
                        "stable_p95_m",
                        "P_D_given_C",
                        "P_C_given_D",
                        "jaccard",
                        "change_mag_mean_m",
                        "change_mag_median_m",
                        "change_mag_p95_m",
                        "change_displacement_mag_mean_m",
                        "change_displacement_mag_median_m",
                        "change_displacement_mag_p95_m",
                    ]
                ].style.format(
                    {
                        "area_m2": "{:,.1f}",
                        "change_area_m2": "{:,.1f}",
                        "displacement_area_m2": "{:,.1f}",
                        "intersection_area_m2": "{:,.1f}",
                        "stable_p95_m": "{:.3f}",
                        "P_D_given_C": "{:.3f}",
                        "P_C_given_D": "{:.3f}",
                        "jaccard": "{:.3f}",
                        "change_mag_mean_m": "{:.3f}",
                        "change_mag_median_m": "{:.3f}",
                        "change_mag_p95_m": "{:.3f}",
                        "change_displacement_mag_mean_m": "{:.3f}",
                        "change_displacement_mag_median_m": "{:.3f}",
                        "change_displacement_mag_p95_m": "{:.3f}",
                    }
                )
            )

            display(
                agreement_classes_df.style.format(
                    {
                        "area_m2": "{:,.1f}",
                        "area_pct": "{:.2f}",
                    }
                )
            )
            """
        ),
        md_cell(
            """
            ### Representative Agreement Map

            This map shows one representative pair, highlighting where change detection and displacement agree or disagree inside the flux and flank areas.
            """
        ),
        code_cell(
            """
            representative_pair_scores = (
                agreement_df.groupby("pair_label", as_index=False)["jaccard"]
                .mean()
                .sort_values(["jaccard", "pair_label"], ascending=[False, True])
            )
            representative_pair_label = representative_pair_scores.iloc[0]["pair_label"]
            representative_pair = next(item for item in pair_sequence if f"{item[0]} -> {item[1]}" == representative_pair_label)
            representative_data = get_pair_products(*representative_pair)

            agreement_cmap = plt.matplotlib.colors.ListedColormap(["#bdbdbd", "#2b8cbe", "#fdae61", "#d7191c"])
            agreement_norm = plt.matplotlib.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], agreement_cmap.N)
            legend_labels = {
                3: "Changed + displacement",
                2: "Changed + not displacement",
                1: "Unchanged + displacement",
                0: "Neither",
            }

            fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
            for ax, area_name in zip(axes, required_area_names):
                agreement_map = agreement_maps[(representative_pair_label, area_name)]
                masked_map = np.ma.masked_where(agreement_map < 0, agreement_map)
                ax.imshow(representative_data["img_prev_rgb"])
                ax.imshow(masked_map, cmap=agreement_cmap, norm=agreement_norm, alpha=0.72)
                ax.set_title(f"{area_name.capitalize()}: {representative_pair_label}")
                ax.axis("off")

            handles = [
                plt.matplotlib.patches.Patch(color=agreement_cmap(agreement_norm(code)), label=legend_labels[code])
                for code in [3, 2, 1, 0]
            ]
            fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
            fig.suptitle("Figure 1. Representative 4-class agreement map", fontsize=15)
            plt.show()
            """
        ),
        md_cell(
            """
            ### Agreement-Class Composition Plot

            This stacked-bar figure shows how each pair is split among the four agreement classes for the flux and flank areas.
            """
        ),
        code_cell(
            """
            class_order = [
                ("changed_and_displacement", "Changed + displacement", "#d7191c"),
                ("changed_not_displacement", "Changed + not displacement", "#fdae61"),
                ("unchanged_displacement", "Unchanged + displacement", "#2b8cbe"),
                ("neither", "Neither", "#bdbdbd"),
            ]

            fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True, constrained_layout=True)

            for ax, area_name in zip(axes, required_area_names):
                area_classes = agreement_classes_df[agreement_classes_df["area_name"] == area_name].copy()
                pivot = area_classes.pivot(index="pair_label", columns="class_key", values="area_pct").fillna(0.0)
                pair_order = agreement_df[agreement_df["area_name"] == area_name]["pair_label"].tolist()
                pivot = pivot.reindex(pair_order)
                bottom = np.zeros(len(pivot), dtype=float)

                for class_key, class_label, class_color in class_order:
                    values = pivot[class_key].to_numpy() if class_key in pivot.columns else np.zeros(len(pivot), dtype=float)
                    ax.bar(pivot.index, values, bottom=bottom, color=class_color, edgecolor="white", linewidth=0.5, label=class_label)
                    bottom += values

                ax.set_ylim(0, 100)
                ax.set_ylabel("Area [%]")
                ax.set_title(f"{area_name.capitalize()}")
                ax.tick_params(axis="x", rotation=45)

            axes[0].legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.22))
            axes[1].set_xlabel("Image pair")
            fig.suptitle("Figure 2. Agreement-class composition by pair", fontsize=15)
            plt.show()
            """
        ),
        md_cell(
            """
            ### P(D|C) Time-Series Plot

            This plot tracks the fraction of changed pixels that also exceed the displacement threshold through time.
            """
        ),
        code_cell(
            """
            fig, ax = plt.subplots(figsize=(16, 5), constrained_layout=True)

            for area_name in required_area_names:
                area_df = agreement_df[agreement_df["area_name"] == area_name]
                ax.plot(area_df["curr_date"], area_df["P_D_given_C"], "-o", linewidth=2.2, markersize=6, label=area_name.capitalize())

            ax.set_title("Figure 3. Time series of P(D|C)")
            ax.set_ylabel("P(D|C)")
            ax.set_xlabel("Observation date")
            ax.set_ylim(0, 1.05)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax.legend()
            plt.xticks(rotation=45)
            plt.show()
            """
        ),
        md_cell(
            """
            ### Jaccard Overlap Time-Series Plot

            This plot shows how strongly the change mask and displacement mask overlap through time using the Jaccard index.
            """
        ),
        code_cell(
            """
            fig, ax = plt.subplots(figsize=(16, 5), constrained_layout=True)

            for area_name in required_area_names:
                area_df = agreement_df[agreement_df["area_name"] == area_name]
                ax.plot(area_df["curr_date"], area_df["jaccard"], "-o", linewidth=2.2, markersize=6, label=area_name.capitalize())

            ax.set_title("Figure 4. Time series of Jaccard overlap")
            ax.set_ylabel("Jaccard")
            ax.set_xlabel("Observation date")
            ax.set_ylim(0, 1.05)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax.legend()
            plt.xticks(rotation=45)
            plt.show()
            """
        ),
        md_cell(
            """
            ### Displacement Distribution Plot

            This violin-plot comparison contrasts displacement magnitudes for changed and unchanged pixels, with the stable-terrain threshold shown as a reference line.
            """
        ),
        code_cell(
            """
            distribution_records = []

            for prev_date, curr_date in pair_sequence:
                pair_data = get_pair_products(prev_date, curr_date)
                for area_name in required_area_names:
                    area_mask = analysis_area_lookup[area_name]["mask"] & pair_data["valid_mask"] & (~pair_data["stable_mask"])
                    change_mask = pair_data["change_mask"] & area_mask
                    unchanged_mask = area_mask & (~pair_data["change_mask"])

                    for group_name, sample_mask in [("Changed pixels", change_mask), ("Unchanged pixels", unchanged_mask)]:
                        samples = pair_data["mag_m"][sample_mask]
                        if samples.size == 0:
                            continue
                        distribution_records.append(
                            pd.DataFrame(
                                {
                                    "pair_label": pair_data["label"],
                                    "area_name": area_name,
                                    "group": group_name,
                                    "magnitude_m": samples,
                                    "stable_p95_m": pair_data["stable_p95_m"],
                                }
                            )
                        )

            distribution_df = pd.concat(distribution_records, ignore_index=True)
            fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True, constrained_layout=True)

            for ax, area_name in zip(axes, required_area_names):
                area_dist = distribution_df[distribution_df["area_name"] == area_name]
                changed = area_dist[area_dist["group"] == "Changed pixels"]["magnitude_m"].to_numpy()
                unchanged = area_dist[area_dist["group"] == "Unchanged pixels"]["magnitude_m"].to_numpy()
                plot_data = [changed, unchanged]
                violin = ax.violinplot(plot_data, positions=[1, 2], showmeans=True, showmedians=True, widths=0.8)
                colors = ["#d7191c", "#2b8cbe"]
                for body, color in zip(violin["bodies"], colors):
                    body.set_facecolor(color)
                    body.set_edgecolor("black")
                    body.set_alpha(0.5)
                for key in ["cbars", "cmins", "cmaxes", "cmeans", "cmedians"]:
                    if key in violin:
                        violin[key].set_color("black")
                        violin[key].set_linewidth(1.0)

                stable_ref = area_dist["stable_p95_m"].median()
                ax.axhline(stable_ref, color="black", linestyle="--", linewidth=1.5, label=f"Median stable_p95 = {stable_ref:.3f} m")
                ax.set_xticks([1, 2])
                ax.set_xticklabels(["Changed", "Unchanged"])
                ax.set_title(area_name.capitalize())
                ax.set_ylabel("Displacement magnitude [m]")
                ax.legend(loc="upper right")

            fig.suptitle("Figure 5. Displacement magnitude distributions with stable-terrain threshold", fontsize=15)
            plt.show()
            """
        ),
        md_cell(
            """
            ## 6. Spatial Aggregation Through Time by Analysis Area

            Time-series plots show *when* activity increases. The maps below show *where* activity repeatedly concentrates inside each analysis polygon:

            - **change recurrence**: fraction of pairwise comparisons in which a pixel was marked as changed,
            - **mean displacement magnitude**: average dense-field magnitude across all pairs,
            - **displacement variability**: temporal standard deviation of the dense-field magnitude.
            """
        ),
        md_cell(
            """
            ### Spatial Aggregation Maps

            These maps show where change is repeatedly detected and where displacement magnitude is strongest or most variable within each analysis area.
            """
        ),
        code_cell(
            """
            spatial_results = {}
            comparison_records = []

            for area in analysis_areas:
                area_mask = area["mask"]
                pair_count = 0
                recurrence_sum = None
                mag_sum = None
                mag_sq_sum = None

                for prev_date, curr_date in pair_sequence:
                    pair_data = get_pair_products(prev_date, curr_date)
                    effective_mask = area_mask & pair_data["valid_mask"] & (~pair_data["stable_mask"]) & pair_data["reliable_motion_mask"]

                    change_arr = np.zeros_like(pair_data["change_mask"], dtype=np.uint8)
                    change_arr[effective_mask] = pair_data["change_mask"][effective_mask].astype(np.uint8)

                    flow_arr = np.zeros_like(pair_data["flow"])
                    flow_arr[effective_mask] = pair_data["flow"][effective_mask]
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

                spatial_results[area["name"]] = {
                    "mask": area_mask,
                    "change_recurrence": np.ma.masked_where(~area_mask, change_recurrence),
                    "mean_mag": np.ma.masked_where(~area_mask, mean_mag),
                    "std_mag": np.ma.masked_where(~area_mask, std_mag),
                }

                comparison_records.append(
                    {
                        "area_name": area["name"],
                        "max_change_recurrence": float(change_recurrence[area_mask].max()),
                        "mean_change_recurrence": float(change_recurrence[area_mask].mean()),
                        "mean_displacement_m": float(mean_mag[area_mask].mean()),
                        "max_displacement_m": float(mean_mag[area_mask].max()),
                        "mean_variability_m": float(std_mag[area_mask].mean()),
                    }
                )

            comparison_df = pd.DataFrame(comparison_records).sort_values("area_name").reset_index(drop=True)
            display(
                comparison_df.style.format(
                    {
                        "max_change_recurrence": "{:.3f}",
                        "mean_change_recurrence": "{:.3f}",
                        "mean_displacement_m": "{:.3f}",
                        "max_displacement_m": "{:.3f}",
                        "mean_variability_m": "{:.3f}",
                    }
                )
            )

            fig, axes = plt.subplots(len(analysis_areas), 3, figsize=(18, 6 * len(analysis_areas)), constrained_layout=True)
            if len(analysis_areas) == 1:
                axes = np.array([axes])

            for row_idx, area in enumerate(analysis_areas):
                row_axes = axes[row_idx]
                result = spatial_results[area["name"]]

                row_axes[0].imshow(rgb_ref)
                im0 = row_axes[0].imshow(result["change_recurrence"], cmap="Reds", alpha=0.75, vmin=0, vmax=1)
                row_axes[0].set_title(f"{area['name']}: change recurrence")
                row_axes[0].axis("off")
                fig.colorbar(
                    im0,
                    ax=row_axes[0],
                    fraction=0.046,
                    pad=0.08,
                    orientation="horizontal",
                    label="Fraction of pairs",
                )

                row_axes[1].imshow(rgb_ref)
                im1 = row_axes[1].imshow(result["mean_mag"], cmap="viridis", alpha=0.75)
                row_axes[1].set_title(f"{area['name']}: mean dense-field magnitude")
                row_axes[1].axis("off")
                fig.colorbar(
                    im1,
                    ax=row_axes[1],
                    fraction=0.046,
                    pad=0.08,
                    orientation="horizontal",
                    label="Mean displacement [m]",
                )

                row_axes[2].imshow(rgb_ref)
                im2 = row_axes[2].imshow(result["std_mag"], cmap="magma", alpha=0.75)
                row_axes[2].set_title(f"{area['name']}: temporal variability")
                row_axes[2].axis("off")
                fig.colorbar(
                    im2,
                    ax=row_axes[2],
                    fraction=0.046,
                    pad=0.08,
                    orientation="horizontal",
                    label="Std. dev. [m]",
                )

            plt.show()
            """
        ),
        md_cell(
            """
            ## 7. Interactive Displacement Browser

            The slider below lets you browse the consecutive image pairs and inspect the dense-field displacement magnitude over the image context.
            """
        ),
        md_cell(
            """
            ### Interactive Displacement Overlay

            This interactive view overlays displacement magnitude on the reference image so you can step through all consecutive pairs.
            """
        ),
        code_cell(
            """
            overlay_cache = {}
            pair_labels = [f"{prev_date} -> {curr_date}" for prev_date, curr_date in pair_sequence]


            def get_overlay_data(pair_label):
                pair_idx = pair_labels.index(pair_label)
                if pair_label not in overlay_cache:
                    prev_date, curr_date = pair_sequence[pair_idx]
                    pair_data = get_pair_products(prev_date, curr_date)
                    flow_for_display = pair_data["flow"].copy()
                    flow_for_display[pair_data["stable_mask"]] = 0
                    flow_for_display[~pair_data["reliable_motion_mask"]] = 0
                    flow_for_display[~pair_data["valid_mask"]] = 0
                    mag_arr = np.linalg.norm(flow_for_display, axis=-1) * pixel_size_m
                    overlay_cache[pair_label] = {
                        "label": f"{prev_date} -> {curr_date}",
                        "base_rgb": pair_data["img_prev_rgb"],
                        "magnitude": mag_arr,
                    }
                return overlay_cache[pair_label]


            def show_displacement_overlay(pair_label=pair_labels[0]):
                item = get_overlay_data(pair_label)
                mag_arr = item["magnitude"]
                base_rgb = item["base_rgb"]
                alpha_map = np.clip(mag_arr / max(np.percentile(mag_arr[mag_arr > 0], 99), 1e-9), 0, 1) if np.any(mag_arr > 0) else np.zeros_like(mag_arr)

                fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
                ax.imshow(base_rgb)
                im = ax.imshow(mag_arr, cmap="viridis", alpha=0.75 * alpha_map)
                ax.set_title(f"Displacement magnitude overlay: {item['label']}")
                ax.axis("off")
                cbar = fig.colorbar(
                    im,
                    ax=ax,
                    fraction=0.046,
                    pad=0.08,
                    orientation="horizontal",
                )
                cbar.set_label("Displacement [m]")
                plt.show()


            if widgets is None:
                print("ipywidgets is not installed in this environment, so the interactive slider is unavailable.")
            else:
                display(
                    widgets.interactive(
                        show_displacement_overlay,
                        pair_label=widgets.SelectionSlider(
                            options=pair_labels,
                            value=pair_labels[0],
                            description="Dates",
                            continuous_update=False,
                            layout=widgets.Layout(width="95%"),
                            style={"description_width": "initial"},
                        ),
                    )
                )
            """
        ),
        md_cell(
            """
            ## Interpretation Notes

            A few reading rules can help when you discuss the area-based results:

            - If the **change area** in one polygon is close to its stable-area false positive area, the mapped change in that polygon should be treated cautiously.
            - If the **mean displacement** in one polygon is only as large as the stable-terrain RMSE, the motion signal in that polygon is weak.
            - Areas with both **high change recurrence** and **high mean displacement** are the most likely to represent persistent activity.

            This area-by-area structure also makes it straightforward to add clustering or to export one summary table per polygon later on.
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
