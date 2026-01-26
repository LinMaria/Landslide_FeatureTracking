import geopandas as gpd
import rasterio
from rasterio import features
import numpy as np
import os
import cv2

def rasterize_vector_mask(vector_path, reference_tif_path, output_path):
    """
    Converts a vector file (.shp, .geojson) into a binary raster mask (.tif)
    that perfectly matches the resolution and extent of the reference_tif
    """
    print(f"Rasterizing vector mask: {vector_path}")

    # 1. Open the Reference Image to get metadata (transform, CRS, dimensions)
    with rasterio.open(reference_tif_path) as src:
        meta = src.meta.copy()
        height, width = src.shape
        transform = src.transform
        crs = src.crs

    # 2. Load the Vector File
    gdf = gpd.read_file(vector_path)

    # 3. Reproject Vector to match the Image's Coordinate System
    # This is critical. If your SHP is in Lat/Lon and TIF is in UTM, this fixes it.
    if gdf.crs != crs:
        print(f"Reprojecting vector from {gdf.crs} to {crs}...")
        gdf = gdf.to_crs(crs)

    # 4. Rasterize
    # We burn the shapes into a grid of zeros.
    # Inside the shape = 255 (White/Stable), Outside = 0 (Black).
    shapes = ((geom, 255) for geom in gdf.geometry)
    
    mask = features.rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,      # Background value
        dtype='uint8'
    )

    # 5. Save the Mask
    # We update metadata to ensure the mask is also a valid GeoTIFF
    meta.update(count=1, dtype='uint8', nodata=0)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(mask, 1)

    print(f"Mask saved to: {output_path}")
    return output_path