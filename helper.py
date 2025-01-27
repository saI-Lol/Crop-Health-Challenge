import numpy as np
import rasterio

def process_row_for_features(index, row):
    features = {'index': index}
    tif_path = row['tif_path']
    if not isinstance(tif_path, str):
        # print(f"Skipping entry due to missing tif_path for index {index}")
        features.update({
            'ndvi': np.nan,
            'evi': np.nan,
            'ndwi': np.nan,
            'gndvi': np.nan,
            'savi': np.nan,
            'msavi': np.nan,
            'red_mean': np.nan,
            'green_mean': np.nan,
            'blue_mean': np.nan,
            'nir_mean': np.nan
        })
        return features

    with rasterio.open(tif_path) as src:
        red = src.read(3)  
        green = src.read(2) 
        blue = src.read(1)
        nir = src.read(4)

        features['ndvi'] = calculate_ndvi(nir, red)
        features['evi'] = calculate_evi(nir, red, blue)
        features['ndwi'] = calculate_ndwi(nir, green)
        features['gndvi'] = calculate_gndvi(nir, green)
        features['savi'] = calculate_savi(nir, red)
        features['msavi'] = calculate_msavi(nir, red)
        features['red_mean'] = np.nanmedian(red)
        features['green_mean'] = np.nanmedian(green)
        features['blue_mean'] = np.nanmedian(blue)
        features['nir_mean'] = np.nanmedian(nir)

    return features


def calculate_ndvi(nir_band, red_band, epsilon=1e-10):
    """Calculate NDVI (Normalized Difference Vegetation Index)."""
    ndvi = (nir_band - red_band) / (nir_band + red_band + epsilon)
    return np.nanmean(ndvi)

def calculate_evi(nir_band, red_band, blue_band, epsilon=1e-10):
    """Calculate EVI (Enhanced Vegetation Index)."""
    evi = 2.5 * (nir_band - red_band) / (nir_band + 6 * red_band - 7.5 * blue_band + 1 + epsilon)
    return np.nanmean(evi)

def calculate_ndwi(nir_band, green_band, epsilon=1e-10):
    """Calculate NDWI (Normalized Difference Water Index)."""
    ndwi = (green_band - nir_band) / (green_band + nir_band + epsilon)
    return np.nanmean(ndwi)

def calculate_gndvi(nir_band, green_band, epsilon=1e-10):
    """Calculate GNDVI (Green Normalized Difference Vegetation Index)."""
    gndvi = (nir_band - green_band) / (nir_band + green_band + epsilon)
    return np.nanmean(gndvi)

def calculate_savi(nir_band, red_band, L=0.5, epsilon=1e-10):
    """Calculate SAVI (Soil Adjusted Vegetation Index)."""
    savi = ((nir_band - red_band) / (nir_band + red_band + L + epsilon)) * (1 + L)
    return np.nanmean(savi)

def calculate_msavi(nir_band, red_band, epsilon=1e-10):
    """Calculate MSAVI (Modified Soil Adjusted Vegetation Index)."""
    msavi = (2 * nir_band + 1 - np.sqrt((2 * nir_band + 1)**2 - 8 * (nir_band - red_band) + epsilon)) / 2
    return np.nanmean(msavi)