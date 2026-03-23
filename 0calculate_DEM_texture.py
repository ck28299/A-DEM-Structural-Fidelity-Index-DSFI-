import os
import numpy as np
import rasterio
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from scipy.ndimage import sobel, gaussian_filter, generic_filter
from scipy.fft import fft2, fftshift
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
import richdem as rd



def get_resolution(dem_file):
    with rasterio.open(dem_file) as src:
        res_x, _ = src.res
    return res_x

def analyze_high_frequency(dem, resolution):
    window_size = max(3, int(6 / resolution))
    valid_dem = np.where(np.isnan(dem), 0, dem)
    low_freq = gaussian_filter(valid_dem, sigma=window_size / 2)
    high_freq = valid_dem - low_freq
    energy_high = np.nansum(high_freq ** 2)
    variation = np.nanstd(high_freq) / (np.nanmean(np.abs(high_freq)) + 1e-6)
    return energy_high, variation

def texture_complexity(dem, resolution, radius=None):
    if radius is None:
        radius = 6 / resolution
    dem_int = (dem * 10000).astype(np.int32)
    radius = max(1, int(round(radius)))
    lbp = local_binary_pattern(dem_int, P=8, R=radius, method='uniform')
    unique_patterns = len(np.unique(lbp))
    contrast = np.nanstd(lbp)
    return unique_patterns, contrast

def slope_distribution_analysis(dem_array):
    rd_dem = rd.rdarray(dem_array, no_data=np.nan)
    slope = rd.TerrainAttribute(rd_dem, attrib='slope_degrees')
    slope_np = np.array(slope)
    slope_valid = slope_np[~np.isnan(slope_np)]
    std_slope = np.nanstd(slope_np)
    skew_slope = skew(slope_valid.flatten())
    kurtosis_slope = kurtosis(slope_valid.flatten())
    return std_slope, skew_slope, kurtosis_slope

def edge_sharpness(dem):
    nan_mask = np.isnan(dem)
    valid_dem = np.where(nan_mask, 0, dem)
    edges = sobel(valid_dem)
    edges[nan_mask] = np.nan
    edge_threshold = 0.3 * np.nanmax(edges)
    valid_edge_mask = ~np.isnan(edges)
    mean_edge_sharpness = np.nanmean(edges)
    std_edge_sharpness = np.nanstd(edges)
    strong_edges_edge_sharpness = np.nansum(edges > edge_threshold) / np.sum(valid_edge_mask)
    return mean_edge_sharpness, std_edge_sharpness, strong_edges_edge_sharpness

def spectral_analysis(dem):
    valid_dem = np.where(np.isnan(dem), 0, dem)
    f = fftshift(fft2(valid_dem))
    magnitude_spectrum = np.abs(f)
    rows, cols = dem.shape
    crow, ccol = rows // 2, cols // 2
    low_band = (crow - rows // 10, crow + rows // 10, ccol - cols // 10, ccol + cols // 10)
    mid_band = (crow - rows // 4, crow + rows // 4, ccol - cols // 4, ccol + cols // 4)
    mask_low = np.zeros_like(dem)
    mask_low[low_band[0]:low_band[1], low_band[2]:low_band[3]] = 1
    mask_mid = np.zeros_like(dem)
    mask_mid[mid_band[0]:mid_band[1], mid_band[2]:mid_band[3]] = 1
    mask_mid -= mask_low
    mask_high = 1 - mask_mid - mask_low
    total_energy = np.sum(magnitude_spectrum)
    low_freq_spectral = np.sum(magnitude_spectrum * mask_low) / total_energy
    mid_freq_spectral = np.sum(magnitude_spectrum * mask_mid) / total_energy
    high_freq_spectral = np.sum(magnitude_spectrum * mask_high) / total_energy
    return low_freq_spectral, mid_freq_spectral, high_freq_spectral

def compute_global_metrics(dem, resolution, window_size=5):
    pixel_area = resolution ** 2

    def roughness_func(window):
        window = window.reshape((window_size, window_size))
        dzdx = np.gradient(window, axis=1)
        dzdy = np.gradient(window, axis=0)
        slope = np.sqrt(dzdx ** 2 + dzdy ** 2)
        surface_area = np.sum(np.sqrt(1 + slope ** 2)) * pixel_area
        flat_area = pixel_area * window_size * window_size
        return surface_area / flat_area

    def relief_func(window):
        return np.nanmax(window) - np.nanmin(window)

    roughness_array = generic_filter(dem, roughness_func, size=window_size, mode='nearest')
    relief_array = generic_filter(dem, relief_func, size=window_size, mode='nearest')
    return roughness_array, relief_array

def summarize_array(arr):
    arr_flat = arr.flatten()
    arr_valid = arr_flat[~np.isnan(arr_flat)]
    return {
        'mean': np.nanmean(arr_valid),
        'std': np.nanstd(arr_valid),
        'skew': skew(arr_valid),
        'kurtosis': kurtosis(arr_valid)
    }

def compute_glcm_features(dem, distances=[1], angles=[0]):
    dem_valid = np.copy(dem)
    dem_valid[np.isnan(dem_valid)] = 0
    dem_uint8 = np.clip((dem_valid - np.nanmin(dem_valid)) / (np.nanmax(dem_valid) - np.nanmin(dem_valid)) * 255, 0, 255).astype(np.uint8)
    glcm = graycomatrix(dem_uint8, distances=distances, angles=angles, symmetric=True, normed=True)
    props = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']
    results = {prop: graycoprops(glcm, prop).mean() for prop in props}
    return results

def extract_features(dem, resolution):
    features = {}

    energy_high, variation = analyze_high_frequency(dem, resolution)
    features['energy_high'] = energy_high
    features['variation'] = variation

    unique_patterns, contrast = texture_complexity(dem, resolution)
    features['unique_patterns'] = unique_patterns
    features['contrast'] = contrast

    std_slope, skew_slope, kurtosis_slope = slope_distribution_analysis(dem)
    features['std_slope'] = std_slope
    features['skew_slope'] = skew_slope
    features['kurtosis_slope'] = kurtosis_slope

    mean_edge_sharpness, std_edge_sharpness, strong_edges_edge_sharpness = edge_sharpness(dem)
    features['mean_edge_sharpness'] = mean_edge_sharpness
    features['std_edge_sharpness'] = std_edge_sharpness
    features['strong_edges_edge_sharpness'] = strong_edges_edge_sharpness

    low_freq_spectral, mid_freq_spectral, high_freq_spectral = spectral_analysis(dem)
    features['low_freq_spectral'] = low_freq_spectral
    features['mid_freq_spectral'] = mid_freq_spectral
    features['high_freq_spectral'] = high_freq_spectral

    global_roughness, global_relief = compute_global_metrics(dem, resolution)
    roughness_features = summarize_array(global_roughness)
    relief_features = summarize_array(global_relief)
    features.update({f'roughness_{k}': v for k, v in roughness_features.items()})
    features.update({f'relief_{k}': v for k, v in relief_features.items()})

    glcm_feats = compute_glcm_features(dem)
    features.update({f'glcm_{k}': v for k, v in glcm_feats.items()})

    return features

def load_features_from_folder(folder, label, geomorph_type):
    features = []
    labels = []
    for file in os.listdir(folder):
        if file.endswith('.tif'):
            path = os.path.join(folder, file)
            with rasterio.open(path) as src:
                dem = src.read(1)
            resolution = get_resolution(path)
            feats = extract_features(dem, resolution)
            feats['filename'] = file
            feats['geomorph'] = geomorph_type
            features.append(feats)
            labels.append(label)
    return features, labels


geomorph_data = [
    ('class1', r'real DEM file', r'pseudo high-resolution DEM file'),
    ('class2', r'', r''),
    ('class3', r'', r''),
    ('class4', r'', r''),
    ('class5', r'', r''),
    ('class6', r'', r''),
    ('class7', r'', r''),
    ('class8', r'', r''),

]

all_features = []
all_labels = []

for geomorph_type, real_folder, fake_folder in geomorph_data:
    real_X, real_y = load_features_from_folder(real_folder, label=1, geomorph_type=geomorph_type)
    fake_X, fake_y = load_features_from_folder(fake_folder, label=0, geomorph_type=geomorph_type)
    all_features += real_X + fake_X
    all_labels += real_y + fake_y



filtered_features = []
for feat in all_features:
    filtered = {k: v for k, v in feat.items() if not isinstance(v, np.ndarray)}
    filtered_features.append(filtered)

X_df = pd.DataFrame(filtered_features)
X_df['label'] = all_labels


non_feature_cols = ['filename', 'geomorph', 'label']
feature_cols = [col for col in X_df.columns if col not in non_feature_cols]
scaler = MinMaxScaler(feature_range=(1, 100))
X_df[feature_cols] = scaler.fit_transform(X_df[feature_cols])
X_df[feature_cols] = X_df[feature_cols].clip(lower=1, upper=100)


output_path = 'multi_geomorph_features.csv'
os.makedirs('output', exist_ok=True)
X_df.to_csv(output_path, index=False)
print(f"save: {output_path}")
