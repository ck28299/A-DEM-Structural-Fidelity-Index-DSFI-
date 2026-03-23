import os
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import gaussian_filter, sobel, generic_filter
from scipy.stats import skew, kurtosis
from scipy.fftpack import fft2, fftshift
from skimage.feature import local_binary_pattern
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
import richdem as rd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

def load_feature_weights(weight_csv_path):
    df = pd.read_csv(weight_csv_path)
    return dict(zip(df['Feature'], df['Importance']))


def read_dem(dem_path):
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
    return dem

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

def extract_texture_features(dem, resolution):
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


def compute_texture_index(features, weights):
    total = 0
    for key, weight in weights.items():
        val = features.get(key)
        if val is not None:
            total += val * weight
        else:
            print(f' {key} no')
    return total

def process_dem_folder(folder_path, weights, label):
    all_features = []
    all_texture_indices = []

    for fname in tqdm(os.listdir(folder_path), desc=f"process {label}"):
        if fname.lower().endswith(".tif"):
            dem_path = os.path.join(folder_path, fname)
            try:
                dem = read_dem(dem_path)
                resolution = get_resolution(dem_path)
                features = extract_texture_features(dem, resolution)
                texture_index = compute_texture_index(features, weights)
                features['texture_index'] = texture_index
                features['filename'] = fname
                all_features.append(features)
                all_texture_indices.append(texture_index)
            except Exception as e:
                print(f" {fname}， {e}")
    return pd.DataFrame(all_features), np.array(all_texture_indices)


def main_multi_compare(folder_list, weights_csv,data_csv, label_list=None):
    print("load...")
    weights = load_feature_weights(weights_csv)

    if label_list is None:
        label_list = [f"Group {i+1}" for i in range(len(folder_list))]

    all_dfs = []
    all_indices = []

    for folder, label in zip(folder_list, label_list):
        print(f" {label}")
        df, _ = process_dem_folder(folder, weights, label)
        all_dfs.append(df)
        all_indices.append(df['texture_index'].values)


    # combined = np.concatenate(all_indices).reshape(-1, 1)
    # scaler = MinMaxScaler(feature_range=(1, 100))
    # scaled = scaler.fit_transform(combined).flatten()
    #
    # offset = 0
    # for i, df in enumerate(all_dfs):
    #     length = len(df)
    #     df['texture_index_norm'] = scaled[offset:offset + length]
    #     offset += length

    print("\n boxplot")
    plt.figure(figsize=(10, 6))
    data = [df['texture_index'] for df in all_dfs]
    plt.boxplot(data, labels=label_list)

    plt.title("Multi-group Texture Index Comparison")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("texture_index_comparison_boxplot.svg")
    plt.show()
    plt.show()

    for df, label in zip(all_dfs, label_list):
        df.to_csv(f"{label}_texture_features.csv", index=False)



    boxplot_data = []
    for label, df in zip(label_list, all_dfs):
        for value in df['texture_index']:
            boxplot_data.append({'Group': label, 'TextureIndex': value})

    boxplot_df = pd.DataFrame(boxplot_data)
    boxplot_df.to_csv(data_csv, index=False)
    print(" texture_index_boxplot_data.csv")

    print("work done！")



    return all_dfs


if __name__ == "__main__":
    folders = [
        r"",
        r"",
        ...




    ]
    labels = ['class1', 'class1_persudo', 'class2', ... ]
    weight_csv = r""
    data_csv=r''
    main_multi_compare(folders, weight_csv, data_csv,labels)
