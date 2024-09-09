import scipy.io as scio
from scipy import interpolate
import numpy as np
from post_process.find_peaks import calculate_difference_mean, find_box, normalize_matrix_with_max_values

def preprocess_data(NmrData, ins_tpyer="quadratic"):
    b = NmrData['b'][0]
    S = NmrData['S']
    new_b = np.linspace(0, np.max(b), 30)
    interp_func = np.vectorize(lambda i: interpolate.interp1d(b, S[i, :], kind=ins_tpyer, fill_value='extrapolate')(new_b), signature='()->(n)')
    return np.stack(interp_func(np.arange(S.shape[0])))[np.newaxis, :, :]

def get_noise_S(S, db):
    snr = np.exp(np.log(10) * float(db) / 10)
    noise_S = np.zeros_like(S)
    sigma = np.sqrt(1. / snr)

    for i in range(S.shape[0]):
        for j in range(S.shape[2]):
            noise = np.random.randn(S.shape[1], 1)
            signal_energy_per_column = np.linalg.norm(S[i, :, j], 2)
            noise_energy_per_column = np.linalg.norm(noise, 2)
            mult = sigma * signal_energy_per_column / noise_energy_per_column
            noise *= mult
            noise_S[i, :, j] = S[i, :, j] + noise.flatten()
    return noise_S

def process_result(mean, ppm, var, idx_peaks, HNMR, expand=5):
    cs_spec = np.zeros([(ppm.size), 1])
    spec_whole = np.zeros([len(mean[0, :]), ppm.size])
    cs_spec[idx_peaks, :] = HNMR
    spec_whole[0:140, idx_peaks[0, :]] = mean.T
    spec_var = np.zeros([len(mean[0, :]), ppm.size])
    spec_var[0:140, idx_peaks[0, :]] = var.T
    merged_boxes = find_box(spec_var, expand_margin=1, expand_margin_x=expand) 
    norm_var = normalize_matrix_with_max_values(spec_whole, spec_var, merged_boxes, cs_spec)
    max_var = calculate_difference_mean(norm_var, merged_boxes)
    return norm_var, max_var, merged_boxes

def load_data(type):
    data = {
        'NmrData':scio.loadmat(f'Dataset/{type}_net_input.mat'),
        'b': scio.loadmat(f'Dataset/{type}_net_input.mat')['b'],
        'idx_peaks': scio.loadmat(f'Dataset/{type}_net_input.mat')['idx_peaks'],
        'ppm': scio.loadmat(f'Dataset/{type}_net_input.mat')['ppm'],
        'HNMR': scio.loadmat(f'Dataset/{type}_net_input.mat')['HNMR'],
    }

    return data

def result_process(type):
    paraments = {
    "QGC": (0.025, 0.7, "quadratic", 0.6, 10),
    "GSP": (0.035, 0.7, "quadratic", 0.6, 5),
    "M6": (0.035, 0.7, "quadratic", 0.6, 20),
    "JNN": (0.06, 0.9, "linear", 0.6, 5),
    "TSP": (0.05, 0.7, "linear", 0.6, 20),
    "EC": (0.03, 0.9, "linear", 0.45, 20),
    "AMDK": (0.02, 0.9, "quadratic", 0.45, 10),
    "BPP1": (0.03, 0.9, "linear", 0.9, 5),
    "BPP2": (0.03, 0.9, "linear", 0.6, 0),
    "QG": (0.06, 0.9, "linear", 0.5, 5),
}
    if type not in paraments:
        return (0.035, 0.7, "quadratic", 0.6, 5)
    return paraments[type]
