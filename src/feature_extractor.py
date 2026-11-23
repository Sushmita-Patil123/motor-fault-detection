import numpy as np
import scipy.stats as stats
import pandas as pd

class VibrationFeatureExtractor:
    """
    Extracts time-domain and frequency-domain features from raw vibration time series windows.
    Matches the exact feature format in feature_time_48k_2048_load_1.csv:
    ['max', 'min', 'mean', 'sd', 'rms', 'skewness', 'kurtosis', 'crest', 'form']
    """
    
    @staticmethod
    def extract_time_features(signal_window: np.ndarray) -> dict:
        signal_window = np.asarray(signal_window, dtype=np.float64).flatten()
        
        max_val = float(np.max(signal_window))
        min_val = float(np.min(signal_window))
        mean_val = float(np.mean(signal_window))
        sd_val = float(np.std(signal_window, ddof=1)) if len(signal_window) > 1 else float(np.std(signal_window))
        rms_val = float(np.sqrt(np.mean(np.square(signal_window))))
        skew_val = float(stats.skew(signal_window))
        kurt_val = float(stats.kurtosis(signal_window))
        
        peak_abs = max(abs(max_val), abs(min_val))
        crest_val = float(peak_abs / rms_val) if rms_val > 1e-12 else 0.0
        
        mean_abs = abs(mean_val)
        form_val = float(rms_val / mean_abs) if mean_abs > 1e-12 else 0.0
        
        return {
            'max': max_val,
            'min': min_val,
            'mean': mean_val,
            'sd': sd_val,
            'rms': rms_val,
            'skewness': skew_val,
            'kurtosis': kurt_val,
            'crest': crest_val,
            'form': form_val
        }
    
    @staticmethod
    def extract_frequency_features(signal_window: np.ndarray, fs: float = 48000.0) -> dict:
        signal_window = np.asarray(signal_window, dtype=np.float64).flatten()
        n = len(signal_window)
        if n == 0:
            return {'peak_freq': 0.0, 'spectral_energy': 0.0, 'spectral_centroid': 0.0, 'freqs': np.array([]), 'fft_mag': np.array([])}
            
        fft_vals = np.fft.rfft(signal_window)
        fft_mag = np.abs(fft_vals) / n
        freqs = np.fft.rfftfreq(n, d=1.0/fs)
        
        peak_freq = float(freqs[np.argmax(fft_mag)])
        spectral_energy = float(np.sum(np.square(fft_mag)))
        spectral_centroid = float(np.sum(freqs * fft_mag) / (np.sum(fft_mag) + 1e-12))
        
        return {
            'peak_freq': peak_freq,
            'spectral_energy': spectral_energy,
            'spectral_centroid': spectral_centroid,
            'freqs': freqs,
            'fft_mag': fft_mag
        }

    @staticmethod
    def segment_signal(raw_signal: np.ndarray, window_size: int = 2048, step_size: int = 2048) -> list:
        raw_signal = np.asarray(raw_signal, dtype=np.float64).flatten()
        windows = []
        for i in range(0, len(raw_signal) - window_size + 1, step_size):
            windows.append(raw_signal[i : i + window_size])
        return windows
