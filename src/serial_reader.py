import os
import time
import numpy as np
import scipy.io as sio

class MotorDataStreamer:
    """
    Handles reading raw vibration data from USB Serial (ESP32/Arduino)
    or simulated motor replays from CWRU .mat files.
    """
    
    def __init__(self, workspace_path: str = r"c:\FAULT_DETECTION"):
        self.workspace_path = workspace_path
        self.raw_dir = os.path.join(workspace_path, "raw")
        self.available_sim_files = {
            "Normal (Healthy Motor)": "Time_Normal_1_098.mat",
            "Ball Fault 0.007\" (Minor)": "B007_1_123.mat",
            "Ball Fault 0.014\" (Moderate)": "B014_1_190.mat",
            "Ball Fault 0.021\" (Severe)": "B021_1_227.mat",
            "Inner Race Fault 0.007\" (Minor)": "IR007_1_110.mat",
            "Inner Race Fault 0.014\" (Moderate)": "IR014_1_175.mat",
            "Inner Race Fault 0.021\" (Severe)": "IR021_1_214.mat",
            "Outer Race Fault 0.007\" (Minor)": "OR007_6_1_136.mat",
            "Outer Race Fault 0.014\" (Moderate)": "OR014_6_1_202.mat",
            "Outer Race Fault 0.021\" (Severe)": "OR021_6_1_239.mat",
        }
        
    def load_simulated_window(self, sim_key: str, window_idx: int = 0, window_size: int = 2048) -> np.ndarray:
        filename = self.available_sim_files.get(sim_key, "Time_Normal_1_098.mat")
        filepath = os.path.join(self.raw_dir, filename)
        
        if not os.path.exists(filepath):
            # Fallback synthetic waveform if file not found
            t = np.linspace(0, window_size/48000, window_size)
            return 0.1 * np.sin(2 * np.pi * 30 * t) + 0.02 * np.random.randn(window_size)
            
        mat = sio.loadmat(filepath)
        de_keys = [k for k in mat.keys() if 'DE_time' in k]
        if not de_keys:
            return np.random.randn(window_size) * 0.05
            
        raw_signal = mat[de_keys[0]].flatten()
        start = (window_idx * window_size) % (len(raw_signal) - window_size)
        return raw_signal[start : start + window_size]
