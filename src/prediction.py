import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.io as sio

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(SCRIPT_DIR, "models")):
    BASE_DIR = SCRIPT_DIR
elif os.path.exists(os.path.join(SCRIPT_DIR, "..", "models")):
    BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
else:
    BASE_DIR = r"c:\FAULT_DETECTION"

MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
LE_PATH = os.path.join(MODELS_DIR, "label_encoder.joblib")

FEATURE_COLS = ['max', 'min', 'mean', 'sd', 'rms', 'skewness', 'kurtosis', 'crest', 'form']

BASE_HEALTH_MAP = {
    'Normal_1': 100.0,
    'Ball_007_1': 78.0,
    'IR_007_1': 72.0,
    'OR_007_6_1': 70.0,
    'Ball_014_1': 52.0,
    'IR_014_1': 45.0,
    'OR_014_6_1': 42.0,
    'Ball_021_1': 22.0,
    'IR_021_1': 15.0,
    'OR_021_6_1': 10.0
}

LOCATION_MAP = {
    'Normal_1': 'Motor Bearings Healthy',
    'Ball_007_1': 'Bearing Rolling Element (Ball)',
    'Ball_014_1': 'Bearing Rolling Element (Ball)',
    'Ball_021_1': 'Bearing Rolling Element (Ball)',
    'IR_007_1': 'Bearing Inner Raceway',
    'IR_014_1': 'Bearing Inner Raceway',
    'IR_021_1': 'Bearing Inner Raceway',
    'OR_007_6_1': 'Bearing Outer Raceway (6 o\'clock)',
    'OR_014_6_1': 'Bearing Outer Raceway (6 o\'clock)',
    'OR_021_6_1': 'Bearing Outer Raceway (6 o\'clock)'
}

SEVERITY_MAP = {
    'Normal_1': '0.000" (Nominal)',
    'Ball_007_1': '0.007" (Minor Defect)',
    'Ball_014_1': '0.014" (Moderate Defect)',
    'Ball_021_1': '0.021" (Severe Defect)',
    'IR_007_1': '0.007" (Minor Defect)',
    'IR_014_1': '0.014" (Moderate Defect)',
    'IR_021_1': '0.021" (Severe Defect)',
    'OR_007_6_1': '0.007" (Minor Defect)',
    'OR_014_6_1': '0.014" (Moderate Defect)',
    'OR_021_6_1': '0.021" (Severe Defect)'
}


def extract_features_from_signal(signal_window: np.ndarray) -> dict:
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
        'max': max_val, 'min': min_val, 'mean': mean_val,
        'sd': sd_val, 'rms': rms_val, 'skewness': skew_val,
        'kurtosis': kurt_val, 'crest': crest_val, 'form': form_val
    }


def resolve_target_file(path: str) -> str:
    clean_path = path.strip().strip('<>').strip('"').strip("'")
    if os.path.exists(clean_path):
        resolved = clean_path
    elif os.path.exists(os.path.join(BASE_DIR, clean_path)):
        resolved = os.path.join(BASE_DIR, clean_path)
    else:
        return clean_path
        
    if os.path.isdir(resolved):
        for root, dirs, files in os.walk(resolved):
            data_files = [f for f in files if not f.endswith('.pdf') and not f.endswith('.zip') and not f.endswith('.py') and not f.endswith('.html')]
            if data_files:
                chosen = os.path.join(root, data_files[-1] if len(data_files) > 1 else data_files[0])
                print(f"[INFO] Directory detected. Selected snapshot: {os.path.basename(chosen)} ({len(data_files)} total recordings)")
                return chosen
    return resolved


def load_and_parse_input_file(filepath: str, sample_idx: int = 0) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
        
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.mat':
        mat = sio.loadmat(filepath)
        target_arr = None
        for k in ['Data', 'data', 'X', 'val']:
            if k in mat and isinstance(mat[k], np.ndarray) and np.issubdtype(mat[k].dtype, np.number):
                target_arr = mat[k]
                break
                
        if target_arr is None:
            de_keys = [k for k in mat.keys() if 'DE_time' in k or 'vibration' in k.lower() or 'acc' in k.lower()]
            if not de_keys:
                for k, v in mat.items():
                    if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number) and not k.startswith('__'):
                        de_keys = [k]
                        break
            if de_keys:
                target_arr = mat[de_keys[0]]
                
        if target_arr is None:
            raise ValueError(f"Could not find vibration data inside {filepath}")
            
        if target_arr.ndim == 2 and target_arr.shape[0] > 1 and target_arr.shape[1] > 100:
            print(f"[INFO] Matrix dataset detected with {target_arr.shape[0]} trials x {target_arr.shape[1]} samples.")
            all_feats = []
            for r in range(min(target_arr.shape[0], 200)):
                all_feats.append(extract_features_from_signal(target_arr[r]))
            return pd.DataFrame(all_feats)
        else:
            raw_signal = target_arr.flatten()
            window_size = 2048
            if len(raw_signal) >= window_size:
                feats = extract_features_from_signal(raw_signal[:window_size])
            else:
                feats = extract_features_from_signal(raw_signal)
            return pd.DataFrame([feats])
        
    elif ext in ['.xlsx', '.xls']:
        df = pd.read_excel(filepath)
        has_all_feats = all(c in df.columns for c in FEATURE_COLS)
        if has_all_feats:
            return df[FEATURE_COLS]
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            raw_signal = df[numeric_cols[0]].dropna().values
            feats = extract_features_from_signal(raw_signal[:2048] if len(raw_signal) >= 2048 else raw_signal)
            return pd.DataFrame([feats])
            
    else:
        try:
            df = pd.read_csv(filepath, sep=r'\s+', header=None, engine='c')
        except Exception:
            try:
                df = pd.read_csv(filepath)
            except Exception:
                df = pd.read_csv(filepath, sep=None, engine='python')
                
        has_all_feats = all(c in df.columns for c in FEATURE_COLS)
        if has_all_feats:
            return df[FEATURE_COLS]
            
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            raw_signal = df[numeric_cols[0]].dropna().values
            window_size = 2048
            if len(raw_signal) >= window_size:
                feats = extract_features_from_signal(raw_signal[:window_size])
            else:
                feats = extract_features_from_signal(raw_signal)
            return pd.DataFrame([feats])
        else:
            raise ValueError(f"No numeric data found in {filepath}")


def analyze_motor_data(input_path: str, trial_idx: int = 0):
    resolved_path = resolve_target_file(input_path)
    if not os.path.exists(resolved_path):
        print(f"[ERROR] Path does not exist: '{input_path}'")
        sys.exit(1)
        
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LE_PATH):
        print(f"[ERROR] Trained model files not found in {MODELS_DIR}. Run 'python src/train_models.py' first.")
        sys.exit(1)
        
    model = joblib.load(MODEL_PATH)
    le = joblib.load(LE_PATH)
    
    feats_df = load_and_parse_input_file(resolved_path)
    
    if len(feats_df) > 1:
        trial_idx = min(trial_idx, len(feats_df) - 1)
        sample_feat = feats_df.iloc[trial_idx:trial_idx+1][FEATURE_COLS]
    else:
        sample_feat = feats_df.iloc[0:1][FEATURE_COLS]
    
    pred_encoded = model.predict(sample_feat)[0]
    pred_class = le.inverse_transform([pred_encoded])[0]
    probs = model.predict_proba(sample_feat)[0]
    prob_dict = {c: float(p) for c, p in zip(le.classes_, probs)}
    
    rms_val = float(sample_feat['rms'].values[0])
    
    # Calculate probability-weighted Health Index
    weighted_hi = sum(prob_dict.get(c, 0.0) * BASE_HEALTH_MAP.get(c, 50.0) for c in le.classes_)
    if rms_val > 0.15:
        rms_penalty = min(20.0, (rms_val - 0.15) * 15.0)
        weighted_hi = max(5.0, weighted_hi - rms_penalty)
    health_index = int(round(float(np.clip(weighted_hi, 0.0, 100.0))))
    
    # Fault Location and Severity
    loc = LOCATION_MAP.get(pred_class, 'Motor Bearing Assembly')
    sev = SEVERITY_MAP.get(pred_class, 'Moderate')
    
    # DYNAMIC STATUS, PREDICTION, AND RECOMMENDED ACTION BASED ON HEALTH INDEX
    if health_index < 35:
        status = "Abnormal vibration detected (Critical Alert)"
        prediction = f"Catastrophic failure / motor seizure imminent in < 80 operating hours. Defect at {loc}."
        action = "Urgent machine shutdown recommended. Immediately inspect and replace worn bearing assembly."
    elif health_index < 65:
        status = "Moderate vibration fault detected (Alert)"
        prediction = f"Bearing wear at {loc} ({sev}) likely to reach critical threshold in 120-400 operating hours."
        action = "Inspect bearing assembly and replace worn bearing during next planned shutdown. Check lubrication."
    elif health_index < 85:
        status = "Abnormal vibration detected (Early Warning)"
        prediction = f"Early degradation detected at {loc}. Defect growth likely within ~1,500 operating hours."
        action = "Schedule non-urgent bearing inspection during next planned downtime. Monitor weekly vibration."
    else:
        status = "Normal operation (Healthy)"
        prediction = "Motor operating within normal degradation limits (Safe)."
        action = "No maintenance needed. Continue regular operation schedule."
        if pred_class != 'Normal_1':
            loc = "Motor Bearings Healthy"
            
    print("\n" + "-"*45)
    print("Example Output")
    print("-"*45 + "\n")
    print(f"Machine Status: {status}")
    print(f"Fault Location: {loc}")
    print(f"Health Index: {health_index}%\n")
    print("Prediction:")
    print(f"{prediction}\n")
    print("Recommended Action:")
    print(f"{action}")
    print("\n" + "-"*45 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Induction Motor Fault Detection & Health Prediction Tool")
    parser.add_argument("file", nargs="?", default=None, help="Path to new motor sensor data file")
    parser.add_argument("--trial", type=int, default=0, help="Trial index if file contains multiple runs (e.g. Data.mat)")
    args = parser.parse_args()
    
    filepath = args.file
    if not filepath:
        print("\n" + "="*50)
        print("  INDUCTION MOTOR HEALTH PREDICTION")
        print("="*50)
        print("Enter the path to your motor sensor file:")
        print("  e.g.  Data.mat")
        print("  or    dataset/1st_test")
        print("="*50)
        
        user_input = input("\nEnter file path: ").strip()
        filepath = user_input.replace('"', '').replace("'", "").replace('<', '').replace('>', '')
            
    analyze_motor_data(filepath, trial_idx=args.trial)


if __name__ == "__main__":
    main()
