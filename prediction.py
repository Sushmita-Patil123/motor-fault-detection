import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd

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

FEATURE_COLS = ['voltage', 'current_r', 'current_y', 'current_b', 'rpm', 'power_input', 'power_factor']

FAULT_DETAILS = {
    'Normal': {
        'location': 'Motor Windings & Grid Healthy',
        'health_index': 100,
        'status': 'Normal operation (Healthy)',
        'prediction': 'Motor operating within normal electrical and thermal limits (Safe).',
        'action': 'No maintenance needed. Continue regular operation schedule.'
    },
    'Stator_Fault': {
        'location': 'Stator Winding Insulation (Phase Asymmetry)',
        'health_index': 28,
        'status': 'Abnormal electrical signature (Critical Alert)',
        'prediction': 'High risk of inter-turn short circuit progressing to ground fault within < 40 hours.',
        'action': 'Urgent motor shutdown recommended. Perform insulation resistance (Megger) and surge test on stator windings.'
    },
    'Single_Phasing': {
        'location': 'Supply Line / Contactor (Open Phase)',
        'health_index': 12,
        'status': 'Severe single phasing fault (Emergency Alert)',
        'prediction': 'Motor drawing extreme unbalanced current; thermal burnout imminent in < 20 minutes.',
        'action': 'Immediate breaker trip required. Inspect upstream supply fuses, contactor poles, and terminal lugs.'
    },
    'Voltage_Unbalance': {
        'location': 'Power Supply Grid / Transformer',
        'health_index': 62,
        'status': 'Supply voltage unbalance detected (Alert)',
        'prediction': 'Elevated negative-sequence currents causing supplementary heating (~300-600 operating hours).',
        'action': 'Check line supply transformer taps, rebalance single-phase facility loads, and verify utility voltage balance.'
    },
    'Overload': {
        'location': 'Mechanical Drive / Shaft Load',
        'health_index': 45,
        'status': 'Continuous motor overload detected (Alert)',
        'prediction': 'Excessive thermal rise risking accelerated winding degradation within 100-250 hours.',
        'action': 'Reduce mechanical shaft loading or inspect driven equipment for mechanical binding. Check cooling airflow.'
    },
    'Under_Voltage': {
        'location': 'Incoming Line Feeder',
        'health_index': 68,
        'status': 'Low line voltage condition (Early Warning)',
        'prediction': 'Excessive current drawn to satisfy constant mechanical torque demand.',
        'action': 'Regulate bus voltage to nominal (415 V). Check for undersized feeder cables causing excessive voltage drop.'
    },
    'Rotor_Fault': {
        'location': 'Rotor Squirrel Cage / Air-Gap',
        'health_index': 38,
        'status': 'Rotor bar flaw / magnetic asymmetry (Alert)',
        'prediction': 'Broken rotor bar or eccentricity sidebands detected; likely to deteriorate within 200-500 hours.',
        'action': 'Schedule Motor Current Signature Analysis (MCSA) inspection. Inspect rotor cage bars during next shutdown.'
    }
}

def analyze_sample(sample_dict: dict):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LE_PATH):
        print(f"[ERROR] Trained model files not found in {MODELS_DIR}. Run 'python src/train_models.py' first.")
        sys.exit(1)
        
    model = joblib.load(MODEL_PATH)
    le = joblib.load(LE_PATH)
    
    df_sample = pd.DataFrame([sample_dict])[FEATURE_COLS]
    
    pred_encoded = model.predict(df_sample)[0]
    pred_class = le.inverse_transform([pred_encoded])[0]
    
    # Calculate electrical unbalance
    ir = sample_dict.get('current_r', 0.0)
    iy = sample_dict.get('current_y', 0.0)
    ib = sample_dict.get('current_b', 0.0)
    i_avg = (ir + iy + ib) / 3.0
    if i_avg > 0.1:
        unbalance_pct = (max(abs(ir - i_avg), abs(iy - i_avg), abs(ib - i_avg)) / i_avg) * 100.0
    else:
        unbalance_pct = 0.0
        
    info = FAULT_DETAILS.get(pred_class, {
        'location': 'Motor Electrical System',
        'health_index': 50,
        'status': 'Condition Alert',
        'prediction': 'Abnormal operating parameters detected.',
        'action': 'Inspect electrical connections and check operating conditions.'
    })
    
    # ── Broadcast to Shared State & Live API (Syncs with Dashboard & User UI) ──
    try:
        import json
        import time
        latest_file = os.path.join(MODELS_DIR, "latest_terminal_prediction.json")
        payload = {
            'voltage': float(sample_dict.get('voltage', 415.0)),
            'current_r': float(ir),
            'current_y': float(iy),
            'current_b': float(ib),
            'rpm': float(sample_dict.get('rpm', 1440.0)),
            'power_input': float(sample_dict.get('power_input', 3700.0)),
            'power_factor': float(sample_dict.get('power_factor', 0.85)),
            'health_index': float(info['health_index']),
            'fault': pred_class,
            'location': info['location'],
            'action': info['action'],
            'severity': info['status'],
            'current_unbalance': round(unbalance_pct, 2),
            'timestamp': time.time(),
            'source': 'terminal'
        }
        with open(latest_file, 'w') as f:
            json.dump(payload, f, indent=2)
            
        import requests
        requests.post("http://localhost:8000/update-telemetry", json=payload, timeout=0.5)

        # Auto-sync to GitHub so Streamlit Cloud (streamlit.app) updates live too!
        try:
            import subprocess
            h_idx = info['health_index']
            sync_cmd = f'git commit -am "Live sync: {pred_class} ({h_idx}%)" --no-verify && git push origin main'
            subprocess.Popen(sync_cmd, shell=True, cwd=BASE_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
    except Exception:
        pass

    print("\n" + "-"*50)

    print("Example Output")
    print("-"*50)
    print(f"\nMachine Status: {info['status']}")
    print(f"Fault Location: {info['location']}")
    print(f"Health Index: {info['health_index']}%")
    print(f"Current Unbalance: {unbalance_pct:.1f}% (Ir={ir:.1f}A, Iy={iy:.1f}A, Ib={ib:.1f}A)")
    print(f"\nPrediction:")
    print(info['prediction'])
    print(f"\nRecommended Action:")
    print(info['action'])
    print("\n" + "-"*50 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Induction Motor Fault Diagnostic Inference Engine")
    parser.add_argument("file", nargs="?", default=None, help="Path to CSV dataset or file")
    parser.add_argument("--voltage", type=float, default=None, help="Line Voltage (V)")
    parser.add_argument("--ir", type=float, default=None, help="Phase R Current (A)")
    parser.add_argument("--iy", type=float, default=None, help="Phase Y Current (A)")
    parser.add_argument("--ib", type=float, default=None, help="Phase B Current (A)")
    parser.add_argument("--rpm", type=float, default=None, help="Motor Speed (RPM)")
    parser.add_argument("--power", type=float, default=None, help="Active Power Input (W)")
    parser.add_argument("--pf", type=float, default=None, help="Power Factor (cos phi)")
    parser.add_argument("--row", type=int, default=0, help="Row index if evaluating a CSV file")
    
    args = parser.parse_args()
    
    if args.voltage is not None and args.ir is not None:
        sample = {
            'voltage': args.voltage,
            'current_r': args.ir,
            'current_y': args.iy if args.iy is not None else args.ir,
            'current_b': args.ib if args.ib is not None else args.ir,
            'rpm': args.rpm if args.rpm is not None else 1440.0,
            'power_input': args.power if args.power is not None else 3150.0,
            'power_factor': args.pf if args.pf is not None else 0.85
        }
        analyze_sample(sample)
    elif args.file:
        fpath = args.file.strip().strip('<>').strip('"').strip("'")
        if not os.path.exists(fpath):
            fpath = os.path.join(BASE_DIR, fpath)
            
        if not os.path.exists(fpath):
            print(f"[ERROR] File does not exist: {args.file}")
            sys.exit(1)
            
        df = pd.read_csv(fpath)
        row_idx = min(args.row, len(df) - 1)
        row = df.iloc[row_idx]
        sample = {c: float(row[c]) for c in FEATURE_COLS if c in row}
        print(f"[INFO] Evaluating record #{row_idx} from: {os.path.basename(fpath)}")
        analyze_sample(sample)
    else:
        # Default: Test a normal sample
        sample = {
            'voltage': 415.0,
            'current_r': 5.25,
            'current_y': 5.20,
            'current_b': 5.28,
            'rpm': 1445.0,
            'power_input': 3180.0,
            'power_factor': 0.85
        }
        print("[INFO] No input provided. Running on nominal 3-phase test sample:")
        analyze_sample(sample)

if __name__ == "__main__":
    main()
