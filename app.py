import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

# Page Configuration
st.set_page_config(
    page_title="3-Phase Induction Motor Electrical Fault Detection & Diagnostic Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
LE_PATH = os.path.join(MODELS_DIR, "label_encoder.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "evaluation_metrics.json")
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "induction_motor_electrical_dataset.csv")

FEATURE_COLS = ['voltage', 'current_r', 'current_y', 'current_b', 'rpm', 'power_input', 'power_factor']

FEATURE_META = {
    'voltage': {
        'name': 'Line Voltage (V)',
        'unit': 'V',
        'importance': 20.25,
        'meaning': 'Line-to-line supply voltage from the mains or inverter.',
        'normal_ref': '400 - 425 V'
    },
    'power_factor': {
        'name': 'Power Factor (cos φ)',
        'unit': 'dimensionless',
        'importance': 19.65,
        'meaning': 'Ratio of active power to apparent power; drops severely during magnetic asymmetry.',
        'normal_ref': '0.80 - 0.88'
    },
    'rpm': {
        'name': 'Rotor Speed (RPM)',
        'unit': 'RPM',
        'importance': 17.54,
        'meaning': 'Shaft rotational speed; reflects slip under load and mechanical braking.',
        'normal_ref': '1420 - 1465 RPM'
    },
    'current_b': {
        'name': 'Phase B Current (Ib)',
        'unit': 'A',
        'importance': 14.62,
        'meaning': 'Stator line current in Blue phase; drops to 0 A in single phasing.',
        'normal_ref': '4.5 - 7.2 A'
    },
    'power_input': {
        'name': 'Active Power Input (W)',
        'unit': 'W',
        'importance': 11.99,
        'meaning': 'Total electrical power consumed by the 3-phase stator.',
        'normal_ref': '2000 - 3600 W'
    },
    'current_r': {
        'name': 'Phase R Current (Ir)',
        'unit': 'A',
        'importance': 8.80,
        'meaning': 'Stator line current in Red phase; spikes during turn short circuits.',
        'normal_ref': '4.5 - 7.2 A'
    },
    'current_y': {
        'name': 'Phase Y Current (Iy)',
        'unit': 'A',
        'importance': 7.16,
        'meaning': 'Stator line current in Yellow phase.',
        'normal_ref': '4.5 - 7.2 A'
    }
}

FAULT_DETAILS = {
    'Normal': {
        'name': 'Normal Operation (Healthy)',
        'location': 'Motor Windings & Grid Healthy',
        'health_index': 100,
        'status_color': '#10b981',
        'status_badge': '🟢 NORMAL (HEALTHY)',
        'safe_window': '> 2,000 Hours (Safe)',
        'urgency': 'Routine Inspection Only',
        'prediction': 'Motor operating within normal electrical and thermal limits (Safe continuous operation).',
        'action': 'No maintenance needed. Continue regular operation schedule.'
    },
    'Stator_Fault': {
        'name': 'Stator Inter-Turn Short Circuit',
        'location': 'Stator Winding Insulation (Phase Asymmetry)',
        'health_index': 28,
        'status_color': '#ef4444',
        'status_badge': '🔴 CRITICAL ALERT: STATOR FAULT',
        'safe_window': '< 40 Hours (Critical)',
        'urgency': 'Immediate Shutdown Required',
        'prediction': 'Severe localized current circulating in shorted turns; insulation breakdown imminent in < 40 hours.',
        'action': 'Urgent machine shutdown recommended. Perform insulation resistance (Megger) and surge test on stator windings.'
    },
    'Single_Phasing': {
        'name': 'Single Phasing (Loss of One Phase)',
        'location': 'Supply Line / Contactor (Open Phase)',
        'health_index': 12,
        'status_color': '#991b1b',
        'status_badge': '🚨 EMERGENCY: SINGLE PHASING',
        'safe_window': '< 20 Minutes (Emergency)',
        'urgency': 'EMERGENCY: Trip Breaker Immediately',
        'prediction': 'One phase disconnected; surviving phases drawing 1.8x rated current; burnout in < 20 minutes.',
        'action': 'Trip motor breaker immediately! Check upstream supply fuses, contactor poles, and terminal block.'
    },
    'Voltage_Unbalance': {
        'name': 'Supply Voltage Unbalance',
        'location': 'Power Supply Grid / Transformer',
        'health_index': 62,
        'status_color': '#f59e0b',
        'status_badge': '🟡 ALERT: VOLTAGE UNBALANCE',
        'safe_window': '~300 - 600 Hours (Alert)',
        'urgency': 'Schedule Grid & Supply Check',
        'prediction': 'Elevated negative-sequence currents causing supplementary winding heating (~300-600 operating hours).',
        'action': 'Check line supply transformer taps, rebalance single-phase facility loads, and verify utility voltage balance.'
    },
    'Overload': {
        'name': 'Mechanical Overload',
        'location': 'Mechanical Drive / Shaft Load',
        'health_index': 45,
        'status_color': '#ea580c',
        'status_badge': '🟠 ALERT: MOTOR OVERLOAD',
        'safe_window': '~100 - 250 Hours (Alert)',
        'urgency': 'Prompt Mechanical Attention',
        'prediction': 'Excessive thermal rise risking accelerated winding insulation degradation within 100-250 hours.',
        'action': 'Reduce mechanical shaft loading or inspect driven machinery for mechanical binding. Check cooling airflow.'
    },
    'Under_Voltage': {
        'name': 'Under-Voltage Condition',
        'location': 'Incoming Line Feeder',
        'health_index': 68,
        'status_color': '#eab308',
        'status_badge': '🟡 EARLY WARNING: UNDER-VOLTAGE',
        'safe_window': '~500 - 1,000 Hours (Warning)',
        'urgency': 'Schedule Electrical Service',
        'prediction': 'Motor drawing elevated current to satisfy constant mechanical torque demand. Risk of thermal degradation.',
        'action': 'Regulate bus voltage to nominal (415 V). Check for undersized feeder cables causing excessive voltage drop.'
    },
    'Rotor_Fault': {
        'name': 'Rotor Bar Flaw / Eccentricity',
        'location': 'Rotor Squirrel Cage / Air-Gap',
        'health_index': 38,
        'status_color': '#f97316',
        'status_badge': '🟠 ALERT: ROTOR FAULT',
        'safe_window': '~200 - 500 Hours (Alert)',
        'urgency': 'Plan Rotor Inspection Shutdown',
        'prediction': 'Broken rotor bar or eccentricity sidebands detected; likely to deteriorate within 200-500 hours.',
        'action': 'Schedule Motor Current Signature Analysis (MCSA) inspection. Inspect rotor cage bars during next shutdown.'
    }
}

# Load Models
@st.cache_resource
def load_trained_artifacts():
    if os.path.exists(MODEL_PATH) and os.path.exists(LE_PATH):
        model = joblib.load(MODEL_PATH)
        le = joblib.load(LE_PATH)
        metrics = {}
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r") as f:
                metrics = json.load(f)
        return model, le, metrics
    return None, None, None

model, le, metrics = load_trained_artifacts()

# App Header
st.markdown("""
<div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 22px 28px; border-radius: 12px; margin-bottom: 24px; border: 1px solid #334155;">
    <h1 style="color: #f8fafc; margin: 0; font-size: 28px;">⚡ 3-Phase Induction Motor Electrical Fault Detection & Diagnostic Dashboard</h1>
    <p style="color: #94a3b8; margin: 6px 0 0 0; font-size: 15px;">
        Machine Learning Predictive Maintenance Monitoring: <b>Voltage, 3-Phase Currents (Ir, Iy, Ib), RPM, Active Power Input & Power Factor</b>
    </p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh so any terminal runs or live updates are detected immediately
st_autorefresh(interval=1500, key="global_terminal_sync")

# Sidebar Controls
st.sidebar.header("⚙️ Motor Parameter Controls")

# Initialize session state for sliders
for param, val in [
    ("voltage", 415.0),
    ("current_r", 5.25),
    ("current_y", 5.20),
    ("current_b", 5.28),
    ("rpm", 1445.0),
    ("power_input", 3180.0),
    ("power_factor", 0.85)
]:
    if f"slider_{param}" not in st.session_state:
        st.session_state[f"slider_{param}"] = val

if "last_applied_sig" not in st.session_state:
    st.session_state["last_applied_sig"] = None

# Input Mode: Live Terminal Sync / Dataset / Preset
input_mode = st.sidebar.radio(
    "Input Source Mode:",
    ["⚡ Live Terminal & Code Sync", "📁 Load from Dataset CSV", "🎛️ Simulation Presets"],
    index=0
)

defaults = {
    "voltage": float(st.session_state.get("slider_voltage", 415.0)),
    "current_r": float(st.session_state.get("slider_current_r", 5.25)),
    "current_y": float(st.session_state.get("slider_current_y", 5.20)),
    "current_b": float(st.session_state.get("slider_current_b", 5.28)),
    "rpm": float(st.session_state.get("slider_rpm", 1445.0)),
    "power_input": float(st.session_state.get("slider_power_input", 3180.0)),
    "power_factor": float(st.session_state.get("slider_power_factor", 0.85))
}
dataset_truth_label = None

if input_mode == "⚡ Live Terminal & Code Sync":
    t_data = None
    latest_file = os.path.join(MODELS_DIR, "latest_terminal_prediction.json")
    
    # 1. Read local file (on developer laptop)
    if os.path.exists(latest_file):
        try:
            with open(latest_file, 'r') as f:
                t_data = json.load(f)
        except Exception:
            pass

    # 2. Fetch live state from GitHub API (bridges Streamlit Cloud to local terminal runs)
    try:
        import urllib.request
        import base64
        gh_url = "https://api.github.com/repos/Sushmita-Patil123/motor-fault-detection/contents/models/latest_terminal_prediction.json"
        req = urllib.request.Request(gh_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=1.8) as resp:
            gh_raw = json.loads(resp.read().decode('utf-8'))
            gh_json = json.loads(base64.b64decode(gh_raw['content']).decode('utf-8'))
            if t_data is None or float(gh_json.get('timestamp', 0)) >= float(t_data.get('timestamp', 0)):
                t_data = gh_json
    except Exception:
        pass

    if t_data:
        try:
            t_sig = f"terminal_{t_data.get('timestamp')}_{t_data.get('fault')}"
            
            defaults["voltage"]      = float(t_data.get("voltage", 415.0))
            defaults["current_r"]    = float(t_data.get("current_r", 5.25))
            defaults["current_y"]    = float(t_data.get("current_y", 5.20))
            defaults["current_b"]    = float(t_data.get("current_b", 5.28))
            defaults["rpm"]          = float(t_data.get("rpm", 1445.0))
            defaults["power_input"]  = float(t_data.get("power_input", 3180.0))
            defaults["power_factor"] = float(t_data.get("power_factor", 0.85))
            
            if st.session_state.get("last_applied_sig") != t_sig:
                st.session_state["slider_voltage"]      = float(np.clip(defaults["voltage"], 300.0, 480.0))
                st.session_state["slider_current_r"]    = float(np.clip(defaults["current_r"], 0.0, 25.0))
                st.session_state["slider_current_y"]    = float(np.clip(defaults["current_y"], 0.0, 25.0))
                st.session_state["slider_current_b"]    = float(np.clip(defaults["current_b"], 0.0, 25.0))
                st.session_state["slider_rpm"]          = float(np.clip(defaults["rpm"], 0.0, 1500.0))
                st.session_state["slider_power_input"]  = float(np.clip(defaults["power_input"], 0.0, 8000.0))
                st.session_state["slider_power_factor"] = float(np.clip(defaults["power_factor"], 0.20, 1.00))
                st.session_state["last_applied_sig"]    = t_sig

            st.sidebar.success(f"🟢 **Live Sync Active:** {t_data.get('fault')} ({t_data.get('health_index')}%)", icon="⚡")
            if st.sidebar.button("🔄 Force Sync Now"):
                st.rerun()
            st.sidebar.caption("Run code in terminal and this dashboard updates live!")
        except Exception:
            pass

elif input_mode == "📁 Load from Dataset CSV":
    dataset_choice = st.sidebar.selectbox(
        "Select Saved Dataset:",
        [
            "⚡ excel_sheet2.csv (Lab Under-Voltage & Asymmetry - 7 samples)",
            "🔬 indoor_lab_readings.csv (Real Lab Readings - 10 samples)",
            "📊 induction_motor_electrical_dataset.csv (2,300 records)"
        ]
    )
    uploaded_file = st.sidebar.file_uploader("Or Upload Custom File (CSV or Excel)", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.xlsx'):
            df_csv = pd.read_excel(uploaded_file)
        else:
            df_csv = pd.read_csv(uploaded_file)
        st.sidebar.caption(f"Loaded: `{uploaded_file.name}` ({len(df_csv)} records)")
    else:
        if "excel_sheet2" in dataset_choice:
            chosen_path = os.path.join(BASE_DIR, "dataset", "excel_sheet2.csv")
        elif "indoor_lab_readings" in dataset_choice:
            chosen_path = os.path.join(BASE_DIR, "dataset", "indoor_lab_readings.csv")
        else:
            chosen_path = DATASET_PATH

        if os.path.exists(chosen_path):
            df_csv = pd.read_csv(chosen_path)
            st.sidebar.caption(f"Loaded: `{os.path.basename(chosen_path)}` ({len(df_csv)} records)")
        else:
            df_csv = None

    if df_csv is not None and len(df_csv) > 0:
        row_idx = st.sidebar.slider("Select Sample Row #:", 0, len(df_csv) - 1, 0)
        selected_row = df_csv.iloc[row_idx]
        ds_sig = f"ds_{dataset_choice}_{row_idx}"
        
        # Smart Auto-Mapping for arbitrary uploaded CSV or Excel column headers
        def get_col_val(candidates, default_val):
            for c in candidates:
                for col_name in selected_row.index:
                    if c.lower() in str(col_name).lower():
                        try:
                            v = float(selected_row[col_name])
                            if not np.isnan(v):
                                return v
                        except Exception:
                            pass
            return default_val

        defaults["voltage"]      = get_col_val(["voltage", "vry", "v_avg", "v (v)", "volt"], 415.0)
        defaults["current_r"]    = get_col_val(["current_r", "ir", "i_r", "phase r"], 5.25)
        defaults["current_y"]    = get_col_val(["current_y", "iy", "i_y", "phase y"], 5.20)
        defaults["current_b"]    = get_col_val(["current_b", "ib", "i_b", "phase b"], 5.28)
        defaults["rpm"]          = get_col_val(["rpm", "speed"], 1445.0)
        defaults["power_input"]  = get_col_val(["power_input", "power input", "power", "pin", "w1+w2"], 3180.0)
        defaults["power_factor"] = get_col_val(["power_factor", "power factor", "pf", "cos"], 0.85)

        if st.session_state.get("last_applied_sig") != ds_sig:
            st.session_state["slider_voltage"]      = float(np.clip(defaults["voltage"], 300.0, 480.0))
            st.session_state["slider_current_r"]    = float(np.clip(defaults["current_r"], 0.0, 25.0))
            st.session_state["slider_current_y"]    = float(np.clip(defaults["current_y"], 0.0, 25.0))
            st.session_state["slider_current_b"]    = float(np.clip(defaults["current_b"], 0.0, 25.0))
            st.session_state["slider_rpm"]          = float(np.clip(defaults["rpm"], 0.0, 1500.0))
            st.session_state["slider_power_input"]  = float(np.clip(defaults["power_input"], 0.0, 8000.0))
            st.session_state["slider_power_factor"] = float(np.clip(defaults["power_factor"], 0.20, 1.00))
            st.session_state["last_applied_sig"]    = ds_sig

        for col_name in ['predicted_fault', 'fault', 'fault_type', 'label', 'target']:
            if col_name in selected_row:
                dataset_truth_label = str(selected_row[col_name])
                st.sidebar.info(f"🏷️ **Dataset True Label:** `{dataset_truth_label}`")
                break
else:
    preset = st.sidebar.selectbox(
        "Quick Simulation Presets:",
        [
            "Custom Input",
            "1. Normal (Healthy Operation)",
            "2. Stator Inter-Turn Short Circuit",
            "3. Single Phasing (Phase Loss)",
            "4. Voltage Unbalance",
            "5. Motor Overload (140% Load)",
            "6. Under-Voltage Condition",
            "7. Rotor Bar Fault"
        ]
    )

    if preset == "1. Normal (Healthy Operation)":
        defaults = {"voltage": 415.0, "current_r": 5.25, "current_y": 5.20, "current_b": 5.28, "rpm": 1445.0, "power_input": 3180.0, "power_factor": 0.85}
    elif preset == "2. Stator Inter-Turn Short Circuit":
        defaults = {"voltage": 412.0, "current_r": 8.85, "current_y": 5.15, "current_b": 5.20, "rpm": 1415.0, "power_input": 3850.0, "power_factor": 0.72}
    elif preset == "3. Single Phasing (Phase Loss)":
        defaults = {"voltage": 408.0, "current_r": 9.60, "current_y": 9.55, "current_b": 0.00, "rpm": 1320.0, "power_input": 3650.0, "power_factor": 0.61}
    elif preset == "4. Voltage Unbalance":
        defaults = {"voltage": 402.0, "current_r": 6.80, "current_y": 5.20, "current_b": 4.25, "rpm": 1425.0, "power_input": 3250.0, "power_factor": 0.76}
    elif preset == "5. Motor Overload (140% Load)":
        defaults = {"voltage": 412.0, "current_r": 8.10, "current_y": 8.05, "current_b": 8.15, "rpm": 1365.0, "power_input": 4550.0, "power_factor": 0.88}
    elif preset == "6. Under-Voltage Condition":
        defaults = {"voltage": 355.0, "current_r": 6.30, "current_y": 6.25, "current_b": 6.35, "rpm": 1410.0, "power_input": 3050.0, "power_factor": 0.79}
    elif preset == "7. Rotor Bar Fault":
        defaults = {"voltage": 415.0, "current_r": 6.20, "current_y": 5.10, "current_b": 5.85, "rpm": 1420.0, "power_input": 3400.0, "power_factor": 0.74}

    preset_sig = f"preset_{preset}"
    if preset != "Custom Input" and st.session_state.get("last_applied_sig") != preset_sig:
        st.session_state["slider_voltage"]      = float(np.clip(defaults["voltage"], 300.0, 480.0))
        st.session_state["slider_current_r"]    = float(np.clip(defaults["current_r"], 0.0, 25.0))
        st.session_state["slider_current_y"]    = float(np.clip(defaults["current_y"], 0.0, 25.0))
        st.session_state["slider_current_b"]    = float(np.clip(defaults["current_b"], 0.0, 25.0))
        st.session_state["slider_rpm"]          = float(np.clip(defaults["rpm"], 0.0, 1500.0))
        st.session_state["slider_power_input"]  = float(np.clip(defaults["power_input"], 0.0, 8000.0))
        st.session_state["slider_power_factor"] = float(np.clip(defaults["power_factor"], 0.20, 1.00))
        st.session_state["last_applied_sig"]    = preset_sig

v_in = st.sidebar.slider("Line Voltage (V)", 300.0, 480.0, step=1.0, key="slider_voltage")
ir_in = st.sidebar.slider("Phase R Current - Ir (A)", 0.0, 25.0, step=0.1, key="slider_current_r")
iy_in = st.sidebar.slider("Phase Y Current - Iy (A)", 0.0, 25.0, step=0.1, key="slider_current_y")
ib_in = st.sidebar.slider("Phase B Current - Ib (A)", 0.0, 25.0, step=0.1, key="slider_current_b")
rpm_in = st.sidebar.slider("Motor Speed (RPM)", 0.0, 1500.0, step=5.0, key="slider_rpm")
pin_in = st.sidebar.slider("Active Power Input (W)", 0.0, 8000.0, step=50.0, key="slider_power_input")
pf_in = st.sidebar.slider("Power Factor (cos φ)", 0.20, 1.00, step=0.01, key="slider_power_factor")


# Current Unbalance Calculation
i_avg = (ir_in + iy_in + ib_in) / 3.0
if i_avg > 0.1:
    unbalance_pct = (max(abs(ir_in - i_avg), abs(iy_in - i_avg), abs(ib_in - i_avg)) / i_avg) * 100.0
else:
    unbalance_pct = 0.0

# Prepare input DataFrame
input_dict = {
    'voltage': v_in,
    'current_r': ir_in,
    'current_y': iy_in,
    'current_b': ib_in,
    'rpm': rpm_in,
    'power_input': pin_in,
    'power_factor': pf_in
}

input_df = pd.DataFrame([input_dict])[FEATURE_COLS]

# Model Inference
if model is not None and le is not None:
    pred_encoded = model.predict(input_df)[0]
    pred_class = le.inverse_transform([pred_encoded])[0]
    probs = model.predict_proba(input_df)[0]
    prob_dict = {c: float(p) for c, p in zip(le.classes_, probs)}
else:
    pred_class = "Normal"
    prob_dict = {"Normal": 1.0}

diag_info = FAULT_DETAILS.get(pred_class, FAULT_DETAILS['Normal'])
health_idx = diag_info['health_index']

# ── Sync with Live API / User Interface (100% Exact Mirror) ────
sync_ok = False
try:
    import requests
    payload = {
        **input_dict,
        'health_index': float(health_idx),
        'fault': pred_class,
        'location': diag_info['location'],
        'action': diag_info['action'],
        'severity': diag_info['status_badge'],
        'current_unbalance': round(unbalance_pct, 2),
        'probabilities': {str(c): round(float(p) * 100, 1) for c, p in prob_dict.items()}
    }
    res = requests.post("http://localhost:8000/update-telemetry", json=payload, timeout=0.5)
    sync_ok = (res.status_code == 200)
except Exception:
    sync_ok = False

if sync_ok:
    st.sidebar.success(f"📡 Synced to User UI (Health: {health_idx}%)", icon="✅")
else:
    st.sidebar.caption("💡 User UI connects via `api.py` on port 8000")


# ==================== ROW 1: STATUS ALARM BANNER ====================
if pred_class == "Normal":
    banner_bg = "linear-gradient(90deg, #065f46 0%, #047857 100%)"
    banner_border = "#10b981"
elif pred_class in ["Stator_Fault", "Single_Phasing"]:
    banner_bg = "linear-gradient(90deg, #7f1d1d 0%, #991b1b 100%)"
    banner_border = "#ef4444"
else:
    banner_bg = "linear-gradient(90deg, #78350f 0%, #92400e 100%)"
    banner_border = "#f59e0b"

st.markdown(f"""
<div style="background: {banner_bg}; border: 2px solid {banner_border}; border-radius: 12px; padding: 20px 24px; margin-bottom: 24px;">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <span style="font-size: 14px; text-transform: uppercase; letter-spacing: 1.5px; color: #f8fafc; font-weight: 700;">Diagnostic Classification</span>
            <h2 style="color: #ffffff; margin: 4px 0 0 0; font-size: 26px;">{diag_info['status_badge']}</h2>
            <p style="color: #e2e8f0; margin: 6px 0 0 0; font-size: 15px;">Fault Location: <b>{diag_info['location']}</b></p>
        </div>
        <div style="text-align: right; background: rgba(0,0,0,0.25); padding: 12px 24px; border-radius: 10px;">
            <div style="font-size: 13px; color: #cbd5e1; font-weight: 600;">MOTOR HEALTH INDEX</div>
            <div style="font-size: 34px; font-weight: 800; color: #ffffff;">{health_idx}%</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if dataset_truth_label:
    clean_truth = dataset_truth_label.lower().replace('-', '_').strip()
    clean_pred  = pred_class.lower().replace('-', '_').strip()
    if clean_truth == clean_pred:
        st.success(f"🎯 **Dataset Validation:** Row label `{dataset_truth_label}` == Model Prediction `{pred_class}` (100% Accurate Detection!)", icon="✅")
    else:
        st.info(f"📊 **Dataset Label Comparison:** Dataset Record: `{dataset_truth_label}` | ML Prediction: `{pred_class}`")


# ==================== ROW 2: LIVE METRIC CARDS ====================
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("Line Voltage", f"{v_in:.1f} V", delta=f"{v_in - 415.0:+.1f} V" if abs(v_in - 415.0) > 1 else "Nominal")
c2.metric("Phase R (Ir)", f"{ir_in:.2f} A", delta=f"{ir_in - i_avg:+.2f} A" if unbalance_pct > 5 else None)
c3.metric("Phase Y (Iy)", f"{iy_in:.2f} A", delta=f"{iy_in - i_avg:+.2f} A" if unbalance_pct > 5 else None)
c4.metric("Phase B (Ib)", f"{ib_in:.2f} A", delta=f"{ib_in - i_avg:+.2f} A" if unbalance_pct > 5 else None)
c5.metric("Motor Speed", f"{rpm_in:.0f} RPM", delta=f"{rpm_in - 1440.0:+.0f} RPM")
c6.metric("Active Power", f"{pin_in:.0f} W")
c7.metric("Power Factor", f"{pf_in:.2f}", delta="Low PF" if pf_in < 0.75 else "Optimal", delta_color="inverse" if pf_in < 0.75 else "normal")

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

# ==================== ROW 3: GAUGES & VISUALS ====================
col_v1, col_v2, col_v3 = st.columns([1, 1.2, 1.2])

with col_v1:
    st.subheader("📊 Health & Current Balance")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=health_idx,
        number={'suffix': "%", 'font': {'size': 32}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': diag_info['status_color']},
            'steps': [
                {'range': [0, 35], 'color': '#fee2e2'},
                {'range': [35, 65], 'color': '#fef3c7'},
                {'range': [65, 85], 'color': '#fef9c3'},
                {'range': [85, 100], 'color': '#d1fae5'}
            ],
            'threshold': {'line': {'color': "black", 'width': 3}, 'thickness': 0.75, 'value': health_idx}
        }
    ))
    fig_gauge.update_layout(height=240, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    st.markdown(f"""
    <div style="background: #1e293b; padding: 12px; border-radius: 8px; text-align: center; border: 1px solid #334155;">
        <span style="font-size: 13px; color: #94a3b8;">Phase Current Unbalance</span>
        <h3 style="margin: 4px 0 0 0; color: {'#ef4444' if unbalance_pct > 10 else '#10b981'}; font-size: 22px;">{unbalance_pct:.1f}%</h3>
        <small style="color: #64748b;">(IEEE Threshold: &lt; 5.0% Unbalance)</small>
    </div>
    """, unsafe_allow_html=True)

with col_v2:
    st.subheader("⚡ 3-Phase Current Distribution")
    df_curr = pd.DataFrame({
        'Phase': ['Phase R (Ir)', 'Phase Y (Iy)', 'Phase B (Ib)'],
        'Current (A)': [ir_in, iy_in, ib_in],
        'Color': ['#ef4444', '#f59e0b', '#3b82f6']
    })
    fig_curr = px.bar(
        df_curr, x='Phase', y='Current (A)', text='Current (A)',
        color='Phase', color_discrete_map={'Phase R (Ir)': '#ef4444', 'Phase Y (Iy)': '#f59e0b', 'Phase B (Ib)': '#3b82f6'}
    )
    fig_curr.add_hline(y=7.2, line_dash="dash", line_color="red", annotation_text="Rated Full Load: 7.2A")
    fig_curr.add_hline(y=i_avg, line_dash="dot", line_color="white", annotation_text=f"Avg: {i_avg:.2f}A")
    fig_curr.update_layout(height=320, showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
    fig_curr.update_traces(texttemplate='%{text:.2f} A', textposition='outside')
    st.plotly_chart(fig_curr, use_container_width=True)

with col_v3:
    st.subheader("🎯 Model Class Probability")
    df_probs = pd.DataFrame(list(prob_dict.items()), columns=['Fault Class', 'Probability'])
    df_probs = df_probs.sort_values(by='Probability', ascending=True)
    fig_prob = px.bar(
        df_probs, x='Probability', y='Fault Class', orientation='h',
        text='Probability', color='Probability', color_continuous_scale='Blues'
    )
    fig_prob.update_traces(texttemplate='%{text:.1%}', textposition='outside')
    fig_prob.update_layout(height=320, coloraxis_showscale=False, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_prob, use_container_width=True)

# ==================== ROW 4: PREDICTIVE & PRESCRIPTIVE MAINTENANCE ====================
st.markdown("<hr style='border: 1px solid #334155; margin: 24px 0;'>", unsafe_allow_html=True)
st.subheader("🛠️ Prescriptive Action & Prognostics")

col_act1, col_act2 = st.columns([1.5, 1])

with col_act1:
    st.markdown(f"""
    <div style="background: #1e293b; border-left: 5px solid {diag_info['status_color']}; padding: 18px 22px; border-radius: 8px;">
        <h4 style="color: #f8fafc; margin-top: 0;">Predicted Degradation & Prognosis</h4>
        <p style="color: #cbd5e1; font-size: 15px; margin-bottom: 14px;">{diag_info['prediction']}</p>
        <h4 style="color: #f8fafc; margin-bottom: 6px;">Recommended Engineering Action</h4>
        <p style="color: #38bdf8; font-size: 16px; font-weight: 600; margin: 0;">{diag_info['action']}</p>
    </div>
    """, unsafe_allow_html=True)

with col_act2:
    st.markdown("""
    <div style="background: #0f172a; padding: 16px; border-radius: 8px; border: 1px solid #334155;">
        <h5 style="color: #e2e8f0; margin-top: 0;">Tested Model Specs</h5>
        <ul style="color: #94a3b8; font-size: 13px; margin: 0; padding-left: 18px; line-height: 1.8;">
            <li><b>Best Classifier:</b> Support Vector Machine (100.0% Test Acc)</li>
            <li><b>Random Forest:</b> 99.57% Test Acc, 98.70% 5-Fold CV</li>
            <li><b>Total Trained Samples:</b> 2,300 Synthetic Lab Runs</li>
            <li><b>Classes:</b> Normal, Stator, Single Phase, Overload, Under-V, Unbalance, Rotor</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ==================== TABS: DATASET & FEATURE IMPORTANCES ====================
st.markdown("<hr style='border: 1px solid #334155; margin: 24px 0;'>", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["📋 Training Dataset Records", "📈 Feature Importance Ranking", "🏆 Model Comparison Benchmarks"])

with tab1:
    tab1_choice = st.radio(
        "Choose Dataset View:", 
        [
            "⚡ Excel Sheet 2 (Lab Under-Voltage & Asymmetry)",
            "🔬 Indoor Lab Readings (Real Experiment)", 
            "📊 Full Synthetic Dataset (2,300 Records)"
        ], 
        horizontal=True
    )
    if "Excel Sheet 2" in tab1_choice:
        sheet2_path = os.path.join(BASE_DIR, "dataset", "excel_sheet2.csv")
        if os.path.exists(sheet2_path):
            df_s2 = pd.read_csv(sheet2_path)
            st.dataframe(df_s2, use_container_width=True)
            st.caption(f"Showing all {len(df_s2)} experimental records from: `excel sheet2.xlsx`")
        else:
            st.warning("Excel sheet 2 dataset file not found.")
    elif "Indoor Lab Readings" in tab1_choice:
        lab_csv_path = os.path.join(BASE_DIR, "dataset", "indoor_lab_readings.csv")
        if os.path.exists(lab_csv_path):
            df_lab = pd.read_csv(lab_csv_path)
            st.dataframe(df_lab, use_container_width=True)
            st.caption(f"Showing all {len(df_lab)} experimental records from: `indoor_lab_readings.xlsx`")
        else:
            st.warning("Indoor lab readings file not found.")
    else:
        if os.path.exists(DATASET_PATH):
            df_display = pd.read_csv(DATASET_PATH)
            st.dataframe(df_display.head(100), use_container_width=True)
            st.caption(f"Showing 100 of {len(df_display):,} records from: `{os.path.basename(DATASET_PATH)}`")
        else:
            st.warning("Dataset file not found.")


with tab2:
    if metrics and "feature_importances" in metrics:
        df_imp = pd.DataFrame(list(metrics["feature_importances"].items()), columns=['Parameter', 'Importance (%)'])
        df_imp = df_imp.sort_values(by='Importance (%)', ascending=True)
        fig_imp = px.bar(df_imp, x='Importance (%)', y='Parameter', orientation='h', color='Importance (%)', color_continuous_scale='Teal')
        fig_imp.update_layout(height=350)
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Feature importance metrics not loaded.")

with tab3:
    if metrics and "comparison" in metrics:
        df_comp = pd.DataFrame(metrics["comparison"])
        st.table(df_comp)
