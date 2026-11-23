"""
api.py — FastAPI Backend for 3-Phase Induction Motor Fault Detection
Bridges the ML model to any frontend (user UI, mobile app, etc.)
Run: python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import numpy as np
import os
import json
import time
from pathlib import Path

# ─── Load Model & Encoder ────────────────────────────────────────────────────
BASE = Path(__file__).parent
model_path  = BASE / "models" / "best_model.joblib"
encoder_path = BASE / "models" / "label_encoder.joblib"
metrics_path = BASE / "models" / "evaluation_metrics.json"

model   = joblib.load(model_path)
encoder = joblib.load(encoder_path)

with open(metrics_path) as f:
    metrics = json.load(f)

# ─── Fault Metadata Maps (Synchronized with app.py & prediction.py) ────────
HEALTH_INDEX = {
    "Normal":           100,
    "Under_Voltage":     68,
    "Voltage_Unbalance": 62,
    "Overload":          45,
    "Rotor_Fault":       38,
    "Stator_Fault":      28,
    "Single_Phasing":    12,
}

SEVERITY = {
    "Normal":           "NORMAL",
    "Rotor_Fault":      "WARNING",
    "Voltage_Unbalance":"WARNING",
    "Under_Voltage":    "WARNING",
    "Overload":         "CRITICAL",
    "Stator_Fault":     "CRITICAL",
    "Single_Phasing":   "EMERGENCY",
}

LOCATION = {
    "Normal":           "Motor Windings & Grid Healthy",
    "Rotor_Fault":      "Rotor Squirrel Cage / Air-Gap",
    "Voltage_Unbalance":"Power Supply Grid / Transformer",
    "Under_Voltage":    "Incoming Line Feeder",
    "Overload":         "Mechanical Drive / Shaft Load",
    "Stator_Fault":     "Stator Winding Insulation (Phase Asymmetry)",
    "Single_Phasing":   "Supply Line / Contactor (Open Phase)",
}

ACTIONS = {
    "Normal":           "Motor operating within normal electrical and thermal limits. Continue regular operation schedule.",
    "Rotor_Fault":      "Schedule Motor Current Signature Analysis (MCSA) inspection. Inspect rotor cage bars during next shutdown.",
    "Voltage_Unbalance":"Check line supply transformer taps, rebalance single-phase facility loads, and verify utility voltage balance.",
    "Under_Voltage":    "Regulate bus voltage to nominal (415 V). Check for undersized feeder cables causing excessive voltage drop.",
    "Overload":         "Reduce mechanical shaft loading or inspect driven equipment for mechanical binding. Check cooling airflow.",
    "Stator_Fault":     "Urgent machine shutdown recommended. Perform insulation resistance (Megger) and surge test on stator windings.",
    "Single_Phasing":   "Trip motor breaker immediately! Check upstream supply fuses, contactor poles, and terminal block.",
}

COLORS = {
    "Normal":           "#00C851",
    "Under_Voltage":    "#FFBB33",
    "Voltage_Unbalance":"#FF8800",
    "Overload":         "#ea580c",
    "Rotor_Fault":      "#f97316",
    "Stator_Fault":     "#ef4444",
    "Single_Phasing":   "#991b1b",
}

ICONS = {
    "Normal":           "🟢",
    "Under_Voltage":    "🟡",
    "Voltage_Unbalance":"🟡",
    "Overload":         "🟠",
    "Rotor_Fault":      "🟠",
    "Stator_Fault":     "🔴",
    "Single_Phasing":   "🚨",
}


# ─── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Induction Motor Fault Detection API",
    description="Live health monitoring API for 3-Phase Induction Motor",
    version="2.0.0"
)

# Allow all origins (so user_ui HTML can call this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve user_ui folder as static files
ui_path = BASE / "user_ui"
if ui_path.exists():
    app.mount("/ui", StaticFiles(directory=str(ui_path), html=True), name="ui")

# ─── Pydantic Input Schema ────────────────────────────────────────────────────
class MotorReading(BaseModel):
    voltage:      float = 415.0   # Volts
    current_r:    float = 7.2     # Phase R current (A)
    current_y:    float = 7.2     # Phase Y current (A)
    current_b:    float = 7.2     # Phase B current (A)
    rpm:          float = 1440.0  # Speed (RPM)
    power_input:  float = 3700.0  # Power Input (W)
    power_factor: float = 0.85    # Power Factor (0–1)
    health_index: float | None = None
    fault:        str | None = None
    location:     str | None = None
    action:       str | None = None
    severity:     str | None = None
    current_unbalance: float | None = None
    probabilities: dict | None = None


# ─── Helper: Current Unbalance ────────────────────────────────────────────────
def calc_unbalance(ir, iy, ib):
    avg = (ir + iy + ib) / 3
    if avg == 0:
        return 0.0
    return round(max(abs(ir - avg), abs(iy - avg), abs(ib - avg)) / avg * 100, 2)

# ─── Shared Live State (Synchronizes Developer Dashboard & User UI) ────────
_shared_state = {
    "fault": "Normal",
    "severity": "NORMAL",
    "health_index": 100,
    "color": "#00C851",
    "icon": "✅",
    "location": "Motor Windings & Grid Healthy",
    "action": "Motor is healthy. Continue normal operation. Schedule next inspection in 30 days.",
    "current_unbalance": 0.5,
    "probabilities": {"Normal": 100.0},
    "inputs": {
        "voltage": 415.0,
        "current_r": 5.25,
        "current_y": 5.20,
        "current_b": 5.28,
        "rpm": 1445.0,
        "power_input": 3180.0,
        "power_factor": 0.85
    },
    "timestamp": time.time(),
    "version": 1
}

def compute_prediction_result(reading: MotorReading):
    features = np.array([[
        reading.voltage,
        reading.current_r,
        reading.current_y,
        reading.current_b,
        reading.rpm,
        reading.power_input,
        reading.power_factor,
    ]])

    pred_encoded  = model.predict(features)[0]
    pred_proba    = model.predict_proba(features)[0] if hasattr(model, "predict_proba") else []
    fault_label   = encoder.inverse_transform([pred_encoded])[0]

    unbalance     = calc_unbalance(reading.current_r, reading.current_y, reading.current_b)
    health_index  = HEALTH_INDEX.get(fault_label, 50)

    if len(pred_proba) > 0:
        classes = encoder.inverse_transform(list(range(len(pred_proba))))
        prob_map = {str(c): round(float(p) * 100, 1) for c, p in zip(classes, pred_proba)}
    else:
        prob_map = {fault_label: 100.0}

    return {
        "fault":           fault_label,
        "severity":        SEVERITY.get(fault_label, "UNKNOWN"),
        "health_index":    health_index,
        "color":           COLORS.get(fault_label, "#888"),
        "icon":            ICONS.get(fault_label, "❓"),
        "location":        LOCATION.get(fault_label, "Unknown"),
        "action":          ACTIONS.get(fault_label, "Contact engineer."),
        "current_unbalance": unbalance,
        "probabilities":   prob_map,
        "inputs": {
            "voltage":      reading.voltage,
            "current_r":    reading.current_r,
            "current_y":    reading.current_y,
            "current_b":    reading.current_b,
            "rpm":          reading.rpm,
            "power_input":  reading.power_input,
            "power_factor": reading.power_factor,
        },
        "timestamp": time.time(),
        "version": _shared_state.get("version", 1) + 1
    }

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return FileResponse(str(BASE / "user_ui" / "index.html"))

import socket

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

@app.get("/system-info")
def system_info():
    ip = get_local_ip()
    return {
        "local_ip": ip,
        "ui_url_local": f"http://localhost:8000",
        "ui_url_network": f"http://{ip}:8000",
        "status": "online"
    }

@app.get("/health")
def api_health():
    return {"status": "online", "local_ip": get_local_ip(), "timestamp": time.time()}

@app.get("/live-telemetry")
def get_live_telemetry():
    """Polled by User UI to get real-time parameters set by Developer Dashboard."""
    return _shared_state

@app.post("/update-telemetry")
def update_telemetry(reading: MotorReading):
    """Called by app.py (Streamlit) whenever developer moves sliders or changes presets."""
    global _shared_state
    result = compute_prediction_result(reading)
    
    # Apply exact attributes from developer dashboard for 100% precision
    if reading.health_index is not None:
        result["health_index"] = reading.health_index
    if reading.fault is not None:
        result["fault"] = reading.fault
        result["color"] = COLORS.get(reading.fault, result["color"])
        result["icon"] = ICONS.get(reading.fault, result["icon"])
    if reading.location is not None:
        result["location"] = reading.location
    if reading.action is not None:
        result["action"] = reading.action
    if reading.severity is not None:
        result["severity"] = reading.severity
    if reading.current_unbalance is not None:
        result["current_unbalance"] = reading.current_unbalance
    if reading.probabilities is not None:
        result["probabilities"] = reading.probabilities

    _shared_state = result
    return {"status": "ok", "state": _shared_state}


@app.post("/predict")
def predict(reading: MotorReading):
    global _shared_state
    result = compute_prediction_result(reading)
    _shared_state = result
    return result

@app.get("/status")
def get_status():
    """Returns the last prediction result for live polling."""
    return _shared_state


@app.get("/model-info")
def model_info():
    return {
        "best_model":    metrics.get("best_model", "SVM"),
        "test_accuracy": metrics.get("test_accuracy", 1.0),
        "cv_accuracy":   metrics.get("cv_mean", 0.9995),
        "fault_classes": list(HEALTH_INDEX.keys()),
        "features":      ["voltage", "current_r", "current_y", "current_b", "rpm", "power_input", "power_factor"],
        "motor_specs": {
            "type":        "3-Phase Squirrel Cage Induction Motor",
            "power":       "3.7 kW",
            "voltage":     "415 V",
            "frequency":   "50 Hz",
            "rated_rpm":   1440,
            "rated_current": "7.2 A"
        }
    }
