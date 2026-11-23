import numpy as np
import pandas as pd
import os

# Seed for reproducibility
np.random.seed(42)

# Motor Nameplate Specs:
# 3-Phase Squirrel Cage Induction Motor
# Rated Power: 3.7 kW (5 HP), Rated Voltage: 415 V (Line-to-Line, 50 Hz, Star connected)
# Rated Full-Load Current: 7.2 A, Rated Speed: 1440 RPM (Synchronous Speed: 1500 RPM, 4 Poles)
# Rated Full-Load Power Factor: 0.84, Rated Efficiency: 86.5%

records = []

def generate_samples(fault_label, n_samples, v_range, ir_mult, iy_mult, ib_mult, rpm_range, pf_range, pin_mult=None, noise_std=0.015):
    for _ in range(n_samples):
        # Base voltage
        v = np.random.uniform(v_range[0], v_range[1])
        
        # Load fraction (from 0.25 to 1.15 under normal variations)
        load = np.random.uniform(0.3, 1.05)
        
        # Base healthy line current for this load level (A)
        # At no-load ~ 2.5 A (magnetizing), at full load ~ 7.2 A
        base_i = 2.4 + load * 4.8 + np.random.normal(0, 0.08)
        base_i = max(base_i, 1.8)
        
        # Phase currents with fault multipliers
        ir = base_i * ir_mult() * (1 + np.random.normal(0, noise_std))
        iy = base_i * iy_mult() * (1 + np.random.normal(0, noise_std))
        ib = base_i * ib_mult() * (1 + np.random.normal(0, noise_std))
        
        ir = max(0.0, ir)
        iy = max(0.0, iy)
        ib = max(0.0, ib)
        
        # Speed (RPM)
        base_rpm = np.random.uniform(rpm_range[0], rpm_range[1]) - (load - 0.5) * 25.0
        rpm = round(float(np.clip(base_rpm + np.random.normal(0, 4.0), 0.0, 1500.0)), 1)
        
        # Power Factor (cos phi)
        pf = float(np.clip(np.random.uniform(pf_range[0], pf_range[1]) + np.random.normal(0, 0.01), 0.15, 0.95))
        
        # Power Input (W) = sqrt(3) * V * I_avg * PF (or custom pin_mult for abnormal losses)
        i_avg = (ir + iy + ib) / 3.0
        if pin_mult is not None:
            pin = np.sqrt(3) * v * max(i_avg, 1.0) * pf * pin_mult()
        else:
            pin = np.sqrt(3) * v * i_avg * pf
        pin = round(float(max(pin + np.random.normal(0, 25.0), 100.0)), 1)
        
        # Derived diagnostic metric: Current Unbalance (%)
        if i_avg > 0.1:
            i_unb = (max(abs(ir - i_avg), abs(iy - i_avg), abs(ib - i_avg)) / i_avg) * 100.0
        else:
            i_unb = 0.0
            
        records.append({
            'voltage': round(float(v), 2),
            'current_r': round(float(ir), 2),
            'current_y': round(float(iy), 2),
            'current_b': round(float(ib), 2),
            'rpm': rpm,
            'power_input': pin,
            'power_factor': round(pf, 3),
            'current_unbalance_pct': round(float(i_unb), 2),
            'fault': fault_label
        })

# 1. Normal (Healthy) Operation: 400 samples
# Balanced currents (unbalance < 3%), normal PF (0.78 - 0.88), rated speed (1420 - 1465 RPM)
generate_samples(
    fault_label="Normal",
    n_samples=400,
    v_range=(410.0, 420.0),
    ir_mult=lambda: np.random.uniform(0.98, 1.02),
    iy_mult=lambda: np.random.uniform(0.98, 1.02),
    ib_mult=lambda: np.random.uniform(0.98, 1.02),
    rpm_range=(1435.0, 1465.0),
    pf_range=(0.80, 0.87)
)

# 2. Stator Inter-Turn Short Circuit: 350 samples
# Asymmetrical spike in one phase (e.g. Phase R or Phase Y draws 1.35x - 1.70x), PF drops (0.68 - 0.76)
generate_samples(
    fault_label="Stator_Fault",
    n_samples=350,
    v_range=(405.0, 418.0),
    ir_mult=lambda: np.random.uniform(1.38, 1.72),
    iy_mult=lambda: np.random.uniform(0.92, 1.04),
    ib_mult=lambda: np.random.uniform(0.92, 1.04),
    rpm_range=(1405.0, 1435.0),
    pf_range=(0.67, 0.77),
    pin_mult=lambda: np.random.uniform(1.15, 1.30)
)

# 3. Single Phasing (Phase Loss): 300 samples
# One phase drops to 0.0 A (e.g. Phase B), remaining 2 phases surge to 1.7x - 2.1x, RPM drops, PF drops
generate_samples(
    fault_label="Single_Phasing",
    n_samples=300,
    v_range=(400.0, 415.0),
    ir_mult=lambda: np.random.uniform(1.65, 2.05),
    iy_mult=lambda: np.random.uniform(1.65, 2.05),
    ib_mult=lambda: 0.0, # Phase B disconnected
    rpm_range=(1300.0, 1370.0),
    pf_range=(0.55, 0.68),
    pin_mult=lambda: np.random.uniform(1.20, 1.40)
)

# 4. Voltage Unbalance: 300 samples
# Supply voltage asymmetry (3% - 6%), leading to 15% - 25% current unbalance
generate_samples(
    fault_label="Voltage_Unbalance",
    n_samples=300,
    v_range=(395.0, 425.0),
    ir_mult=lambda: np.random.uniform(1.18, 1.32),
    iy_mult=lambda: np.random.uniform(0.95, 1.05),
    ib_mult=lambda: np.random.uniform(0.78, 0.88),
    rpm_range=(1415.0, 1445.0),
    pf_range=(0.72, 0.81)
)

# 5. Motor Overload (125% - 160% Rated Capacity): 350 samples
# All 3 phases rise symmetrically (1.25x - 1.65x), speed drops significantly, high active power
generate_samples(
    fault_label="Overload",
    n_samples=350,
    v_range=(405.0, 415.0),
    ir_mult=lambda: np.random.uniform(1.28, 1.65),
    iy_mult=lambda: np.random.uniform(1.28, 1.65),
    ib_mult=lambda: np.random.uniform(1.28, 1.65),
    rpm_range=(1340.0, 1395.0),
    pf_range=(0.84, 0.91),
    pin_mult=lambda: np.random.uniform(1.30, 1.65)
)

# 6. Under-Voltage Condition: 300 samples
# Voltage drops below 370 V (340V - 375V), current rises symmetrically, slight speed drop
generate_samples(
    fault_label="Under_Voltage",
    n_samples=300,
    v_range=(335.0, 372.0),
    ir_mult=lambda: np.random.uniform(1.12, 1.25),
    iy_mult=lambda: np.random.uniform(1.12, 1.25),
    ib_mult=lambda: np.random.uniform(1.12, 1.25),
    rpm_range=(1395.0, 1430.0),
    pf_range=(0.76, 0.83)
)

# 7. Rotor Fault (Broken Rotor Bar / Eccentricity): 300 samples
# Modulated current oscillations, lower PF, slight speed ripple
generate_samples(
    fault_label="Rotor_Fault",
    n_samples=300,
    v_range=(410.0, 420.0),
    ir_mult=lambda: np.random.uniform(1.08, 1.22),
    iy_mult=lambda: np.random.uniform(0.92, 1.05),
    ib_mult=lambda: np.random.uniform(1.02, 1.15),
    rpm_range=(1410.0, 1438.0),
    pf_range=(0.70, 0.78),
    pin_mult=lambda: np.random.uniform(1.08, 1.18)
)

df = pd.DataFrame(records)
# Shuffle dataset
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

out_csv = r"c:\FAULT_DETECTION\dataset\induction_motor_electrical_dataset.csv"
df.to_csv(out_csv, index=False)

print(f"Generated dataset successfully with {len(df)} total records.")
print("\nClass Distribution:")
print(df['fault'].value_counts())
print("\nFirst 5 Records:")
print(df.head())
