@echo off
title 3-Phase Induction Motor Fault Detection Dashboard
echo ======================================================================
echo Starting 3-Phase Induction Motor Electrical Diagnostic Dashboard...
echo ======================================================================
cd /d "C:\FAULT_DETECTION"
"C:\Users\prart\Python311\python.exe" -m streamlit run app.py
pause
