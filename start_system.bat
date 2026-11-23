@echo off
title 3-Phase Induction Motor Fault Detection System
color 0A
echo.
echo ============================================================
echo   3-Phase Induction Motor - Fault Detection System
echo ============================================================
echo.
echo [1] Starting FastAPI Backend (Port 8000)...
echo [2] Starting Developer Dashboard (Port 8501)...
echo.

cd /d C:\FAULT_DETECTION

:: Start FastAPI backend in a new window
start "Motor Fault API - Port 8000" cmd /k "C:\Users\prart\Python311\python.exe -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload"

:: Wait 3 seconds for API to start
timeout /t 3 /nobreak >nul

:: Start Developer Streamlit Dashboard in a new window
start "Developer Dashboard - Port 8501" cmd /k "C:\Users\prart\Python311\python.exe -m streamlit run app.py --server.port 8501"

:: Wait 3 seconds for Streamlit to start
timeout /t 3 /nobreak >nul

:: Open User Interface in browser
echo.
echo ============================================================
echo   SYSTEM STARTED!
echo.
echo   USER Interface       ->  http://localhost:8000/ui
echo   DEVELOPER Dashboard  ->  http://localhost:8501
echo   API Docs             ->  http://localhost:8000/docs
echo ============================================================
echo.

start "" "http://localhost:8000/ui"

echo Press any key to exit this window...
pause >nul
