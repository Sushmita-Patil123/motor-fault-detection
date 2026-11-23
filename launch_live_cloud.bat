@echo off
title 3-Phase Induction Motor - Live Cloud System
color 0B
echo.
echo ======================================================================
echo    3-Phase Induction Motor Fault Detection - Cloudflare Live Host
echo ======================================================================
echo.
echo [1/3] Starting FastAPI Backend on Port 8000...
cd /d C:\FAULT_DETECTION
start "Motor Fault API - Port 8000" cmd /k "C:\Users\prart\Python311\python.exe -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload"

echo [2/3] Waiting 4 seconds for API to initialize...
timeout /t 4 /nobreak >nul

echo [3/3] Starting Cloudflare Tunnel for Worldwide Access...
echo.
echo ======================================================================
echo   A public HTTPS link (https://*.trycloudflare.com) will appear below.
echo   Share this link to open the LIVE health dashboard on ANY device!
echo ======================================================================
echo.

C:\FAULT_DETECTION\cloudflared.exe tunnel --url http://localhost:8000
