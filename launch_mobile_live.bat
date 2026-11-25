@echo off
title 3-Phase Motor - Live Mobile Tunnel
color 0A
echo.
echo ======================================================================
echo    3-Phase Induction Motor - Live Mobile Internet Link
echo ======================================================================
echo.

cd /d C:\FAULT_DETECTION

echo [1/2] Starting Motor Fault API on Port 8000...
start "Motor Fault API - Port 8000" cmd /k "C:\Users\prart\Python311\python.exe -m uvicorn api:app --host 0.0.0.0 --port 8000"

echo [2/2] Waiting 3 seconds for server to start...
timeout /t 3 /nobreak >nul

echo.
echo ======================================================================
echo   Connecting to Secure Public Internet Tunnel...
echo   Open the HTTPS link shown below on ANY mobile phone (4G/5G/Data)!
echo ======================================================================
echo.

ssh -o StrictHostKeyChecking=no -R 80:127.0.0.1:8000 nokey@localhost.run
pause
