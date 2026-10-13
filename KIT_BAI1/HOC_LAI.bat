@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PY=C:\Users\Thanh Hue\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
rem Khong cho thu vien nao tu len mang (moi mo hinh da nam san trong may)
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
echo ============================================================
echo  HOC LAI sau khi sua code (giu ket qua doc anh, khong phai doc anh lai)
echo ============================================================
if not exist runs\du_lieu_chungket.txt (echo Chua chay CHAY_BAI1.bat lan nao. & pause & exit /b)
set /p DL=<runs\du_lieu_chungket.txt
del /q runs\chungket\outputs\*.pkl runs\chungket\outputs\*.npz runs\chungket\outputs\chon.json runs\chungket\outputs\greedy.json runs\chungket\outputs\sent_inventory.json 2>nul
del /q runs\chungket\cache\nen_feats_*.pkl runs\chungket\cache\p2_*.pkl runs\chungket\cache\p2_*.pt runs\chungket\cache\p2_*.json runs\chungket\cache\nang_cao_orders.json 2>nul
"%PY%" tu_dong.py "%DL%" chungket
echo.
echo File nop: %~dp0runs\chungket\predictions.json
pause
