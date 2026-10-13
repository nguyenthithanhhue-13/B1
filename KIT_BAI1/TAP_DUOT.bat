@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PY=C:\Users\Thanh Hue\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
rem Khong cho thu vien nao tu len mang (moi mo hinh da nam san trong may)
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
echo ============================================================
echo  TAP DUOT voi bo du lieu nho (khoang 10 phut)
echo ============================================================
set PROCS=3
"%PY%" kiem_tra_de.py "..\du_lieu_tap_duot"
echo.
"%PY%" tu_dong.py "..\du_lieu_tap_duot" tap1
pause
