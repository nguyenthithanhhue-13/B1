@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PY=C:\Users\Thanh Hue\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
echo ============================================================
echo  TAP DUOT voi bo du lieu nho (khoang 10 phut)
echo ============================================================
set PROCS=3
"%PY%" kiem_tra_de.py "..\du_lieu_tap_duot"
echo.
"%PY%" tu_dong.py "..\du_lieu_tap_duot" tap1
pause
