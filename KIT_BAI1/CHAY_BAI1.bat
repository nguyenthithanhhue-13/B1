@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PY=C:\Users\Thanh Hue\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
rem Khong cho thu vien nao tu len mang (moi mo hinh da nam san trong may)
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
echo ============================================================
echo  BAI 1 - chay tu dong. KHONG tat cua so nay cho toi khi thay "XONG TAT CA".
echo ============================================================
if not "%~1"=="" (set "DL=%~1") else (set /p "DL=Keo tha THU MUC BAI MOI vao day roi bam Enter: ")
set "DL=%DL:"=%"
if not exist runs mkdir runs
echo %DL%> runs\du_lieu_chungket.txt
"%PY%" kiem_tra_de.py "%DL%"
echo.
"%PY%" tu_dong.py "%DL%" chungket
echo.
echo File nop: %~dp0runs\chungket\predictions.json
pause
