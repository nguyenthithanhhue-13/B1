@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================================
echo  CAI DAT tren may KHAC (khong phai may Hue). Can MANG, mat 20-40 phut.
echo  Can co Python 3.13 (python.org) - khi cai nho tick 'Add python.exe to PATH'.
echo ============================================================
python --version || (echo Chua co Python. Cai tu python.org roi chay lai file nay. & pause & exit /b)
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-deps easyocr==1.7.2
echo.
echo Tai cac mo hinh (doc anh, dich, YOLO, OCR, doc cau)...
python tai_mo_hinh_cv.py
python -c "from transformers import AutoModel, AutoTokenizer as T; [ (AutoModel.from_pretrained(m), T.from_pretrained(m)) for m in ['intfloat/multilingual-e5-small','bkai-foundation-models/vietnamese-bi-encoder','sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2','Helsinki-NLP/opus-mt-zh-vi','Helsinki-NLP/opus-mt-en-vi','Helsinki-NLP/opus-mt-vi-en','Helsinki-NLP/opus-mt-en-zh','facebook/nllb-200-distilled-600M'] ]; print('[ok]  mo hinh doc cau + dich')"
echo.
echo XONG. Neu dong nao ghi [LOI] thi chay lai file nay.
pause
