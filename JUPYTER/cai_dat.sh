#!/bin/bash
# CÀI ĐẶT TRÊN JUPYTERHUB (chạy một lần):   bash ~/B1/JUPYTER/cai_dat.sh
# - kiểm tra Python / GPU
# - cài thư viện còn thiếu (không đụng tới torch có sẵn của máy chủ)
# - nếu có file ~/mo_hinh_*.zip (mang từ máy nhà lên): giải nén vào đúng chỗ
# - nếu máy chủ có mạng: tải nốt các mô hình còn thiếu
cd "$(dirname "$0")/.."
B1="$(pwd)"
echo "=================== 1. Python và GPU ==================="
python --version
python -c "import torch; print('torch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'KHÔNG CÓ')" 2>/dev/null || {
  echo "chưa có torch -> cài"; pip install -q torch torchvision; }
nproc 2>/dev/null | sed 's/^/số nhân CPU: /'; free -g 2>/dev/null | head -2

echo "=================== 2. Cài thư viện ==================="
pip install -q numpy scipy scikit-learn onnxruntime pillow tokenizers huggingface_hub transformers sentence-transformers \
    sentencepiece sacremoses sacrebleu pandas openpyxl lightgbm soundfile matplotlib seaborn ultralytics easyocr 2>&1 | grep -v "already satisfied" | tail -5
python -c "import cv2" 2>/dev/null || {
  echo "opencv thiếu thư viện đồ họa của máy chủ -> đổi sang bản headless"
  pip uninstall -y -q opencv-python opencv-python-headless; pip install -q opencv-python-headless; }
python -c "import numpy, cv2, sklearn, onnxruntime, lightgbm, transformers, ultralytics, easyocr; print('[ok] thư viện đủ')" || echo "[LỖI] thiếu thư viện - xem dòng lỗi ở trên"

echo "=================== 3. Mô hình mang từ nhà (nếu có) ==================="
for z in ~/mo_hinh_*.zip "$B1"/mo_hinh_*.zip; do
  [ -f "$z" ] || continue
  echo "giải nén $z ..."; python -m zipfile -e "$z" ~ && echo "[ok] $z"
done

echo "=================== 4. Máy chủ có mạng không? ==================="
if python -c "import urllib.request; urllib.request.urlopen('https://huggingface.co', timeout=8)" 2>/dev/null; then
  echo "CÓ MẠNG -> tải các mô hình còn thiếu (lần đầu khoảng 5-15 phút)"
  python "$B1/KIT_BAI2/tai_mo_hinh_cv.py" $1
else
  echo "KHÔNG CÓ MẠNG -> chỉ dùng được mô hình đã mang lên (mo_hinh_*.zip). Bài 1 không cần mạng."
fi

echo "=================== 5. Kiểm tra bài 1 ==================="
M="$B1/KIT_BAI1/outputs/e5_small/onnx/model.onnx"
S=$(stat -c %s "$M" 2>/dev/null || echo 0)
if [ "$S" -lt 100000000 ]; then      # git clone không có git-lfs: file chỉ là "con trỏ" vài trăm byte -> tải lại từ Hugging Face
  echo "thiếu mô hình đọc câu của bài 1 -> tải lại (470 MB) ..."
  rm -f "$M"; (cd "$B1/KIT_BAI1" && env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE python src/embed.py download)
  S=$(stat -c %s "$M" 2>/dev/null || echo 0)
fi
if [ "$S" -gt 100000000 ]; then echo "[ok] mô hình đọc câu của bài 1 có đủ ($((S/1000000)) MB)"; else
  echo "[LỖI] không tải được KIT_BAI1/outputs/e5_small/onnx/model.onnx (phải ~470 MB). Upload kit.zip thay vì git clone."; fi
chmod +x "$B1"/JUPYTER/*.sh 2>/dev/null
echo "XONG CÀI ĐẶT. Chạy bài 1:  bash ~/B1/JUPYTER/bai1.sh <thư mục dữ liệu>"
