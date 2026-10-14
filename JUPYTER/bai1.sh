#!/bin/bash
# BÀI 1 TRÊN JUPYTERHUB:   bash ~/B1/JUPYTER/bai1.sh <thư mục dữ liệu>
# (thư mục dữ liệu: chuột phải thư mục trong cây file bên trái -> Copy Path -> dán vào sau lệnh)
cd "$(dirname "$0")/../KIT_BAI1" || exit 1
[ -z "$1" ] && { echo "Thiếu thư mục dữ liệu. Ví dụ: bash ~/B1/JUPYTER/bai1.sh data/bai1"; exit 1; }
D="$1"; [ -e "$D" ] || [ ! -e "$HOME/$D" ] || D="$HOME/$D"
mkdir -p runs; echo "$D" > runs/du_lieu_chungket.txt
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUTF8=1
python -u kiem_tra_de.py "$D"
python -u tu_dong.py "$D" chungket 2>&1 | tee -a runs/man_hinh_chungket.txt
echo; echo "File nộp: $(pwd)/runs/chungket/predictions.json   (chuột phải -> Download)"
