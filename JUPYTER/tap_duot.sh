#!/bin/bash
# TẬP THỬ bài 1 trên bộ dữ liệu nhỏ (10-20 phút):   bash ~/B1/JUPYTER/tap_duot.sh
cd "$(dirname "$0")/../KIT_BAI1" || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUTF8=1 PROCS=${PROCS:-3}
python -u tu_dong.py ../du_lieu_tap_duot tap1
echo "Tập xong thì xóa:  rm -rf ~/B1/KIT_BAI1/runs/tap1"
