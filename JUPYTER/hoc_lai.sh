#!/bin/bash
# HỌC LẠI sau khi sửa code (giữ kết quả đọc ảnh):   bash ~/B1/JUPYTER/hoc_lai.sh
cd "$(dirname "$0")/../KIT_BAI1" || exit 1
[ -f runs/du_lieu_chungket.txt ] || { echo "Chưa chạy bai1.sh lần nào."; exit 1; }
D="$(cat runs/du_lieu_chungket.txt)"
rm -f runs/chungket/outputs/*.pkl runs/chungket/outputs/*.npz runs/chungket/outputs/chon.json runs/chungket/outputs/greedy.json runs/chungket/outputs/sent_inventory.json
rm -f runs/chungket/cache/nen_feats_*.pkl runs/chungket/cache/p2_*.pkl runs/chungket/cache/p2_*.pt runs/chungket/cache/p2_*.json runs/chungket/cache/nang_cao_orders.json
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUTF8=1
python -u tu_dong.py "$D" chungket 2>&1 | tee -a runs/man_hinh_chungket.txt
echo; echo "File nộp: $(pwd)/runs/chungket/predictions.json   |  sổ điểm: runs/chungket/nhat_ky_diem.txt"
