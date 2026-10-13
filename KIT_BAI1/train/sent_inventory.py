"""Tạo kho câu cờ chuẩn cho nlp3.restore_sentences: các câu gấp / không gấp / dễ vỡ / không dễ vỡ gặp >= 5 lần trong train + validation.
Câu gõ sai lúc dự đoán được sửa về câu chuẩn gần nhất trong kho (khoảng cách sửa chữ nhỏ).
Cách chạy:  python train/sent_inventory.py      Ghi: outputs/sent_inventory.json = [[dạng không dấu, dạng chuẩn có dấu], ...].
"""

import collections
import json
import re
import sys

sys.path.insert(0, "src")
import nlp2
from common import OUT, load_split
from nlp import dl_distance

C = collections.Counter()
for sp in ("train", "validation"):
    for s in load_split(sp)[2]:
        for x in re.split(r"(?<=[.!?])\s+", s["mission"]["text"]):
            b = x.strip().rstrip(".!?").strip()
            if b:
                C[b] += 1
pat = re.compile("|".join(f"(?:{p})" for p in (nlp2.URGENT_POS, nlp2.URGENT_NEG, nlp2.FRAGILE_POS, nlp2.FRAGILE_NEG)))
byk = collections.defaultdict(collections.Counter)
for b, n in C.items():
    k = re.sub(r"\s+", " ", nlp2._unaccent(b.lower())).strip()
    byk[k][b] += n
inv = []
tot = {k: sum(f.values()) for k, f in byk.items()}
for k, forms in byk.items():
    if tot[k] < 5 or len(k) < 6 or len(k.split()) > 10 or not pat.search(k):
        continue
    if re.search(r"nham|diem nhan|ghe|da roi|da giao|o do|giao (toi|den|o)", k):
        continue
    if any(tot[k2] > tot[k] and dl_distance(k, k2, 2) <= 2 for k2 in byk if k2 != k and abs(len(k2) - len(k)) <= 2):
        continue
    acc = [f for f, n in forms.most_common() if not f.isascii()]
    canon = acc[0] if acc else k
    canon = canon[0].upper() + canon[1:]
    inv.append([k, canon])
json.dump(inv, open(OUT / "sent_inventory.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
print(len(inv), "câu chuẩn")
