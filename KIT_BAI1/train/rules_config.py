"""Bước 4 (luật chính xác): gộp các luật đã tìm thành file cấu hình cho src/rules.py.
Cách chạy:  SUF=_tv THR=0.95 PARTIAL=7,8 python train/rules_config.py outputs/rules_final.json <tên>:<KEY>:<CELL> ...
  <tên> là số robot, hoặc "r@KEY" cho bộ luật phụ dùng để bỏ phiếu (vd 6@nwf:nwf:-).
  Với robot trong PARTIAL, nhóm có độ khớp train < THR được đánh dấu bỏ (dùng mô hình học thay luật).
"""

import collections
import json
import os
import sys

sys.path.insert(0, "src")
sys.path.insert(0, "train")
from lib import load
import rules

out = sys.argv[1]
suffix = os.environ.get("SUF", "")
thr = float(os.environ.get("THR", "0"))
partial = set(os.environ.get("PARTIAL", "").split(",")) - {""}

cfg = {}
for spec in sys.argv[2:]:
    name, key, cell = spec.split(":")
    robot = name.split("@")[0]
    groups = json.load(open(f"cache/p3_search_r{robot}_{key}_{cell}{suffix}.json", encoding="utf-8"))
    cfg[name] = {"key": key, "cell": cell, "groups": groups}


def predict(x, r):
    return rules.predict(
        cfg, int(r), x["w"], x["legs"], x["s"]["style"] == "night", x["m"]["urgent"], x["m"]["fragile"]
    )


if thr > 0 and partial:
    train = load("train")
    for r in partial & set(cfg):
        acc = collections.defaultdict(lambda: [0, 0])
        for x in train:
            m = x["m"]
            g = str(
                rules.flag_tuple(
                    cfg[r]["key"],
                    x["s"]["style"] == "night",
                    x["w"]["rain"],
                    m["urgent"],
                    m["fragile"],
                    bool(m["via"]),
                    m["goal"],
                )
            )
            acc[g][0] += predict(x, r) == x["y"][int(r)]
            acc[g][1] += 1
        for g, G in cfg[r]["groups"].items():
            a, n = acc[g]
            G["skip"] = a / max(1, n) < thr
        print(f"R{r} bỏ nhóm:", [g for g, G in cfg[r]["groups"].items() if G["skip"]])

json.dump(cfg, open(out, "w"), ensure_ascii=False)
val = load("validation")
for r in sorted((k for k in cfg if "@" not in k), key=int):
    ok = sum(predict(x, r) == x["y"][int(r)] for x in val)
    print(f"R{r} validation {ok}/{len(val)} = {ok / len(val):.3f}")
