"""Học bộ phân loại HistGradientBoosting "ml4" (dự phòng cho robot 5 và 7 khi mô hình chính không cho kết quả).
Cách chạy:  python train/ml4_train.py <trainonly|final> 5 7
Ghi: cache/p2_mlfeats_<split>.pkl (đặc trưng bước đi), outputs/strategy_ml4_<chế độ>.pkl.
"""

import json
import pickle
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, "train")
sys.path.insert(0, "src")
from lib import load
from common import CACHE, OUT
from strat_ml import ml4_row, move_feats


def qcfg_of(cfg):
    while "qcfg" not in cfg and "fallback" in cfg:
        cfg = cfg["fallback"]
    return cfg.get("qcfg")


def step_feats(split, scenes):
    path = CACHE / f"p2_mlfeats_{split}.pkl"
    if path.exists():
        return pickle.load(open(path, "rb"))[0]
    fs = [move_feats(x["w"], x["legs"], x["m"]["urgent"], x["m"]["fragile"], bool(x["m"]["via"])) for x in scenes]
    pickle.dump((fs, np.array([x["y"] for x in scenes])), open(path, "wb"))
    return fs


mode = sys.argv[1]
robots = [int(a) for a in sys.argv[2:]]
hybrid = json.load(
    open(OUT / ("strategy_hybrid.json" if mode == "trainonly" else "strategy_hybrid_final.json"), encoding="utf-8")
)
splits = ["train"] if mode == "trainonly" else ["train", "validation"]
data = {sp: load(sp) for sp in splits}
feats = {sp: step_feats(sp, data[sp]) for sp in splits}
models = {}
for r in robots:
    qcfg = qcfg_of(hybrid[str(r)])
    X, y = [], []
    for sp in splits:
        for x, f in zip(data[sp], feats[sp]):
            rows, valid = ml4_row(
                x["w"], x["legs"], r, qcfg, f, x["s"]["style"] == "night", x["m"]["urgent"], x["m"]["fragile"]
            )
            for d in range(4):
                if valid[d]:
                    X.append(rows[d])
                    y.append(int(x["y"][r] == d))
    models[r] = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=5, min_samples_leaf=40, l2_regularization=1.0, random_state=0
    ).fit(np.array(X), np.array(y))
    print("đã học R", r, len(y), flush=True)
pickle.dump(models, open(OUT / f"strategy_ml4_{mode}.pkl", "wb"))
