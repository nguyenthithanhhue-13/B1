"""Bước 2 (luật chính xác): đổi mặt Pareto sang ma trận đặc trưng cho bộ tìm luật.
Cách chạy:  python train/rules_pareto.py      Đọc: cache/p2_paretotu_*.pkl.  Ghi: cache/p3_enum5p_<split>_<0|1>.pkl.
"""

import pickle

import numpy as np

for split in ("train", "validation"):
    for legged in (0, 1):
        fronts = pickle.load(open(f"cache/p2_paretotu_{split}_{legged}.pkl", "rb"))
        out = []
        for fr in fronts:
            scene = {}
            for d, (rel, F) in fr.items():
                F = np.asarray(F)
                G = np.zeros((len(F), 31), np.int16)
                G[:, 0:8] = F[:, 0:8]
                G[:, 9:19] = F[:, 9:19]
                if rel != "S":
                    G[:, 19 + "RLB".index(rel)] = 1
                scene[d] = (G, np.ones(len(F), np.int64))
            out.append(scene)
        pickle.dump(out, open(f"cache/p3_enum5p_{split}_{legged}.pkl", "wb"))
        print(split, legged, len(out))
