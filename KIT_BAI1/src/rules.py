"""Luật chiến thuật chính xác theo nhóm điều kiện: chi phí tuyến tính trên mặt Pareto các đường, hòa thì theo thứ tự hướng của ô
điều kiện; gồm cả các bộ luật phụ dùng để bỏ phiếu (khóa "r@KEY"). Thư viện; đọc outputs/rules_<chế độ>.json.
"""

import json
import numpy as np

from common import rel_turn
import hyper

REL = [[rel_turn(h, d) for d in range(4)] for h in range(4)]
NF = 37
GC = {"library": "B", "lecture": "B", "lab": "B", "dorm": "A", "canteen": "A"}


def feats(F, rel):
    F = np.asarray(F, np.float64)
    G = np.zeros((len(F), NF))
    G[:, 0:8] = F[:, 0:8]
    G[:, 9:19] = F[:, 9:19]
    if rel != "S":
        G[:, 19 + "RLB".index(rel)] = 1
    G[:, 31] = G[:, 0] - G[:, 2]
    G[:, 32] = G[:, 0] - G[:, 2] - G[:, 1]
    G[:, 33] = G[:, 5] + G[:, 19]
    G[:, 34] = G[:, 6] + G[:, 20]
    G[:, 35] = G[:, 7] + G[:, 21]
    G[:, 36] = G[:, 33] + G[:, 34] + G[:, 35]
    return G


def flag_tuple(key, night, rain, urgent, fragile, via, goal):
    v = {
        "n": bool(night),
        "w": bool(rain),
        "u": bool(urgent),
        "f": bool(fragile),
        "v": bool(via),
        "g": goal,
        "-": 0,
        "c": GC.get(goal, "C"),
        "k": goal in ("dorm", "canteen", "lab"),
        "q": goal in ("clinic", "gate"),
    }
    return tuple(v[c] for c in key)


def theta(terms):
    th = np.zeros(NF)
    th[0] = 1e6
    for f, w, lev in terms:
        th[int(f)] += float(w) * (1e6, 1e3, 1.0)[int(lev)]
    return th


def qvals(w, legs, r, th):
    fr = hyper.fronts(w, legs, r == 4)
    q = np.full(4, np.inf)
    for d, (rel, F) in fr.items():
        q[d] = float((feats(F, rel) @ th).min())
    return q


def predict(cfg, r, w, legs, night, urgent, fragile):
    R = cfg.get(str(r))
    if not R or not legs or not legs[-1]:
        return None
    goal = next((t for t, v in w["landmarks"].items() if tuple(legs[-1][0]) in [tuple(p) for p in v]), None)
    via = len(legs) == 2
    g = str(flag_tuple(R["key"], night, w["rain"], urgent, fragile, via, goal))
    G = R["groups"].get(g)
    if G is None or G.get("skip"):
        return None
    q = qvals(w, legs, r, theta(G["terms"]))
    if not np.isfinite(q).any():
        return None
    c = str(flag_tuple(R["cell"], night, w["rain"], urgent, fragile, via, goal))
    kind, order = G["orders"].get(c) or next(iter(G["orders"].values()))
    h = w["heading"]
    mn = q.min()
    ties = [d for d in range(4) if q[d] <= mn + 1e-6]
    if kind == "rel":
        return min(ties, key=lambda d: order.index(REL[h][d]))
    return min(ties, key=lambda d: order.index("UDLR"[d]))


def load(path):
    return json.loads(open(path, encoding="utf-8").read())
