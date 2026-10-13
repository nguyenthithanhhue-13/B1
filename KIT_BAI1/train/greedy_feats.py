"""Đặc trưng một bước đi cho robot 9 (tham lam): khoảng cách tới điểm đến (Manhattan, Euclid, số bước BFS, theo pixel),
thay đổi khoảng cách, trạng thái đoạn, địa điểm đi vào, kiểu rẽ. Thư viện, được train/greedy_rules.py dùng.
"""

import collections
import math
import sys

import numpy as np

sys.path.insert(0, "train")
from lib import *

PL = ["library", "dorm", "sports", "clinic", "canteen", "parking", "lecture", "lab", "office", "gate"]


def bfs(w, tg, legged):
    D = {t: 0 for t in tg}
    q = collections.deque(tg)
    radj = collections.defaultdict(list)
    for n, dd in w["adj"].items():
        for d, info in dd.items():
            if edge_ok(info, legged):
                radj[(n[0] + DRC[d][0], n[1] + DRC[d][1])].append(n)
    while q:
        n = q.popleft()
        for m in radj.get(n, []):
            if m not in D:
                D[m] = D[n] + 1
                q.append(m)
    return D


def feats(w, legs, xy, legged=False):
    s = w["robot"]
    tg = legs[0]
    h = w["heading"]
    tgs = {p for L in legs for p in L}
    lmt = {p: PL.index(t) for t, v in w["landmarks"].items() for p in v if p not in tgs}
    V = bfs(w, tg, legged)
    sp = np.mean([math.dist(xy[a], xy[b]) for a in xy for b in [(a[0], a[1] + 1)] if b in xy] or [1.0])

    def dists(n):
        man = min(abs(n[0] - t[0]) + abs(n[1] - t[1]) for t in tg)
        euc = min(math.hypot(n[0] - t[0], n[1] - t[1]) for t in tg)
        px = min(math.dist(xy[n], xy[t]) for t in tg) / sp if n in xy and all(t in xy for t in tg) else euc
        return [man, euc, min(V.get(n, 30), 30), px]

    d0 = dists(s)
    out = {}
    for d, info in w["adj"].get(s, {}).items():
        if not edge_ok(info, legged):
            continue
        n2 = (s[0] + DRC[d][0], s[1] + DRC[d][1])
        st = info[0]
        rel = REL[h][d]
        dn = dists(n2)
        out[d] = np.array(
            dn
            + [a - b for a, b in zip(dn, d0)]
            + [st == "crowded", st == "covered", st == "normal", bool(info[1]), n2 in lmt]
            + [lmt.get(n2) == k for k in range(10)]
            + [rel == x for x in "SRLB"],
            np.float32,
        )
    return out
