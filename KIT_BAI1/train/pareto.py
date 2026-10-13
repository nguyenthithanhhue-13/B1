"""Bước 1 (chiến thuật): liệt kê mặt Pareto các đường (số đoạn <= ngắn nhất + 5) cho mỗi cảnh và mỗi bước đầu, theo vector đặc trưng
(số đoạn, đông, mái che, không mái che, địa điểm đi xuyên theo loại, bậc thang, rẽ phải / trái / quay đầu).
Cách chạy:  python train/pareto.py <train|validation>
Ghi: cache/p2_paretotu_<split>_0.pkl (robot không chân), cache/p2_paretotu_<split>_1.pkl (robot 4 đi được bậc thang).
"""

import sys, pickle, heapq, collections, numpy as np

sys.path.insert(0, "train")
from lib import *
from multiprocessing import Pool

SLACK = int(sys.argv[2]) if len(sys.argv) > 2 else 5
NFP = 8

UNC = 1
TYPED = True
HOR = 999


def ok_edge(info, legged):
    status, stairs, allowed = info
    return not (status in ("closed", "missing") or not allowed or (stairs and not legged))


def dominated(v, front):
    for u in front:
        if all(a <= b for a, b in zip(u, v)):
            return True
    return False


def fronts(x, legged):
    w = x["w"]
    legs = x["legs"]
    tg = {p for L in legs for p in L}
    lm = {p for v in w["landmarks"].values() for p in v} - tg
    lmt = {p: PLACES.index(t) for t, v in w["landmarks"].items() for p in v if p not in tg}

    def lv(n):
        return tuple(int(lmt.get(n) == k) for k in range(10)) if TYPED else ()

    goal = set(legs[-1])
    via = set(legs[0]) if len(legs) == 2 else None
    adj = w["adj"]

    def bfs(srcs):
        D = {s: 0 for s in srcs}
        q = collections.deque(srcs)
        while q:
            n = q.popleft()
            for d, info in adj.get(n, {}).items():
                if ok_edge(info, legged):
                    n2 = (n[0] + DRC[d][0], n[1] + DRC[d][1])
                    if n2 not in D:
                        D[n2] = D[n] + 1
                        q.append(n2)
        return D

    radj = collections.defaultdict(list)
    for n, dd in adj.items():
        for d, info in dd.items():
            if ok_edge(info, legged):
                radj[(n[0] + DRC[d][0], n[1] + DRC[d][1])].append(n)

    def rbfs(tgts):
        D = {t: 0 for t in tgts}
        q = collections.deque(tgts)
        while q:
            n = q.popleft()
            for m in radj.get(n, []):
                if m not in D:
                    D[m] = D[n] + 1
                    q.append(m)
        return D

    dg = rbfs(list(goal))
    if via:
        best_v = {v: dg.get(v, 10**6) for v in via}
        rv = {}
        for v in via:
            Dv = rbfs([v])
            for n, k in Dv.items():
                rv[n] = min(rv.get(n, 10**6), k + best_v[v])
        rem0 = rv
    else:
        rem0 = dg
    s0 = w["robot"]
    h0 = w["heading"]
    if rem0.get(s0, 10**6) >= 10**6:
        return {}
    H = rem0[s0] + SLACK
    out = {}
    for d, info in adj.get(s0, {}).items():
        if not ok_edge(info, legged):
            continue
        n2 = (s0[0] + DRC[d][0], s0[1] + DRC[d][1])
        st = info[0]
        f0 = (
            (1, st == "crowded", st == "covered", n2 in lm, bool(info[1]), 0, 0, 0)
            + ((st != "covered",) if UNC else ())
            + lv(n2)
        )
        ph = 1 if (via and n2 in via) else (0 if via else 1)
        labels = collections.defaultdict(list)
        res = []
        start = (n2, d, ph)
        heap = [(f0[0], f0, start)]
        labels[start].append(f0)
        while heap:
            hop, f, (n, h, p) = heapq.heappop(heap)
            if f not in labels[(n, h, p)]:
                continue
            if p == 1 and n in goal:
                if not dominated(f, res):
                    res = [u for u in res if not all(a <= b for a, b in zip(f, u))] + [f]
                continue
            rem = dg if p == 1 else rem0
            if hop + rem.get(n, 10**6) > H:
                continue
            if dominated(f, res):
                continue
            for d2, info2 in adj.get(n, {}).items():
                if not ok_edge(info2, legged):
                    continue
                m = (n[0] + DRC[d2][0], n[1] + DRC[d2][1])
                rel = REL[h][d2]
                st2 = info2[0]
                p2 = 1 if (p == 1 or (via and m in via)) else 0
                v = 1 if f[0] < HOR else 0
                g = (
                    (
                        f[0] + 1,
                        f[1] + v * (st2 == "crowded"),
                        f[2] + v * (st2 == "covered"),
                        f[3] + v * (m in lm),
                        f[4] + bool(info2[1]),
                        f[5] + v * (rel == "R"),
                        f[6] + v * (rel == "L"),
                        f[7] + v * (rel == "B"),
                    )
                    + ((f[8] + v * (st2 != "covered"),) if UNC else ())
                    + tuple(a + v * b for a, b in zip(f[8 + UNC :], lv(m)))
                )
                rem2 = dg if p2 == 1 else rem0
                if g[0] + rem2.get(m, 10**6) > H:
                    continue
                key = (m, d2, p2)
                L = labels[key]
                if dominated(g, L):
                    continue
                labels[key] = [u for u in L if not all(a <= b for a, b in zip(g, u))] + [g]
                heapq.heappush(heap, (g[0], g, key))
        if res:
            out[d] = (REL[h0][d], np.array(res, np.int16))
    return out


def job(x):
    return fronts(x, False), fronts(x, True)


if __name__ == "__main__":
    split = sys.argv[1]
    D = load(split)
    for x in D:
        if len(x["legs"]) == 2:
            pass
    with Pool(6) as pool:
        R = pool.map(job, D, chunksize=8)
    T = (
        ("t" if TYPED else "")
        + ("u" if UNC else "")
        + (f"s{SLACK}" if SLACK != 5 else "")
        + (f"h{HOR}" if HOR != 999 else "")
    )
    pickle.dump([r[0] for r in R], open(f"cache/p2_pareto{T}_{split}_0.pkl", "wb"))
    pickle.dump([r[1] for r in R], open(f"cache/p2_pareto{T}_{split}_1.pkl", "wb"))
    sz = [len(v[1]) for r in R for v in r[0].values()]
    print(split, len(D), "cảnh; cỡ mặt Pareto TB", np.mean(sz).round(1), "max", max(sz))
