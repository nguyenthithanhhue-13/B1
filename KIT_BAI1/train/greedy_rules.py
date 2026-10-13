"""Dò luật tham lam gọn cho robot 9 theo nhóm (đêm, gấp, dễ vỡ): điểm bước = khoảng cách + phạt đông / mái che / địa điểm / rẽ,
hòa theo hướng tuyệt đối lên -> phải -> xuống -> trái.
Cách chạy:  python train/greedy_rules.py [KEY=nuf] [số lần khởi đầu ngẫu nhiên=4]      (học trên train, hạt giống 0)
Ghi: cache/p2_greedy_exactfix_<KEY>.json; tham số này được chép vào mục "9" -> "params" của outputs/strategy_hybrid*.json.
"""

import sys
import json
import numpy as np

KEY = sys.argv[1] if len(sys.argv) > 1 else "nuf"
NRS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
sys.path.insert(0, "train")
from greedy_feats import PL, feats
from lib import load

RELS = "SRLB"
ORDERS = ["A0312"]


def prep(split):
    D = load(split)
    X = np.full((len(D), 4, 27), np.nan)
    keys = []
    Y = []
    for i, x in enumerate(D):
        xy = {tuple(n["rc"]): tuple(n["xy"]) for n in x["s"]["nodes"]}
        for d, v in feats(x["w"], x["legs"], xy).items():
            X[i, d] = v
        m = x["m"]
        k = []
        for ch in KEY:
            k.append(
                {
                    "n": x["s"]["style"] == "night",
                    "u": m["urgent"],
                    "f": m["fragile"],
                    "v": bool(m["via"]),
                    "w": x["w"]["rain"],
                }[ch]
            )
        keys.append(tuple(int(b) for b in k))
        Y.append(x["y"][9])
    return X, keys, np.array(Y)


def score(X, p):
    s = (
        X[:, :, p["dist"]]
        + p["a"] * X[:, :, 8]
        + p["c"] * X[:, :, 9]
        + p["b"] * X[:, :, 12]
        + X[:, :, 13:23] @ np.array(p["bt"])
        + X[:, :, 24] * p["R"]
        + X[:, :, 25] * p["L"]
        + X[:, :, 26] * p["B"]
    )
    s = np.where(np.isnan(s), np.inf, s)
    if p["order"].startswith("A"):
        ab = [int(c) for c in p["order"][1:]]
        s = s + np.array([ab.index(d) for d in range(4)])[None, :] * 1e-7
        return s.argmin(1)
    tie = np.array([[p["order"].index(c) for c in RELS]]) * 1e-7
    rel = np.nan_to_num(X[:, :, 23:27]).argmax(-1)
    s = s + np.take(tie[0], rel)
    return s.argmin(1)


VALS = [0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10]


def fit(X, y, rng):
    best = None
    for rs in range(NRS):
        p = dict(
            dist=int(rng.integers(2)),
            a=float(rng.choice(VALS)),
            c=float(rng.choice([0, -0.25, -0.5, 0.25])),
            b=float(rng.choice(VALS)),
            bt=[0.0] * 10,
            R=0.0,
            L=0.0,
            B=float(rng.choice([0, 0.25, 0.5])),
            order="A0312",
        )
        acc = (score(X, p) == y).sum()
        for it in range(5):
            ch = False
            for k in ["dist", "a", "c", "b", "R", "L", "B", "order"] + [("bt", j) for j in range(10)]:
                if k == "dist":
                    cand = [0, 1]
                elif k == "order":
                    cand = ORDERS
                elif k == "c":
                    cand = [-1, -0.5, -0.25, -0.1] + VALS
                else:
                    cand = VALS + ([-0.25, -0.5] if k in ("R", "L") else [])
                for v in cand:
                    q = dict(p, bt=list(p["bt"]))
                    if isinstance(k, tuple):
                        q["bt"][k[1]] = v
                    else:
                        q[k] = v
                    a2 = (score(X, q) == y).sum()
                    if a2 > acc:
                        acc, p, ch = a2, q, True
            if not ch:
                break
        if best is None or acc > best[0]:
            best = (acc, p)
    return best


if __name__ == "__main__":
    XT, KT, YT = prep("train")
    XV, KV, YV = prep("validation")
    rng = np.random.default_rng(0)
    tt = tv = 0
    out = {}
    for k in sorted(set(KT)):
        it = [i for i, kk in enumerate(KT) if kk == k]
        iv = [i for i, kk in enumerate(KV) if kk == k]
        acc, p = fit(XT[it], YT[it], rng)
        av = (score(XV[iv], p) == YV[iv]).sum() if iv else 0
        tt += acc
        tv += av
        out[str(k)] = p
        bt = " ".join(f"{PL[j][:4]}={v:g}" for j, v in enumerate(p["bt"]) if v)
        print(
            f"  {KEY}={k}: train {acc}/{len(it)} val {av}/{len(iv)} | {'euc' if p['dist'] else 'man'} a={p['a']} c={p['c']} b={p['b']} R={p['R']} L={p['L']} B={p['B']} {p['order']} | {bt}",
            flush=True,
        )
    print(f"R9 luật gọn key={KEY}: train {tt / len(YT):.3f} validation {tv / len(YV):.3f}")
    json.dump(out, open(f"cache/p2_greedy_exactfix_{KEY}.json", "w"), ensure_ascii=False)
