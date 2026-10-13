"""Bước 3 (luật chính xác): tìm luật cho một robot, riêng từng nhóm điều kiện (tìm kiếm chùm trên các hạng tử chi phí + chọn thứ tự
phá hòa tốt nhất trong 48 thứ tự).
Cách chạy:  ENUM=5p [FITSPLIT=trainval] [REFINE=0] python train/rules_search.py <robot> <KEY nhóm> <CELL ô hòa> [beam=6] [độ sâu=5]
  KEY / CELL gồm các chữ: n đêm, w mưa, u gấp, f dễ vỡ, v có điểm ghé, c cụm loại đích, g loại đích, - tất cả.
  FITSPLIT=trainval: học trên train + validation (bản nộp); mặc định học train, đo validation.
Ghi: cache/p3_search_r<robot>_<KEY>_<CELL>[_tv].json.
"""

import sys
import os
import pickle
import itertools
import time
import json
import numpy as np

sys.path.insert(0, "train")
from lib import *

NAMES = (
    ["hop", "crowd", "cov", "lm", "stair", "R", "L", "B", "oneway"]
    + [f"t:{p[:4]}" for p in PLACES]
    + ["fR", "fL", "fB", "f_crowd", "f_cov", "f_lm", "run0", "turns", "crowd3", "othgoal", "othvia", "deg3"]
)

NM = NAMES + ["uncov", "normal", "tR_all", "tL_all", "tB_all", "turns_all"]
FL = {
    "n": lambda x: x["s"]["style"] == "night",
    "w": lambda x: x["w"]["rain"],
    "u": lambda x: x["m"]["urgent"],
    "f": lambda x: x["m"]["fragile"],
    "v": lambda x: bool(x["m"]["via"]),
    "g": lambda x: x["m"]["goal"],
    "-": lambda x: 0,
    "c": lambda x: GC.get(x["m"]["goal"], "C"),
}

GC = {"library": "B", "lecture": "B", "lab": "B", "dorm": "A", "canteen": "A"}


def kf(x, key):
    return tuple(FL[c](x) for c in key)


def derive(F):
    F = F.astype(np.float64)
    unc = F[:, 0] - F[:, 2]
    nor = F[:, 0] - F[:, 2] - F[:, 1]
    tR = F[:, 5] + F[:, 19]
    tL = F[:, 6] + F[:, 20]
    tB = F[:, 7] + F[:, 21]
    return np.column_stack([F, unc, nor, tR, tL, tB, tR + tL + tB])


class Data:
    def __init__(self, split, r, items_filter=None):
        D, E = [], []
        for sp in (("train", "validation") if split == "trainval" else (split,)):
            D += load(sp)
            E += pickle.load(open(f"cache/p3_enum{os.environ.get('ENUM', '2')}_{sp}_{int(r == 4)}.pkl", "rb"))
        self.X = []
        Fs = []
        seg = []
        k = 0
        for x, en in zip(D, E):
            if items_filter and not items_filter(x):
                continue
            if not en:
                continue
            self.X.append(x)
            for d in range(4):
                if d in en:
                    Fs.append(derive(en[d][0]))
                    seg += [k * 4 + d] * len(en[d][0])
            k += 1
        self.n = k
        self.F = np.vstack(Fs)
        self.seg = np.array(seg)
        o = np.argsort(self.seg, kind="stable")
        self.F = self.F[o]
        self.seg = self.seg[o]
        self.useg, self.start = np.unique(self.seg, return_index=True)
        self.y = np.array([x["y"][r] for x in self.X])
        self.h = np.array([x["w"]["heading"] for x in self.X])

    def qmat(self, theta):
        c = self.F @ theta
        mn = np.minimum.reduceat(c, self.start)
        Q = np.full(self.n * 4, np.inf)
        Q[self.useg] = mn
        return Q.reshape(self.n, 4)


ORDS = [("rel", "".join(o)) for o in itertools.permutations("SRLB")] + [
    ("abs", "".join(o)) for o in itertools.permutations("UDLR")
]


def rank_tables(h):
    T = np.zeros((len(h), len(ORDS), 4))
    for j, (kind, o) in enumerate(ORDS):
        for d in range(4):
            if kind == "rel":
                T[:, j, d] = [o.index(REL[hh][d]) for hh in h]
            else:
                T[:, j, d] = o.index("UDLR"[d])
    return T


def score(Q, y, R, cells, fixed=None):
    mn = Q.min(1, keepdims=True)
    tie = Q <= mn + 1e-6
    big = np.where(tie[:, None, :], R, 99)
    pred = big.argmin(2)
    ok = pred == y[:, None]
    tot = 0
    best = {}
    P = np.zeros(len(y), int)
    for c in np.unique(cells):
        m = cells == c
        if fixed is not None and c in fixed:
            j = fixed[c]
        else:
            s = ok[m].sum(0)
            j = int(s.argmax())
        best[c] = j
        tot += ok[m, j].sum()
        P[m] = pred[m, j]
    return int(tot), best, P


def build_theta(terms, nf):
    th = np.zeros(nf)
    th[0] = 1e6
    for f, wv, lev in terms:
        th[f] += wv * (1e6, 1e3, 1.0)[lev]
    return th


def search(Dt, cells, beam=6, depth=5, feats=None, weights=(0.5, 1, 2, 3, 4, 5, 6, 8), verbose=False):
    nf = Dt.F.shape[1]
    R = rank_tables(Dt.h)
    feats = feats or [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        31,
        33,
        34,
        35,
        36,
        19,
        20,
        21,
        22,
        23,
        24,
    ]
    cands = []
    for f in feats:
        for lev in (0, 1, 2):
            ws = weights if lev == 0 else (1,)
            for wv in ws:
                for sg in (1, -1):
                    if lev == 0 and sg < 0 and f not in (2, 31):
                        continue
                    cands.append((f, sg * wv, lev))
    s0, _, _ = score(Dt.qmat(build_theta([], nf)), Dt.y, R, cells)
    B = [(s0, ())]
    best = B[0]
    for dep in range(depth):
        nxt = []
        for s, terms in B:
            used = {(t[0], t[2]) for t in terms}
            for c in cands:
                if (c[0], c[2]) in used:
                    continue
                if any(t[2] > c[2] for t in terms[-1:]) and False:
                    continue
                tt = tuple(sorted(terms + (c,)))
                sc, _, _ = score(Dt.qmat(build_theta(tt, nf)), Dt.y, R, cells)
                nxt.append((sc, tt))
        nxt = sorted(set(nxt), key=lambda z: -z[0])[:beam]
        if not nxt or nxt[0][0] <= best[0]:
            if nxt and nxt[0][0] > best[0]:
                best = nxt[0]
            B = nxt
            if dep >= 1 and (not nxt or nxt[0][0] <= best[0]):
                break
        else:
            best = nxt[0]
            B = nxt
        if verbose:
            print("   độ sâu", dep + 1, best[0], "/", Dt.n, [(NM[f], w, l) for f, w, l in best[1]], flush=True)
    return best


def refine(Dt, cells, terms, feats=None, weights=(0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10), rounds=4):
    nf = Dt.F.shape[1]
    R = rank_tables(Dt.h)
    feats = feats or [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 31, 33, 34, 35, 36, 19, 20, 21]
    cur = {(f, l): w for f, w, l in terms}

    def sc(c):
        return score(Dt.qmat(build_theta([(f, w, l) for (f, l), w in c.items() if w != 0], nf)), Dt.y, R, cells)[0]

    best = sc(cur)
    for _ in range(rounds):
        ch = False
        for f in feats:
            ws = list(weights) + ([-w for w in weights] if f in (2, 31) else [])
            for wv in [0] + ws:
                c2 = dict(cur)
                c2[(f, 0)] = wv
                s2 = sc(c2)
                if s2 > best:
                    best, cur, ch = s2, c2, True
        if not ch:
            break
    return best, tuple(sorted((f, w, l) for (f, l), w in cur.items() if w != 0))


def cell_codes(X, CELL):
    keys = [kf(x, CELL) for x in X]
    u = sorted(set(keys), key=str)
    return np.array([u.index(k) for k in keys]), u


if __name__ == "__main__":
    r = int(sys.argv[1])
    KEY = sys.argv[2]
    CELL = sys.argv[3]
    BEAM = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    DEPTH = int(sys.argv[5]) if len(sys.argv) > 5 else 5
    t0 = time.time()
    TR = load("train")
    VA = load("validation")
    groups = sorted({kf(x, KEY) for x in TR}, key=str)
    tt = tv = nt = nv = 0
    OUT = {}
    for g in groups:
        Dt = Data(os.environ.get("FITSPLIT", "train"), r, lambda x: kf(x, KEY) == g)
        Dv = Data("validation", r, lambda x: kf(x, KEY) == g)
        ct, cu = cell_codes(Dt.X, CELL)
        s, terms = search(Dt, ct, BEAM, DEPTH)
        if os.environ.get("REFINE", "1") == "1":
            s, terms = refine(Dt, ct, terms)
        th = build_theta(terms, Dt.F.shape[1])
        sc, bo, _ = score(Dt.qmat(th), Dt.y, rank_tables(Dt.h), ct)
        okv = 0
        if Dv.n:
            cvk = [kf(x, CELL) for x in Dv.X]
            fixed = {}
            cv = np.array([cu.index(k) if k in cu else -1 for k in cvk])
            okv, _, _ = score(Dv.qmat(th), Dv.y, rank_tables(Dv.h), cv, fixed=bo)
        tt += sc
        nt += Dt.n
        tv += okv
        nv += Dv.n
        OUT[str(g)] = dict(
            terms=[(int(f), float(w), int(l)) for f, w, l in terms], orders={str(cu[c]): ORDS[j] for c, j in bo.items()}
        )
        print(
            f"R{r} {KEY}={g}: train {sc}/{Dt.n} val {okv}/{Dv.n} | "
            + " ".join(f"{NM[f]}:{w:+g}@{l}" for f, w, l in terms)
            + " | thứ tự: "
            + " ".join(f"{cu[c]}->{ORDS[j][1]}" for c, j in bo.items())
            + f" ({time.time() - t0:.0f}s)",
            flush=True,
        )
    print(f"R{r} KEY={KEY} CELL={CELL}: train {tt / nt:.4f} val {tv / max(1, nv):.4f}")
    json.dump(
        OUT,
        open(f"cache/p3_search_r{r}_{KEY}_{CELL}{'_tv' if os.environ.get('FITSPLIT') == 'trainval' else ''}.json", "w"),
        ensure_ascii=False,
    )
