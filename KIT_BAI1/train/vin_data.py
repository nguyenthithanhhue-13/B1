"""Tạo dữ liệu cho bộ tìm đường có hàm chi phí học được: đặc trưng từng đoạn đường, điều kiện cảnh, đích / điểm ghé, nhãn.
Cách chạy:  python train/vin_data.py <train|validation>      Ghi: cache/p2_vin_<split>.npz.
"""

import sys, numpy as np

sys.path.insert(0, "train")
from lib import *

PL = PLACES

NF = 4 + 1 + 10 + 4 + (0)
G6 = ["tui do", "hop giay", "goi hang", "thung hang", "tap tai lieu", "buu kien"]
import re as _re

sys.path.insert(0, "src")
from mapref import unaccent as _un


def item6(text):
    u = _un(text)
    for k, g in enumerate(G6):
        if _re.search(r"(?<![a-z])" + g + r"(?![a-z])(?! giat| dat)", u):
            return k
    return 6


def encode(x, legged):
    w = x["w"]
    tg = {p for L in x["legs"] for p in L}
    lt = {p: PL.index(t) for t, v in w["landmarks"].items() for p in v if p not in tg}
    nxt = np.full((325, 4), 324, np.int16)
    E = np.zeros((325, 4, NF), np.uint8)
    for n, dd in w["adj"].items():
        if not (0 <= n[0] < 9 and 0 <= n[1] < 9):
            continue
        for d, info in dd.items():
            if not edge_ok(info, legged):
                continue
            n2 = (n[0] + DRC[d][0], n[1] + DRC[d][1])
            if not (0 <= n2[0] < 9 and 0 <= n2[1] < 9):
                continue
            st = info[0]
            base = [st == "crowded", st == "covered", st == "normal", bool(info[1]), n2 in lt] + [
                lt.get(n2) == k for k in range(10)
            ]
            for h in range(4):
                s = (n[0] * 9 + n[1]) * 4 + h
                nxt[s, d] = (n2[0] * 9 + n2[1]) * 4 + d
                rel = REL[h][d]
                E[s, d] = base + [rel == "S", rel == "R", rel == "L", rel == "B"] + ([])
    return nxt, E


def cond(x):
    m = x["m"]
    s = x["s"]
    mapg = bool(m["goal_ref"]) and m["goal_ref"]["kind"] in (
        "anchor_near",
        "north_most",
        "south_most",
        "west_most",
        "east_most",
    )
    return np.array(
        [x["w"]["rain"], s["style"] == "night", m["urgent"], m["fragile"], bool(m["via"]), mapg]
        + [m["goal"] == t for t in PL]
        + [m["via"] == t for t in PL]
        + ([]),
        np.float32,
    )


ROT_D = [3, 2, 0, 1]


def rotate(x, k):
    if k == 0:
        return x
    w = x["w"]
    R = 9

    def P(p):
        return (p[1], R - 1 - p[0])

    adj = {P(n): {ROT_D[d]: info for d, info in dd.items()} for n, dd in w["adj"].items()}
    w2 = dict(
        w,
        adj=adj,
        landmarks={t: [P(p) for p in v] for t, v in w["landmarks"].items()},
        robot=P(w["robot"]),
        heading=ROT_D[w["heading"]],
    )
    x2 = dict(x, w=w2, legs=[[P(p) for p in L] for L in x["legs"]], y=[ROT_D[v] for v in x["y"]])
    return rotate(x2, k - 1)


if __name__ == "__main__":
    split = sys.argv[1]
    D = load(split)
    if len(sys.argv) > 2 and sys.argv[2] == "aug":
        D = [rotate(x, k) for x in D for k in range(4)]
        split = split + "_aug"
    N = len(D)
    NX = np.zeros((N, 325, 4), np.int16)
    NXL = np.zeros((N, 325, 4), np.int16)
    EE = np.zeros((N, 325, 4, NF), np.uint8)
    EL = np.zeros((N, 325, 4, NF), np.uint8)
    G = np.zeros((N, 81), bool)
    V = np.zeros((N, 81), bool)
    ST = np.zeros(N, np.int16)
    C = np.zeros((N, 26 + (0)), np.float32)
    Y = np.zeros((N, 10), np.int8)
    for i, x in enumerate(D):
        NX[i], EE[i] = encode(x, False)
        NXL[i], EL[i] = encode(x, True)
        for p in x["legs"][-1]:
            G[i, p[0] * 9 + p[1]] = True
        if len(x["legs"]) == 2:
            for p in x["legs"][0]:
                V[i, p[0] * 9 + p[1]] = True
        rb = x["w"]["robot"]
        ST[i] = (rb[0] * 9 + rb[1]) * 4 + x["w"]["heading"]
        C[i] = cond(x)
        Y[i] = x["y"]
    np.savez_compressed(f"cache/p2_vin_{split}.npz", nxt=NX, nxtl=NXL, E=EE, EL=EL, goal=G, via=V, start=ST, C=C, y=Y)
    print(split, N, "xong")
