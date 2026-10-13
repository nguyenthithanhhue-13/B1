"""BẢN NỀN tự thích nghi với dữ liệu mới (tu_dong.py gọi; không cần sửa gì khi đề chỉ đổi dữ liệu / đổi luật robot).

Tự xử lý được mà KHÔNG cần sửa code:
  - luật ngầm của robot đổi                     -> học lại một mô hình cho mỗi robot
  - số robot khác 10, robot_id lạ               -> lấy danh sách robot từ dữ liệu; robot có chân lấy từ robots.json
  - số cảnh / số dòng test khác                 -> đếm theo dữ liệu
  - mission có thêm cờ mới (vd "heavy": true)   -> tự học đọc cờ đó từ câu và đưa vào mô hình chiến thuật
  - nhiều điểm ghé (via là danh sách)           -> lúc học dùng đủ các chặng; lúc dự đoán chỉ đọc được 1 điểm ghé
  - một cảnh bị lỗi khi dự đoán                 -> vẫn trả về một bước đi hợp lệ, đếm và báo số cảnh lỗi

Cách chạy (đặt COURIER_DATA và COURIER_WORK trước; tu_dong.py tự đặt):
  python nen.py fit trainonly|final     học bộ đọc câu + mô hình chiến thuật   -> outputs/nen_<chế độ>.pkl, outputs/nlp3_<chế độ>.pkl
  python nen.py chandoan                bảng chẩn đoán trên validation (dùng mô hình trainonly)
  python nen.py test                    ghi outputs/predictions_nen.json (dùng mô hình final)
  python nen.py chon                    (nang_cao.py gọi) chọn mô hình tốt nhất cho từng robot -> outputs/chon.json
"""
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

for _st in (sys.stdout, sys.stderr):      # in được tiếng Việt trên mọi máy Windows
    try:
        _st.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
from common import *      # noqa: E402
import nlp3               # noqa: E402
from pipeline_p2 import cv_worlds, legs_from_mission      # noqa: E402
from strat_ml import move_feats, is_night      # noqa: E402

KNOWN_KEYS = {"text", "goal", "goal_ref", "via", "via_ref", "urgent", "fragile"}
CLASSIC_REF = {"north", "south", "west", "east", "near", "far"}


# ---------------- đọc dữ liệu một cách tổng quát ----------------
def scene_groups(rows):
    """Gom các dòng liên tiếp cùng ảnh + cùng câu thành một cảnh -> [(ảnh, câu, [robot_id...], vị trí dòng đầu)]."""
    out = []
    for i, r in enumerate(rows):
        if out and out[-1][0] == r["image"] and out[-1][1] == r["mission"]:
            out[-1][2].append(r["robot_id"])
        else:
            out.append((r["image"], r["mission"], [r["robot_id"]], i))
    return out


def legged_robots():
    p = DATA / "robots.json"
    if p.exists():
        rb = load_json(p)
        got = {r["robot_id"] for r in rb if r.get("legged")}
        if got or all("legged" in r for r in rb):
            return got
    return {4}


def true_legs(s):
    """Các chặng đi đúng (từ scenes.json). Chấp nhận via là một loại hoặc một danh sách loại."""
    w = world_from_scene(s)
    m = s["mission"]

    def targets(typ, ref):
        if isinstance(ref, dict) and ref.get("rc") is not None:
            return [tuple(ref["rc"])]
        return list(w["landmarks"].get(typ, []))

    vias = m.get("via")
    vias = vias if isinstance(vias, list) else ([vias] if vias else [])
    vrefs = m.get("via_ref")
    vrefs = vrefs if isinstance(vrefs, list) else [vrefs] * len(vias)
    legs = [targets(v, vrefs[i] if i < len(vrefs) else None) for i, v in enumerate(vias)]
    legs.append(targets(m["goal"], m.get("goal_ref")))
    return w, [l for l in legs if l]


KNOWN_REF = CLASSIC_REF | {"north_most", "south_most", "west_most", "east_most", "anchor_near"}


def norm_for_parser(m, stat):
    """Đưa mission về dạng mà bộ đọc câu cũ hiểu (1 điểm ghé, các kiểu tham chiếu đã biết) để nó không vấp khi học.
    Phần chiến thuật vẫn học từ dữ liệu gốc (đủ các chặng, đúng vị trí)."""
    m = dict(m)
    if isinstance(m.get("via"), list):
        stat["via là danh sách"] += 1
        vr = m.get("via_ref")
        m["via_ref"] = (vr[0] if vr else None) if isinstance(vr, list) else vr
        m["via"] = m["via"][0] if m["via"] else None
    if not m.get("via"):
        m["via"], m["via_ref"] = None, None
    for k in ("goal_ref", "via_ref"):
        r = m.get(k)
        if isinstance(r, dict) and r.get("kind") not in KNOWN_REF:
            stat[f"kiểu tham chiếu lạ '{r.get('kind')}'"] += 1
            m[k] = None
    return m


def place_idx(t):
    return PLACES.index(t) if t in PLACES else -1


def extra_keys(missions):
    """Các trường MỚI trong mission có ít giá trị (cờ / nhãn): mô hình sẽ tự học đọc chúng từ câu."""
    keys = {}
    for k in sorted({k for m in missions for k in m} - KNOWN_KEYS):
        vals = [m.get(k) for m in missions]
        if all(isinstance(v, (bool, int, str, type(None))) for v in vals):
            uniq = sorted({str(v) for v in vals})
            if 2 <= len(uniq) <= 8:
                keys[k] = uniq
    return keys


def scene_feats(w, legs, urgent, fragile, has_via, night, mapgoal, goal, via, extras):
    """Đặc trưng từng hướng đi (cho robot thường và robot có chân) + điều kiện của cảnh."""
    f = move_feats(w, legs, bool(urgent), bool(fragile), bool(has_via))
    tail = [float(night), float(mapgoal), place_idx(goal), place_idx(via) if isinstance(via, str) else -1] + list(extras)
    return {lg: (np.hstack([F, np.tile(tail, (4, 1))]), valid) for lg, (F, valid) in f.items()}


def safe_move(w):
    """Bước đi hợp lệ bất kỳ (dùng khi một cảnh bị lỗi): không bao giờ trả về hướng đâm vào tường nếu tránh được."""
    try:
        ok = sorted(d for d, info in w["adj"].get(w["robot"], {}).items() if info[0] != "closed" and info[2])
        return ok[0] if ok else 0
    except Exception:
        return 0


# ---------------- học ----------------
def train_table(split, ek):
    cp = CACHE / f"nen_feats_{split}.pkl"
    if cp.exists():
        return pickle.load(open(cp, "rb"))
    rows, labels, scenes = load_split(split)
    groups = scene_groups(rows)
    assert len(groups) == len(scenes), f"{split}: {len(groups)} cảnh trong observations nhưng {len(scenes)} cảnh trong scenes.json"
    table, bad = [], 0
    for (img, text, robots, i0), s in zip(groups, scenes):
        try:
            w, legs = true_legs(s)
            m = s["mission"]
            gr = m.get("goal_ref")
            mapgoal = isinstance(gr, dict) and gr.get("kind") not in CLASSIC_REF
            ex = [ek[k].index(str(m.get(k))) for k in ek]
            f = scene_feats(w, legs, m.get("urgent"), m.get("fragile"), len(legs) > 1, is_night(DATA / split / img), mapgoal,
                            m["goal"], m.get("via"), ex)
            table.append((f, robots, [labels[i0 + j] for j in range(len(robots))]))
        except Exception as e:
            bad += 1
            if bad <= 3:
                print(f"   bỏ qua một cảnh {split} khi học ({type(e).__name__}: {e})")
    if bad:
        print(f"   !!! {bad}/{len(scenes)} cảnh {split} không dùng được để học")
    pickle.dump(table, open(cp, "wb"))
    return table


def fit(mode):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    t = time.time()
    splits = ["train"] if mode == "trainonly" else ["train", "validation"]
    missions = [s["mission"] for sp in splits for s in load_split(sp)[2]]
    nstat = Counter()
    p = nlp3.MissionParser3().fit([norm_for_parser(m, nstat) for m in missions], verbose=False)
    if nstat:
        print("!!! bộ đọc câu CHƯA hiểu (đã bỏ qua khi học, cần sửa code nếu muốn điểm cao):", dict(nstat), flush=True)
    p.save(OUT / f"nlp3_{mode}.pkl")
    from nlp2 import save_embed_cache
    save_embed_cache()
    print(f"bộ đọc câu: {time.time() - t:.0f}s", flush=True)
    # cờ mới trong mission: học đọc từ câu
    ek = extra_keys([s["mission"] for s in load_split("train")[2]])
    flag_clf = {}
    for k, uniq in ek.items():
        y = [uniq.index(str(m.get(k))) if str(m.get(k)) in uniq else 0 for m in missions]
        flag_clf[k] = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, lowercase=True),
                                    LogisticRegression(C=20, max_iter=2000)).fit([m["text"] for m in missions], y)
        print(f"trường mới '{k}' {uniq}: đã học đọc từ câu", flush=True)
    legged = legged_robots()
    X, Y = {}, {}
    for sp in splits:
        for f, robots, labs in train_table(sp, ek):
            for r, y in zip(robots, labs):
                F, valid = f[r in legged]
                for d in range(4):
                    if valid[d]:
                        X.setdefault(r, []).append(F[d])
                        Y.setdefault(r, []).append(int(d == y))
    models = {}
    for r in sorted(X, key=str):
        models[r] = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4, min_samples_leaf=60,
                                                   l2_regularization=1.0, random_state=0).fit(np.array(X[r]), np.array(Y[r]))
    pickle.dump({"models": models, "legged": legged, "ek": ek, "flag_clf": flag_clf}, open(OUT / f"nen_{mode}.pkl", "wb"))
    print(f"chiến thuật: {len(models)} robot {sorted(models, key=str)}, robot có chân {sorted(legged, key=str)}; tổng {time.time() - t:.0f}s", flush=True)


# ---------------- dự đoán ----------------
class Predictor:
    def __init__(self, mode):
        self.p = nlp3.MissionParser3.load(OUT / f"nlp3_{mode}.pkl")
        self.__dict__.update(pickle.load(open(OUT / f"nen_{mode}.pkl", "rb")))
        self.stat = Counter()
        # ---- mô hình nâng cao (nang_cao.py tạo ra; không có thì chỉ dùng mô hình nền HGB) ----
        self.chon, self.hy, self.hy2, self.greedy = {}, {}, {}, None
        try:
            if (OUT / "chon.json").exists():
                self.chon = {k: v["cach"] for k, v in json.loads((OUT / "chon.json").read_text(encoding="utf-8")).items()}
            if (OUT / f"hyper_{mode}.npz").exists():
                import hyper
                self.hy = hyper.load(OUT / f"hyper_{mode}.npz")
            if (OUT / f"hypergoal_{mode}.npz").exists():
                import hyper
                self.hy2 = hyper.load(OUT / f"hypergoal_{mode}.npz")
            if (OUT / "greedy.json").exists():
                self.greedy = {"params": json.loads((OUT / "greedy.json").read_text(encoding="utf-8"))}
        except Exception as e:
            print(f"!!! không nạp được mô hình nâng cao ({type(e).__name__}: {e}); dùng mô hình nền", flush=True)
            self.chon = {}

    def advanced(self, cach, r, w, ctx):
        """Bước đi theo mô hình nâng cao; None nếu không áp dụng được (khi đó dùng mô hình nền)."""
        try:
            legs, night, urgent, fragile, mapgoal = ctx
            if cach == "hyper" and self.hy and r in self.hy:
                import hyper
                return hyper.predict(self.hy, r, w, legs, night, bool(urgent), bool(fragile), mapgoal)
            if cach == "hyper_goal" and self.hy2 and r in self.hy2:
                import hyper
                return hyper.predict(self.hy2, r, w, legs, night, bool(urgent), bool(fragile), mapgoal)
            if cach == "hyper_x2" and r in self.hy and r in self.hy2:      # trung bình hai mô hình
                import hyper
                L = [hyper.logp(M, r, w, legs, night, bool(urgent), bool(fragile), mapgoal) for M in (self.hy, self.hy2)]
                L = [x for x in L if x is not None]
                ok = [d for d, info in w["adj"].get(w["robot"], {}).items() if hyper.ok_edge(info, r in self.legged)]
                if L and ok:
                    sc = sum(L) / len(L)
                    return max(ok, key=lambda d: sc[d] if np.isfinite(sc[d]) else -1e18)
                return None
            if cach == "greedy" and self.greedy:
                from strat_ml import StrategyML
                return StrategyML._greedy2_move(w, legs, self.greedy, night, urgent, fragile)
        except Exception:
            self.stat["mô hình nâng cao lỗi (đã dùng mô hình nền)"] += 1
        return None

    def moves(self, f, robots, w, ctx=None, force=None):
        out = []
        for r in robots:
            cach = force or self.chon.get(str(r), "nen")
            if cach != "nen" and ctx is not None:
                d = self.advanced(cach, r, w, ctx)
                if d is not None:
                    out.append(int(d))
                    continue
            model = self.models.get(r)
            if model is None:      # robot chưa từng thấy khi học: mượn mô hình của robot đầu tiên
                self.stat["robot lạ (mượn mô hình khác)"] += 1
                model = next(iter(self.models.values()))
            F, valid = f[r in self.legged]
            idx = [d for d in range(4) if valid[d]]
            if not idx:
                out.append(safe_move(w))
                continue
            out.append(int(idx[int(np.argmax(model.predict_proba(F[idx])[:, 1]))]))
        return out

    def extras(self, text):
        return [int(self.flag_clf[k].predict([text])[0]) for k in self.ek]

    def scene(self, w, text, robots, img_path, truth=None, force=None):
        """truth = mission đúng (scenes.json) nếu muốn bỏ qua khâu đọc câu. Trả về (bước đi, thông tin đọc câu hoặc None)."""
        if w is None:
            self.stat["CV không đọc được ảnh"] += 1
            return [0] * len(robots), None
        try:
            night = is_night(img_path)
            if truth is not None:
                s, m = truth
                _, legs = true_legs(s)      # vị trí đích / điểm ghé đúng (theo hàng, cột)
                gr = m.get("goal_ref")
                mapgoal = isinstance(gr, dict) and gr.get("kind") not in CLASSIC_REF
                ex = [self.ek[k].index(str(m.get(k))) if str(m.get(k)) in self.ek[k] else 0 for k in self.ek]
                f = scene_feats(w, legs, m.get("urgent"), m.get("fragile"), len(legs) > 1, night, mapgoal, m["goal"], m.get("via"), ex)
                return self.moves(f, robots, w, (legs, night, m.get("urgent"), m.get("fragile"), mapgoal), force), None
            L = w["landmarks"]
            r = self.p.parse(text, {t for t, v in L.items() if v}, L)
            m = nlp3.resolve3(r, L, w)
            legs = legs_from_mission(w, m)
            if not legs[-1]:
                raise ValueError("không tìm được đích trên bản đồ")
            mapgoal = bool(m.get("goal_ref")) and m["goal_ref"][0] == "pos"
            f = scene_feats(w, legs, m["urgent"], m["fragile"], len(legs) > 1, night, mapgoal, m["goal"], m.get("via"), self.extras(text))
            return self.moves(f, robots, w, (legs, night, m["urgent"], m["fragile"], mapgoal), force), (m, legs)
        except Exception as e:
            self.stat[f"cảnh lỗi ({type(e).__name__})"] += 1
            return [safe_move(w)] * len(robots), None


def test():
    rows = load_json(DATA / "test" / "observations.json")
    worlds = cv_worlds("test", "v1final")
    P = Predictor("final")
    preds = []
    for img, text, robots, _ in scene_groups(rows):
        preds += P.scene(worlds[img]["world"], text, robots, DATA / "test" / img)[0]
    assert len(preds) == len(rows) and all(isinstance(x, int) and 0 <= x <= 3 for x in preds)
    (OUT / "predictions_nen.json").write_text(json.dumps(preds), encoding="utf-8")
    from nlp2 import save_embed_cache
    save_embed_cache()
    print("đã ghi", OUT / "predictions_nen.json", "| số dòng:", len(preds), "| phân bố:", sorted(Counter(preds).items()), "|", dict(P.stat) or "không cảnh nào lỗi")


def chon():
    """Chọn cho từng robot mô hình đúng nhất trên validation (bản đồ đúng + câu đúng, mô hình chỉ học train) -> outputs/chon.json."""
    rows, labels, scenes = load_split("validation")
    groups = scene_groups(rows)
    (OUT / "chon.json").unlink(missing_ok=True)
    P = Predictor("trainonly")
    ways = ["nen"] + (["hyper"] if P.hy else []) + (["hyper_goal"] if P.hy2 else []) + (["hyper_x2"] if P.hy and P.hy2 else [])         + (["greedy"] if P.greedy else [])
    ok = {wy: Counter() for wy in ways}
    n = Counter()
    for (img, text, robots, i0), s in zip(groups, scenes):
        wt, _ = true_legs(s)
        for wy in ways:
            pr = P.scene(wt, text, robots, DATA / "validation" / img, truth=(s, s["mission"]), force=wy)[0]
            for j, r in enumerate(robots):
                ok[wy][r] += pr[j] == labels[i0 + j]
        for r in robots:
            n[r] += 1
    out = {}
    print("Điểm từng robot trên validation theo từng loại mô hình (nen = mô hình nền):")
    for r in sorted(n, key=str):
        acc = {wy: ok[wy][r] / n[r] for wy in ways if not (wy == "greedy" and r != 9) and not (wy == "hyper" and r not in P.hy)
               and not (wy == "hyper_goal" and r not in P.hy2) and not (wy == "hyper_x2" and (r not in P.hy or r not in P.hy2))}
        best = max(acc, key=lambda k: (acc[k], k != "nen"))
        out[str(r)] = {"cach": best, "diem": acc}
        print(f"   R{r}: " + "  ".join(f"{k}={v:.3f}" for k, v in acc.items()) + f"   -> chọn {best}")
    (OUT / "chon.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Trung bình: nền {sum(v['diem']['nen'] for v in out.values()) / len(out):.4f} -> sau khi chọn {sum(max(v['diem'].values()) for v in out.values()) / len(out):.4f}")


def chandoan():
    rows, labels, scenes = load_split("validation")
    groups = scene_groups(rows)
    worlds = cv_worlds("validation", "v1final")
    P = Predictor("trainonly")
    n = len(scenes)
    cv, nl = Counter(), Counter()
    out = {k: [] for k in ("11", "1n", "c1", "cn")}
    for (img, text, robots, _), s in zip(groups, scenes):
        path = DATA / "validation" / img
        wt, legs_t = true_legs(s)
        m = s["mission"]
        wc = worlds[img]["world"]
        if wc is not None:
            ok = {"đường đi": wc["adj"] == wt["adj"],
                  "địa điểm": {k: sorted(v) for k, v in wc["landmarks"].items() if v} == {k: sorted(v) for k, v in wt["landmarks"].items() if v},
                  "robot": wc["robot"] == wt["robot"] and wc["heading"] == wt["heading"], "thời tiết": wc["rain"] == wt["rain"]}
            for k, v in ok.items():
                cv[k] += v
            cv["CẢ BẢN ĐỒ"] += all(ok.values())
        out["11"] += P.scene(wt, text, robots, path, truth=(s, m))[0]
        pr, info = P.scene(wt, text, robots, path)
        out["1n"] += pr
        if info:
            mm, legs = info
            ok = {"đích": sorted(legs[-1]) == sorted(legs_t[-1]), "điểm ghé": [sorted(x) for x in legs[:-1]] == [sorted(x) for x in legs_t[:-1]],
                  "gấp": bool(mm["urgent"]) == bool(m.get("urgent")), "dễ vỡ": bool(mm["fragile"]) == bool(m.get("fragile"))}
            for k, v in ok.items():
                nl[k] += v
            nl["CẢ CÂU"] += all(ok.values())
        if wc is None:
            out["c1"] += [0] * len(robots)
            out["cn"] += [0] * len(robots)
        else:
            out["c1"] += P.scene(wc, text, robots, path, truth=(s, m))[0]
            out["cn"] += P.scene(wc, text, robots, path)[0]
    print(f"\n=== CHẨN ĐOÁN trên validation ({n} cảnh) ===")
    print("ĐỌC ẢNH  đúng: " + "  ".join(f"{k} {v}/{n}" for k, v in cv.items()))
    print("ĐỌC CÂU  đúng: " + "  ".join(f"{k} {v}/{n}" for k, v in nl.items()))
    names = {"11": "(1) bản đồ ĐÚNG + câu ĐÚNG", "1n": "(2) bản đồ ĐÚNG + máy đọc câu", "c1": "(3) máy đọc ảnh + câu ĐÚNG", "cn": "(4) máy đọc cả hai = ĐIỂM THẬT"}
    S = {}
    for k, name in names.items():
        S[k], per = macro_accuracy(rows, labels, out[k])
        print(f"{name:34s} {S[k]:.4f}   " + " ".join(f"R{r}={v:.2f}" for r, v in per.items()))
    loss = {"CHIẾN THUẬT ROBOT": 1 - S["11"], "ĐỌC CÂU": S["11"] - S["1n"], "ĐỌC ẢNH": S["11"] - S["c1"]}
    print("Mất điểm do: " + " | ".join(f"{k} {v:.3f}" for k, v in loss.items()))
    print(f"=> Khâu yếu nhất: {max(loss, key=loss.get)}")
    if P.stat:
        print("Cảnh có vấn đề:", dict(P.stat))
    (OUT / "chan_doan.json").write_text(json.dumps({"diem": S, "doc_anh": cv, "doc_cau": nl, "van_de": P.stat}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "fit":
        fit(sys.argv[2] if len(sys.argv) > 2 else "final")
    elif cmd == "chandoan":
        chandoan()
    elif cmd == "chon":
        chon()
    elif cmd == "test":
        test()
    else:
        sys.exit(__doc__)
