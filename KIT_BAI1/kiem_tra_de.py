"""Soi bộ dữ liệu MỚI khác bộ vòng private ở đâu (chỉ đọc train / validation và phần không nhãn của test: số dòng, tên trường).
Cách chạy:
  python kiem_tra_de.py <thư mục dữ liệu mới>            -> in các điểm KHÁC so với ho_so_private.json
  python kiem_tra_de.py <thư mục dữ liệu> --luu <file>   -> lưu hồ sơ của bộ này (dùng để tạo file tham chiếu)
Mỗi dòng "KHÁC" là một chỗ code có thể phải sửa; xem SO_TAY_NGAY_THI.md mục 3 để biết sửa file nào.
"""
import json
import sys
from collections import Counter
from pathlib import Path

for _st in (sys.stdout, sys.stderr):      # in được tiếng Việt trên mọi máy Windows
    try:
        _st.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
HERE = Path(__file__).resolve().parent


def tim_du_lieu(p):
    """Trả về thư mục chứa train/ validation/ test/. Nếu người dùng trỏ nhầm vào thư mục bên ngoài thì tự tìm xuống tối đa 3 cấp."""
    from pathlib import Path as _P
    p = _P(str(p).strip().strip('"').strip("'")).expanduser()
    if not p.exists() and (_P.home() / p).exists():      # Jupyter 'Copy Path' cho đường dẫn tính từ thư mục nhà
        p = _P.home() / p
    p = p.resolve()
    ok = lambda d: all((d / s / "observations.json").exists() for s in ("train", "validation", "test"))
    if p.is_dir():
        if ok(p):
            return p
        level = [p]
        for _ in range(3):
            level = [c for d in level for c in sorted(d.iterdir()) if c.is_dir() and c.name not in ("images", "runs", "cache")][:200]
            hit = [d for d in level if ok(d)]
            if hit:
                print(f"(đã tự tìm thấy dữ liệu ở: {hit[0]})")
                return hit[0]
    sys.exit(f"KHÔNG THẤY DỮ LIỆU ở: {p}\nPhải là thư mục có ba thư mục con train, validation, test (mỗi cái có observations.json).")


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def keys_of(items, limit=3000):
    c = Counter()
    for x in items[:limit]:
        if isinstance(x, dict):
            c.update(x.keys())
    return sorted(c)


def values_of(items, limit=40):
    c = Counter(json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else str(x) for x in items)
    return sorted(c) if len(c) <= limit else f"{len(c)} giá trị khác nhau"


def profile(data):
    data = Path(data)
    P = {}
    P["file trong thư mục gốc"] = sorted(p.name for p in data.iterdir())
    if (data / "robots.json").exists():
        rb = load(data / "robots.json")
        P["robots.json: số robot"] = len(rb)
        P["robots.json: trường"] = keys_of(rb)
        P["robots.json: tên + gợi ý"] = [f'{r.get("robot_id")}: {r.get("name")} | legged={r.get("legged")} | {r.get("hint")}' for r in rb]
    for split in ("train", "validation", "test"):
        d = data / split
        if not d.exists():
            P[f"{split}: có thư mục"] = False
            continue
        P[f"{split}: file"] = sorted(p.name for p in d.iterdir())
        rows = load(d / "observations.json")
        P[f"{split}: trường của một dòng"] = keys_of(rows)
        ids = Counter(r.get("robot_id") for r in rows)
        P[f"{split}: các robot_id"] = sorted(ids, key=str)
        per_scene = Counter(Counter(r.get("image") for r in rows).values())
        P[f"{split}: số dòng mỗi ảnh"] = sorted(per_scene)
        P[f"{split}: đuôi ảnh"] = sorted({str(r.get("image", "")).rsplit(".", 1)[-1] for r in rows})
        if (d / "labels.json").exists():
            lab = load(d / "labels.json")
            P[f"{split}: kiểu của một nhãn"] = sorted({type(x).__name__ for x in lab})
            P[f"{split}: giá trị nhãn"] = values_of(lab)
            P[f"{split}: số nhãn / số dòng"] = "bằng nhau" if len(lab) == len(rows) else f"{len(lab)} / {len(rows)}"
        if (d / "sample_submission.json").exists():
            sm = load(d / "sample_submission.json")
            P[f"{split}: kiểu file nộp mẫu"] = type(sm).__name__ + " của " + ",".join(sorted({type(x).__name__ for x in sm})) if isinstance(sm, list) else type(sm).__name__
        if split == "test" or not (d / "scenes.json").exists():
            continue
        sc = load(d / "scenes.json")
        if split == "train":      # cấu trúc scenes.json: chỉ cần soi một split
            P["scene: trường"] = keys_of(sc)
            gv = [list(s["grid"].values()) if isinstance(s["grid"], dict) else list(s["grid"]) for s in sc]
            P["scene: kiểu của grid"] = type(sc[0]["grid"]).__name__ + (" " + ",".join(sc[0]["grid"]) if isinstance(sc[0]["grid"], dict) else "")
            P["scene: grid nằm trong 5..9"] = [min(min(g) for g in gv) >= 5, max(max(g) for g in gv) <= 9]
            P["scene: trường của node"] = keys_of([n for s in sc[:300] for n in s["nodes"]])
            E = [e for s in sc[:300] for e in s["edges"]]
            P["scene: trường của edge"] = keys_of(E)
            P["scene: edge.status"] = values_of([e.get("status") for e in E])
            P["scene: edge.stairs"] = values_of([e.get("stairs") for e in E])
            L = [l for s in sc for l in s["landmarks"]]
            P["scene: trường của landmark"] = keys_of(L)
            P["scene: loại địa điểm"] = values_of([l.get("type") for l in L])
            P["scene: số bản tối đa của một loại <= 2"] = max(max(Counter(l["type"] for l in s["landmarks"]).values()) for s in sc) <= 2
            P["scene: trường của robot"] = keys_of([s["robot"] for s in sc])
            P["scene: robot.heading"] = values_of([s["robot"].get("heading") for s in sc])
            P["scene: weather"] = values_of([s.get("weather") for s in sc])
            P["scene: style"] = values_of([s.get("style") for s in sc])
            P["scene: legend.kind"] = values_of([l.get("kind") for s in sc[:500] for l in s.get("legend", [])])
            P["scene: road_look"] = values_of([k + "->" + v for s in sc for k, v in (s.get("road_look") or {}).items()])
            P["scene: degradation: trường"] = keys_of([s.get("degradation") or {} for s in sc])
        M = [s["mission"] for s in sc]
        P[f"{split}: mission: trường"] = keys_of(M)
        for k in sorted({k for m in M for k in m}):
            if k == "text":
                continue
            vals = [m.get(k) for m in M]
            if k.endswith("_ref"):
                P[f"{split}: mission.{k}.kind"] = values_of([v.get("kind") if isinstance(v, dict) else v for v in vals])
                P[f"{split}: mission.{k}: trường"] = keys_of([v for v in vals if isinstance(v, dict)])
            else:
                P[f"{split}: mission.{k}"] = values_of(vals)
    return P


if __name__ == "__main__":
    data = tim_du_lieu(sys.argv[1])
    P = profile(data)
    if "--luu" in sys.argv:
        out = sys.argv[sys.argv.index("--luu") + 1]
        Path(out).write_text(json.dumps(P, ensure_ascii=False, indent=1), encoding="utf-8")
        print("đã lưu hồ sơ:", out)
        sys.exit()
    ref = load(HERE / "ho_so_private.json")
    n = 0
    for k in sorted(set(P) | set(ref)):
        a, b = ref.get(k, "(không có)"), P.get(k, "(không có)")
        if a == b:
            continue
        if isinstance(a, list) and isinstance(b, list) and all(isinstance(x, str) for x in a + b):
            # danh sách giá trị: chỉ giá trị MỚI mới làm code vấp; thiếu vài giá trị cũ thì không sao
            moi, thieu = [x for x in b if x not in a], [x for x in a if x not in b]
            if moi:
                n += 1
                print(f"KHÁC  {k}\n   giá trị MỚI (code chưa biết): {json.dumps(moi, ensure_ascii=False)[:600]}")
            elif thieu:
                print(f"(ít quan trọng) {k}: bộ mới không có {json.dumps(thieu, ensure_ascii=False)[:300]}")
            continue
        n += 1
        print(f"KHÁC  {k}\n   vòng private: {json.dumps(a, ensure_ascii=False)[:600]}\n   bộ mới      : {json.dumps(b, ensure_ascii=False)[:600]}")
    print(f"\n{n} điểm khác trên {len(set(P) | set(ref))} mục đã soi." + ("" if n else " Cấu trúc dữ liệu GIỐNG vòng private: chạy thẳng tu_dong.py."))
