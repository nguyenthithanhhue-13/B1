"""Soi DỮ LIỆU ĐẦU VÀO: tìm dữ liệu có vấn đề và xem test có KHÁC train / validation không, trước khi học.

Cách chạy:  python soi_du_lieu.py "<thư mục dữ liệu>"        (tu_dong.py tự gọi ở bước 0)
Không huấn luyện gì trên test: chỉ đếm và tính trung bình các đặc trưng của ảnh và câu (không cần nhãn).

In ra hai phần:
  1. DỮ LIỆU CÓ VẤN ĐỀ: thiếu file ảnh, ảnh hỏng, câu rỗng, cảnh trùng nhau, số dòng mỗi cảnh không đều, nhãn lạ...
  2. SO SÁNH ĐẶC TRƯNG giữa train / validation / test. Dòng có chữ LỆCH = test khác validation rõ rệt:
     điểm tự chấm trên validation sẽ KHÔNG đáng tin cho phần đó (điểm thật thường thấp hơn).
Kết quả cũng được lưu ở <thư mục làm việc>/soi_du_lieu.json nếu có COURIER_WORK.
"""
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

for _st in (sys.stdout, sys.stderr):
    try:
        _st.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from kiem_tra_de import tim_du_lieu      # noqa: E402

SAMPLE = int(os.environ.get("SOI_MAU", "300"))      # số ảnh lấy mẫu mỗi split để đo (đủ cho thống kê, chạy nhanh)


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def scenes_of(rows):
    out = []
    for r in rows:
        key = (r.get("image"), r.get("mission"))
        if out and out[-1][0] == key:
            out[-1][1] += 1
        else:
            out.append([key, 1])
    return out


def text_feats(text, vocab):
    words = text.lower().split()
    strip = [w.strip(".,!?:;()[]\"'") for w in words]
    strip = [w for w in strip if w]
    f = {"câu: số ký tự": len(text), "câu: số từ": len(words),
         "câu: số câu con": max(1, sum(text.count(c) for c in ".!?")),
         "câu: tỉ lệ viết KHÔNG DẤU": float(text.isascii()),
         "câu: tỉ lệ viết toàn chữ thường": float(text == text.lower()),
         "câu: có mã đơn [..#..]": float("#" in text)}
    if vocab is not None and strip:
        f["câu: tỉ lệ TỪ LẠ (không có trong train)"] = sum(w not in vocab for w in strip) / len(strip)
    return f


def image_feats(path):
    import cv2
    from PIL import Image
    im = Image.open(path)
    w, h = im.size
    g = np.asarray(im.convert("L").resize((256, 256)), dtype=np.float32)
    return {"ảnh: rộng (px)": w, "ảnh: cao (px)": h, "ảnh: tỉ lệ file JPG": float(str(path).lower().endswith((".jpg", ".jpeg"))),
            "ảnh: độ sáng trung bình": float(g.mean()), "ảnh: tỉ lệ ảnh TỐI (kiểu đêm)": float(g.mean() < 128),
            "ảnh: độ nét (càng thấp càng mờ)": float(cv2.Laplacian(g, cv2.CV_32F).var()),
            "ảnh: độ tương phản": float(g.std()), "ảnh: dung lượng (KB)": path.stat().st_size / 1024}


def scene_feats(s):
    m = s.get("mission", {})
    g = s.get("grid")
    gv = list(g.values()) if isinstance(g, dict) else list(g or [])
    lms = s.get("landmarks", [])
    E = s.get("edges", [])
    f = {"bản đồ: số hàng x cột": float(np.prod(gv)) if gv else 0.0, "bản đồ: số địa điểm": len(lms),
         "bản đồ: số đoạn đường": len(E), "bản đồ: tỉ lệ mưa": float(s.get("weather") == "rain"),
         "yêu cầu: tỉ lệ có điểm ghé": float(bool(m.get("via"))), "yêu cầu: tỉ lệ đích có tham chiếu": float(bool(m.get("goal_ref"))),
         "yêu cầu: tỉ lệ gấp": float(bool(m.get("urgent"))), "yêu cầu: tỉ lệ dễ vỡ": float(bool(m.get("fragile")))}
    if E:
        st = Counter(e.get("status") for e in E)
        for k in ("crowded", "covered", "closed"):
            f[f"bản đồ: tỉ lệ đường {k}"] = st.get(k, 0) / len(E)
        f["bản đồ: tỉ lệ đường có bậc thang"] = sum(bool(e.get("stairs")) for e in E) / len(E)
    d = s.get("degradation") or {}
    for k, v in d.items():
        if isinstance(v, (int, float)):
            f[f"ảnh (theo scenes.json): {k}"] = float(v)
    return f


def main(data):
    data = tim_du_lieu(data)
    rng = random.Random(0)
    probs, stats = [], {}
    vocab = None
    for split in ("train", "validation", "test"):
        d = data / split
        rows = load(d / "observations.json")
        sc = scenes_of(rows)
        # ---- 1. dữ liệu có vấn đề ----
        per = Counter(n for _, n in sc)
        if len(per) > 1:
            probs.append(f"{split}: số dòng mỗi cảnh KHÔNG ĐỀU {dict(per)}")
        dup = [k for k, c in Counter(k for k, _ in sc).items() if c > 1]
        if dup:
            probs.append(f"{split}: {len(dup)} cảnh (ảnh + câu) bị LẶP LẠI")
        empty = sum(1 for (img, text), _ in sc if not (text or "").strip())
        if empty:
            probs.append(f"{split}: {empty} cảnh có câu RỖNG")
        miss = [img for (img, _), _ in sc if not img or not (d / img).exists()]
        if miss:
            probs.append(f"{split}: THIẾU {len(miss)} file ảnh (vd {miss[0]})")
        if (d / "labels.json").exists():
            lab = load(d / "labels.json")
            if len(lab) != len(rows):
                probs.append(f"{split}: số nhãn ({len(lab)}) KHÁC số dòng ({len(rows)})")
            bad = Counter(str(x) for x in lab if x not in (0, 1, 2, 3))
            if bad:
                probs.append(f"{split}: có nhãn ngoài 0-3: {dict(bad)}")
            c = Counter(lab)
            if lab and max(c.values()) / len(lab) > 0.6:
                probs.append(f"{split}: nhãn LỆCH nặng {dict(c)}")
        if (d / "scenes.json").exists():
            S = load(d / "scenes.json")
            if len(S) != len(sc):
                probs.append(f"{split}: scenes.json có {len(S)} cảnh nhưng observations có {len(sc)} cảnh")
        elif split != "test":
            probs.append(f"{split}: KHÔNG có scenes.json (không học được phần đọc câu / chiến thuật theo cách cũ)")
            S = []
        else:
            S = []
        # ---- 2. đặc trưng ----
        texts = [k[1] or "" for k, _ in sc]
        if split == "train":
            vocab = {w.strip(".,!?:;()[]\"'") for t in texts for w in t.lower().split()}
        F = {}

        def add(f):
            for k, v in f.items():
                F.setdefault(k, []).append(v)

        for t in texts:
            add(text_feats(t, None if split == "train" else vocab))
        imgs = [d / k[0] for k, _ in sc if k[0] and (d / k[0]).exists()]
        broken = 0
        for p in rng.sample(imgs, min(SAMPLE, len(imgs))):
            try:
                add(image_feats(p))
            except Exception:
                broken += 1
        if broken:
            probs.append(f"{split}: {broken} ảnh KHÔNG MỞ ĐƯỢC (trong {min(SAMPLE, len(imgs))} ảnh lấy mẫu)")
        for s in S:
            try:
                add(scene_feats(s))
            except Exception:
                pass
        stats[split] = {k: (float(np.mean(v)), float(np.std(v)), len(v)) for k, v in F.items()}
        stats[split]["số cảnh"] = (float(len(sc)), 0.0, 1)

    print("\n=== 1. DỮ LIỆU CÓ VẤN ĐỀ ===")
    print("\n".join("  !!! " + p for p in probs) if probs else "  Không thấy vấn đề nào (đủ ảnh, không câu rỗng, không cảnh lặp, nhãn hợp lệ).")

    print("\n=== 2. SO SÁNH ĐẶC TRƯNG (trung bình) ===")
    print(f"  {'đặc trưng':44s} {'train':>9s} {'valid':>9s} {'test':>9s}   ghi chú")
    keys = sorted({k for s in stats.values() for k in s if k != "số cảnh"})
    shifts = []
    for k in ["số cảnh"] + keys:
        vals = [stats[s].get(k) for s in ("train", "validation", "test")]
        cell = lambda v: f"{v[0]:9.2f}" if v else f"{'-':>9s}"
        note = ""
        tr, va, te = vals
        ratio = "tỉ lệ" in k

        def far(a, b):
            if not a or not b:
                return 0.0
            if ratio:
                return abs(a[0] - b[0]) / 0.10                      # lệch 10 điểm phần trăm = 1.0
            sd = max(1e-9, (a[1] + b[1]) / 2)
            return abs(a[0] - b[0]) / sd / 0.35                     # lệch 0,35 độ lệch chuẩn = 1.0

        if k != "số cảnh":
            if far(va, te) >= 1.0:
                note = "LỆCH: test khác validation"
                shifts.append(k)
            elif far(tr, te) >= 1.0 and not va:
                note = "LỆCH: test khác train"
                shifts.append(k)
            elif far(tr, va) >= 1.0:
                note = "(validation khác train: đề đã báo trước)" if not te else "validation, test khác train"
        print(f"  {k:44s} {cell(tr)} {cell(va)} {cell(te)}   {note}")
    print()
    if shifts:
        print(f"  => {len(shifts)} đặc trưng của TEST khác VALIDATION. Điểm tự chấm trên validation sẽ lạc quan hơn điểm thật ở các phần:")
        for k in shifts:
            print("       - " + k)
        print("     Ý nghĩa: 'câu: ...' lệch -> khâu đọc câu dễ sai trên test;  'ảnh: ...' lệch -> khâu đọc ảnh dễ sai trên test.")
    else:
        print("  => Test giống validation về các đặc trưng đo được: điểm tự chấm trên validation đáng tin hơn.")
    work = os.environ.get("COURIER_WORK")
    if work:
        Path(work).mkdir(parents=True, exist_ok=True)
        (Path(work) / "soi_du_lieu.json").write_text(json.dumps({"van_de": probs, "lech": shifts, "thong_ke": stats}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1])
