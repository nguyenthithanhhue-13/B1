"""b2lib — thư viện chung cho các notebook bài 2 (ảnh + chữ, chạy trên Jupyter có GPU).

Mọi notebook B0..B6 chỉ gọi các hàm ở đây, nên notebook ngắn; sửa logic thì sửa ở file này.

QUY ƯỚC "BẢNG CHUẨN" (phần xử lý dữ liệu của Thái trả về đúng dạng này, phần mô hình của Huệ đọc dạng này):
    id     : mã dòng (để ghép với file nộp mẫu)
    split  : "train" hoặc "test"
    image  : đường dẫn TUYỆT ĐỐI tới ảnh (để trống nếu bài không có ảnh)
    text   : chữ / câu / câu hỏi (để trống nếu bài không có chữ)
    label  : nhãn (chỉ dòng train; dòng test để trống)
    ...    : các cột khác giữ nguyên (số liệu bảng, metadata)
Lưu ở  work_b2/bang_chuan.csv  (luu_bang / doc_bang_chuan).
"""
import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_OFFLINE", "0")          # trên Jupyter có mạng: cho tải mô hình (bai2.py mặc định offline)
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                  # KIT_BAI2 -> dùng lại bai2.py (chấm chéo, ghi file nộp, CLIP đoán theo tên nhãn)
import bai2                                           # noqa: E402

WORK = Path(os.environ.get("B2_WORK", HERE / "work_b2"))
WORK.mkdir(parents=True, exist_ok=True)
CACHE = WORK / "cache"
CACHE.mkdir(exist_ok=True)

IMG_MODELS = {      # tên ngắn -> mô hình Hugging Face (bộ đọc ảnh)
    "clip": "openai/clip-vit-base-patch32",
    "clip_l": "openai/clip-vit-large-patch14",
    "siglip": "google/siglip-base-patch16-224",
    "dinov2": "facebook/dinov2-small",
    "dinov2_b": "facebook/dinov2-base",
}
TXT_MODELS = {      # tên ngắn -> mô hình Hugging Face (bộ đọc chữ, đa ngôn ngữ / tiếng Việt)
    "e5": "intfloat/multilingual-e5-small",
    "e5_b": "intfloat/multilingual-e5-base",
    "bkai": "bkai-foundation-models/vietnamese-bi-encoder",
    "minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}
VLM = "Qwen/Qwen2.5-VL-3B-Instruct"     # mô hình nhìn ảnh + trả lời bằng chữ (khoảng 8 GB VRAM ở bf16)


# ======================= máy =======================
def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def gpu_info():
    """In GPU, VRAM còn trống, số nhân CPU."""
    import torch
    print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}  VRAM trống {free / 1e9:.1f} / {total / 1e9:.1f} GB")
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    print("CPU:", n, "nhân")


# ======================= đọc dữ liệu =======================
def doc_bang(path):
    """Đọc csv / tsv / xlsx / json / jsonl / parquet thành DataFrame. JSON dạng {"data": [...]} cũng đọc được."""
    p = bai2.clean_path(path)
    s = p.suffix.lower()
    if s in (".json",):
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            lists = [v for v in obj.values() if isinstance(v, list)]
            if len(lists) == 1:
                obj = lists[0]
            else:
                return pd.DataFrame([{"key": k, **(v if isinstance(v, dict) else {"value": v})} for k, v in obj.items()])
        return pd.json_normalize(obj)
    if s == ".jsonl":
        return pd.read_json(p, lines=True)
    if s == ".parquet":
        return pd.read_parquet(p)
    return bai2.read_table(p)


def tu_thu_muc(folder, co_nhan=True):
    """Thư mục ảnh -> bảng (image, label). co_nhan=True: mỗi lớp một thư mục con."""
    return bai2.folder_to_table(folder, bai2.IMG_EXT, labeled=co_nhan).rename(columns={"duong_dan": "image", "nhan": "label"})


def chuan_hoa(df, split, cot_id=None, cot_anh=None, cot_chu=None, cot_nhan=None, thu_muc_anh=None):
    """Đổi một bảng bất kỳ thành BẢNG CHUẨN (xem đầu file). cot_chu có thể là danh sách nhiều cột (sẽ nối lại).
    thu_muc_anh: thư mục gốc nếu cột ảnh chỉ ghi tên file."""
    out = pd.DataFrame(index=df.index)
    out["id"] = df[cot_id].astype(str) if cot_id else [f"{split}_{i}" for i in range(len(df))]
    out["split"] = split
    out["image"] = ""
    if cot_anh:
        root = bai2.clean_path(thu_muc_anh) if thu_muc_anh else None
        def full(v):
            v = str(v)
            q = Path(v)
            if not q.is_absolute() and root is not None:
                q = root / q
                if not q.exists() and not q.suffix:          # tên không có đuôi: thử các đuôi ảnh
                    for e in bai2.IMG_EXT:
                        if Path(str(q) + e).exists():
                            q = Path(str(q) + e)
                            break
            return str(q.resolve())
        out["image"] = df[cot_anh].map(full)
    cols = [cot_chu] if isinstance(cot_chu, str) else list(cot_chu or [])
    out["text"] = df[cols].fillna("").astype(str).agg(" . ".join, axis=1) if cols else ""
    out["label"] = df[cot_nhan] if (cot_nhan and cot_nhan in df.columns) else np.nan
    keep = [c for c in df.columns if c not in {cot_id, cot_anh, cot_nhan, *cols} and c not in out.columns]
    out = pd.concat([out, df[keep]], axis=1)
    if cot_anh:
        miss = (~out["image"].map(lambda x: Path(x).exists())).sum()
        print(f"[{split}] {len(out)} dòng; thiếu ảnh: {miss}" + ("  <-- KIỂM TRA LẠI thu_muc_anh" if miss else ""))
    else:
        print(f"[{split}] {len(out)} dòng")
    return out.reset_index(drop=True)


def luu_bang(df, ten="bang_chuan"):
    p = WORK / f"{ten}.csv"
    df.to_csv(p, index=False, encoding="utf-8")
    print("đã lưu", p, df.shape)
    return p


def doc_bang_chuan(ten="bang_chuan"):
    df = pd.read_csv(WORK / f"{ten}.csv", dtype={"id": str}, keep_default_na=True)
    for c in ("image", "text"):
        if c in df.columns:
            df[c] = df[c].fillna("")
    return df


def xem_nhanh(df):
    """Tóm tắt bảng chuẩn: số dòng, số lớp, độ dài chữ, ảnh."""
    tr = df[df.split == "train"]
    print(f"train {len(tr)} dòng | test {(df.split == 'test').sum()} dòng")
    if tr["label"].notna().any():
        vc = tr["label"].value_counts()
        print(f"{len(vc)} lớp; ít nhất {vc.min()} / nhiều nhất {vc.max()} mẫu mỗi lớp")
        print(vc.head(15).to_string())
    if (df["text"] != "").any():
        L = df["text"].str.len()
        print(f"chữ: dài trung bình {L.mean():.0f} ký tự, dài nhất {L.max()}")
    if (df["image"] != "").any():
        from PIL import Image
        smp = df[df.image != ""].sample(min(20, (df.image != "").sum()), random_state=0)
        sz = [Image.open(p).size for p in smp.image]
        print("cỡ ảnh (mẫu 20):", sorted(set(sz))[:6])


# ======================= vector đặc trưng =======================
def _key(*parts):
    return hashlib.md5("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:16]


def anh_embed(paths, ten="clip", batch=64):
    """Ảnh -> vector (mỗi ảnh một hàng). ten: clip, clip_l, siglip, dinov2, dinov2_b hoặc tên mô hình HF. Có lưu đệm."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
    name = IMG_MODELS.get(ten, ten)
    paths = list(paths)
    f = CACHE / f"img_{ten.replace('/', '_')}_{_key(name, len(paths), *paths[:3], *paths[-3:])}.npy"
    if f.exists():
        return np.load(f)
    dev = device()
    proc = AutoImageProcessor.from_pretrained(name)
    m = AutoModel.from_pretrained(name, torch_dtype=torch.float16 if dev == "cuda" else torch.float32).eval().to(dev)
    out, t0 = [], time.time()
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            ims = []
            for p in paths[i:i + batch]:
                try:
                    ims.append(Image.open(p).convert("RGB"))
                except Exception:
                    ims.append(Image.new("RGB", (224, 224)))
            x = proc(images=ims, return_tensors="pt")["pixel_values"].to(dev, m.dtype)
            if hasattr(m, "get_image_features"):        # CLIP / SigLIP: vector đã chiếu sang không gian chung với chữ
                v = m.get_image_features(pixel_values=x)
                v = v if isinstance(v, torch.Tensor) else v.pooler_output
            else:                                      # DINOv2: nối [CLS] với trung bình các mảnh
                h = m(pixel_values=x).last_hidden_state
                v = torch.cat([h[:, 0], h[:, 1:].mean(1)], 1)
            out.append(v.float().cpu().numpy())
            if (i // batch) % 20 == 0:
                print(f"  {ten}: {min(i + batch, len(paths))}/{len(paths)} ảnh ({time.time() - t0:.0f}s)", flush=True)
    X = np.concatenate(out).astype(np.float32)
    np.save(f, X)
    return X


def chu_embed(texts, ten="e5", batch=128, max_len=256):
    """Chữ -> vector đã chuẩn hoá. ten: e5, e5_b, bkai, minilm, clip_text (cùng không gian với ảnh CLIP) hoặc tên HF."""
    import torch
    texts = [str(t) for t in texts]
    name = IMG_MODELS["clip"] if ten == "clip_text" else TXT_MODELS.get(ten, ten)
    f = CACHE / f"txt_{ten.replace('/', '_')}_{_key(name, len(texts), *texts[:3], *texts[-3:])}.npy"
    if f.exists():
        return np.load(f)
    if ten == "clip_text":
        from transformers import CLIPModel, CLIPTokenizer
        tok, m = CLIPTokenizer.from_pretrained(name), CLIPModel.from_pretrained(name).eval().to(device())
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch):
                t = tok(texts[i:i + batch], padding=True, truncation=True, max_length=77, return_tensors="pt").to(device())
                v = m.get_text_features(**t)
                out.append((v if isinstance(v, torch.Tensor) else v.pooler_output).float().cpu().numpy())
        X = np.concatenate(out)
    else:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(name, device=device())
        m.max_seq_length = max_len
        pre = "query: " if "e5" in name else ""
        X = m.encode([pre + t for t in texts], batch_size=batch, show_progress_bar=False, convert_to_numpy=True)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    X = X.astype(np.float32)
    np.save(f, X)
    return X


def dac_trung(df, anh=("clip", "dinov2"), chu=("e5",)):
    """Tính mọi vector cho cả bảng chuẩn (train + test cùng lúc). Trả về {tên: ma trận}, hàng khớp thứ tự df."""
    F = {}
    has_img, has_txt = (df["image"] != "").any(), (df["text"] != "").any()
    for t in (anh if has_img else ()):
        t0 = time.time()
        F["anh_" + t] = anh_embed(df["image"].tolist(), t)
        print(f"ảnh {t}: {F['anh_' + t].shape} ({time.time() - t0:.0f}s)")
    for t in (chu if has_txt else ()):
        t0 = time.time()
        F["chu_" + t] = chu_embed(df["text"].tolist(), t)
        print(f"chữ {t}: {F['chu_' + t].shape} ({time.time() - t0:.0f}s)")
    return F


def ghep_dac_trung(F, ten):
    """Nối nhiều vector theo cột: ghep_dac_trung(F, ['anh_clip', 'chu_e5'])."""
    return np.hstack([F[t] for t in ten])


# ======================= nối với bộ làm sạch của Thái =======================
def xuat_cho_thai(X, ids, y, so_chieu=64, ten="cho_thai"):
    """Cách nối KHUYÊN DÙNG (không lẫn thư viện hai bên): ghi work_b2/<ten>.csv gồm id, pc0..pcK (vector đã PCA), label.
    Thái chạy trong môi trường của Thái:   python run.py <đường dẫn file này> --target label --device cuda
    rồi đưa lại file outputs/<ten>/clean_data.csv cho doc_sach_tu_thai()."""
    from sklearn.decomposition import PCA
    Z = PCA(n_components=min(so_chieu, X.shape[1], X.shape[0] - 1), random_state=0).fit_transform(X)
    d = pd.DataFrame(Z, columns=[f"pc{i}" for i in range(Z.shape[1])])
    d.insert(0, "id", list(ids))
    d["label"] = list(y)
    p = WORK / f"{ten}.csv"
    d.to_csv(p, index=False, encoding="utf-8")
    print(f"đã ghi {p}  ({len(d)} dòng, {Z.shape[1]} cột vector). Gửi đường dẫn này cho Thái.")
    return p


def doc_sach_tu_thai(clean_csv, ids):
    """Đọc clean_data.csv của Thái -> mặt nạ True/False cho từng id train (True = giữ lại để train)."""
    keep = set(pd.read_csv(clean_csv, dtype={"id": str}, encoding="utf-8-sig")["id"].astype(str))
    mask = np.array([str(i) in keep for i in ids])
    print(f"giữ {mask.sum()}/{len(mask)} dòng train")
    return mask



def lam_sach_thai(X, y, thu_muc_thai, so_chieu=64, contamination=0.05):
    """Đưa vector (ảnh / chữ) vào HackathonPipeline của Thái (PyOD + TabPFN + Cleanlab).
    Giảm còn `so_chieu` chiều bằng PCA trước (TabPFN chịu được ít cột). Trả về (mặt nạ dòng sạch, bảng báo cáo)."""
    from sklearn.decomposition import PCA
    sys.path.insert(0, str(bai2.clean_path(thu_muc_thai)))
    from pipeline import HackathonPipeline      # file pipeline.py trong repo của Thái
    Z = PCA(n_components=min(so_chieu, X.shape[1], X.shape[0] - 1), random_state=0).fit_transform(X)
    Xd = pd.DataFrame(Z, columns=[f"pc{i}" for i in range(Z.shape[1])])
    pipe = HackathonPipeline(contamination=contamination, report_path=str(WORK / "data_health_report.png"))
    _, _, report = pipe.fit_transform(Xd, pd.Series(np.asarray(y)))
    mask = report["is_clean_data"].to_numpy()
    print(f"sạch {mask.sum()}/{len(mask)} | outlier {report['is_outlier'].sum()} | nghi sai nhãn {report['is_label_issue'].sum()} | mô hình {pipe.model_used_}")
    return mask, report


# ======================= mô hình nhanh trên vector =======================
def mo_hinh_nhanh(F_train, F_test, y, task=None, chi_so=None):
    """F_train / F_test: {tên: ma trận}. Chấm chéo 5 phần mọi (vector x mô hình) + trung bình top 3, chọn cái tốt nhất.
    Trả về (dự đoán test, xác suất test hoặc None, bảng điểm, tên được chọn). Dùng lại bai2.fit_predict."""
    task = task or bai2.guess_task(np.asarray(y))
    sets = {k: (F_train[k], F_test[k], "san" if k.startswith("san_") else "day") for k in F_train}
    return bai2.fit_predict(sets, np.asarray(y), task, chi_so)


def clip_doan_theo_ten(img_train, img_test, classes, mau="a photo of a {}"):
    """CLIP đoán nhãn chỉ từ tên nhãn (không học). img_*: vector anh_clip. Trả về (xác suất train, xác suất test)."""
    return bai2.clip_zero_shot(img_train, img_test, classes) if mau == "a photo of a {}" else _clip_zs(img_train, img_test, classes, mau)


def _clip_zs(a, b, classes, mau):
    t = chu_embed([mau.format(c) for c in classes], "clip_text")
    def pr(F):
        F = F / np.linalg.norm(F, axis=1, keepdims=True)
        z = 100 * F @ t.T
        z = np.exp(z - z.max(1, keepdims=True))
        return z / z.sum(1, keepdims=True)
    return pr(a), pr(b)


def khop_anh_chu(img_vec, txt_vec, top=5):
    """Bài GHÉP ẢNH <-> CHỮ (retrieval): mỗi chữ tìm top ảnh giống nhất. Dùng anh_clip + chu clip_text (cùng không gian).
    Trả về (chỉ số top ảnh cho mỗi chữ [n_chữ x top], ma trận điểm)."""
    a = img_vec / np.linalg.norm(img_vec, axis=1, keepdims=True)
    b = txt_vec / np.linalg.norm(txt_vec, axis=1, keepdims=True)
    S = b @ a.T
    return np.argsort(-S, axis=1)[:, :top], S


# ======================= tinh chỉnh (finetune) trên GPU =======================
def tinh_chinh(train_df, test_df, anh="clip", chu="e5", epochs=5, lr=3e-5, lr_dau=1e-3, batch=32, max_len=128,
               val=0.15, phut=40, dong_bang_anh=False, dong_bang_chu=False, seed=0):
    """Tinh chỉnh mô hình ẢNH + CHỮ (hoặc chỉ một trong hai) cho bài PHÂN LOẠI, chạy GPU (fp16), dừng khi hết `phut` phút.
    anh / chu: tên ngắn (xem IMG_MODELS, TXT_MODELS) hoặc None để bỏ phần đó.
    Trả về (điểm accuracy trên phần giữ lại, xác suất test [n_test x n_lớp], các lớp, đường dẫn mô hình đã lưu)."""
    import torch
    import torch.nn as nn
    from PIL import Image
    from sklearn.model_selection import train_test_split
    from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
    torch.manual_seed(seed)
    dev = device()
    use_img = bool(anh) and (train_df["image"] != "").any()
    use_txt = bool(chu) and (train_df["text"] != "").any()
    assert use_img or use_txt, "cần ít nhất ảnh hoặc chữ"
    classes, yi = np.unique(train_df["label"].astype(str), return_inverse=True)
    idx = np.arange(len(yi))
    strat = yi if pd.Series(yi).value_counts().min() >= 2 else None
    tr_i, va_i = train_test_split(idx, test_size=val, random_state=seed, stratify=strat)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            d = 0
            if use_img:
                full = AutoModel.from_pretrained(IMG_MODELS.get(anh, anh))
                self.img = getattr(full, "vision_model", full)          # CLIP / SigLIP: chỉ lấy nhánh ảnh
                self.img_proj = getattr(full, "visual_projection", None)
                d += self.img_proj.out_features if self.img_proj is not None else self.img.config.hidden_size
                if dong_bang_anh:
                    for p in self.img.parameters():
                        p.requires_grad = False
            if use_txt:
                self.txt = AutoModel.from_pretrained(TXT_MODELS.get(chu, chu))
                d += self.txt.config.hidden_size
                if dong_bang_chu:
                    for p in self.txt.parameters():
                        p.requires_grad = False
            self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.2), nn.Linear(d, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, len(classes)))

        def forward(self, pix=None, ids=None, att=None):
            z = []
            if use_img:
                o = self.img(pixel_values=pix)
                v = o.pooler_output if getattr(o, "pooler_output", None) is not None else o.last_hidden_state[:, 0]
                z.append(self.img_proj(v) if self.img_proj is not None else v)
            if use_txt:
                h = self.txt(input_ids=ids, attention_mask=att).last_hidden_state
                m = att.unsqueeze(-1).to(h.dtype)
                z.append((h * m).sum(1) / m.sum(1).clamp(min=1))
            return self.head(torch.cat(z, 1))

    net = Net().to(dev)
    proc = AutoImageProcessor.from_pretrained(IMG_MODELS.get(anh, anh)) if use_img else None
    tok = AutoTokenizer.from_pretrained(TXT_MODELS.get(chu, chu)) if use_txt else None
    pre = "query: " if (use_txt and "e5" in TXT_MODELS.get(chu, chu)) else ""

    def batch_of(df, rows, train):
        b = {}
        if use_img:
            ims = []
            for p in df["image"].iloc[rows]:
                try:
                    im = Image.open(p).convert("RGB")
                except Exception:
                    im = Image.new("RGB", (224, 224))
                if train and np.random.rand() < 0.5:
                    im = im.transpose(Image.FLIP_LEFT_RIGHT)      # tăng cường: lật ngang (bỏ dòng này nếu ảnh có chữ / hướng quan trọng)
                ims.append(im)
            b["pix"] = proc(images=ims, return_tensors="pt")["pixel_values"].to(dev)
        if use_txt:
            t = tok([pre + s for s in df["text"].iloc[rows].astype(str)], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            b["ids"], b["att"] = t["input_ids"].to(dev), t["attention_mask"].to(dev)
        return b

    head_params = list(net.head.parameters()) + (list(net.img_proj.parameters()) if use_img and net.img_proj is not None else [])
    hid = {id(p) for p in head_params}
    body = [p for p in net.parameters() if p.requires_grad and id(p) not in hid]
    opt = torch.optim.AdamW([{"params": body, "lr": lr}, {"params": head_params, "lr": lr_dau}], weight_decay=0.01)
    steps = epochs * (len(tr_i) // batch + 2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr, lr_dau], total_steps=steps + 1, pct_start=0.1)
    use_amp = dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)

    def predict(df, rows):
        net.eval()
        out = []
        with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=use_amp):
            for i in range(0, len(rows), batch * 2):
                out.append(net(**batch_of(df, rows[i:i + batch * 2], False)).float().softmax(1).cpu().numpy())
        return np.concatenate(out)

    t0, best, best_state, rng = time.time(), -1.0, None, np.random.default_rng(seed)
    for ep in range(epochs):
        net.train()
        order = rng.permutation(tr_i)
        for i in range(0, len(order) - 1, batch):
            rows = order[i:i + batch]
            with torch.autocast(dev, dtype=torch.float16, enabled=use_amp):
                loss = lossf(net(**batch_of(train_df, rows, True)), torch.as_tensor(yi[rows], device=dev))
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if (time.time() - t0) / 60 > phut:
                break
        acc = float((predict(train_df, va_i).argmax(1) == yi[va_i]).mean())
        print(f"vòng {ep + 1}/{epochs}: accuracy phần giữ lại = {acc:.4f}  loss cuối = {loss.item():.3f}  ({(time.time() - t0) / 60:.1f} phút)", flush=True)
        if acc > best:
            best, best_state = acc, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        if (time.time() - t0) / 60 > phut:
            print(f"dừng: hết {phut} phút")
            break
    net.load_state_dict(best_state)
    save = WORK / f"tinh_chinh_{anh}_{chu}.pt"
    torch.save({"state": best_state, "classes": classes.tolist(), "anh": anh, "chu": chu}, save)
    tp = predict(test_df, np.arange(len(test_df)))
    print(f"TỐT NHẤT: accuracy phần giữ lại = {best:.4f}. Đã lưu {save}")
    return best, tp, classes, save


# ======================= mô hình nhìn ảnh + trả lời chữ (VQA, mô tả ảnh, đọc chữ trong ảnh) =======================
def vlm_tra_loi(paths, cau_hoi, model=VLM, max_new_tokens=64, batch=8, he_thong=None):
    """Hỏi mô hình thị giác-ngôn ngữ về từng ảnh. cau_hoi: một câu dùng cho mọi ảnh, hoặc danh sách (mỗi ảnh một câu).
    Không cần học. Dùng cho: hỏi đáp về ảnh (VQA), mô tả ảnh (caption), đọc chữ trong ảnh, phân loại theo mô tả.
    Trả về danh sách câu trả lời (chuỗi)."""
    import torch
    from transformers import pipeline
    paths = list(paths)
    qs = [cau_hoi] * len(paths) if isinstance(cau_hoi, str) else list(cau_hoi)
    pipe = pipeline("image-text-to-text", model=model, device_map="auto",
                    torch_dtype=torch.bfloat16 if device() == "cuda" else torch.float32)
    out, t0 = [], time.time()
    for i in range(0, len(paths), batch):
        msgs = []
        for p, q in zip(paths[i:i + batch], qs[i:i + batch]):
            m = [{"role": "user", "content": [{"type": "image", "url": str(p)}, {"type": "text", "text": q}]}]
            if he_thong:
                m.insert(0, {"role": "system", "content": [{"type": "text", "text": he_thong}]})
            msgs.append(m)
        res = pipe(text=msgs, max_new_tokens=max_new_tokens, return_full_text=False)
        out += [r[0]["generated_text"].strip() if isinstance(r, list) else r["generated_text"].strip() for r in res]
        print(f"  {min(i + batch, len(paths))}/{len(paths)} ảnh ({time.time() - t0:.0f}s)", flush=True)
    return out


# ======================= ghép và nộp =======================
def luu_xac_suat(ten, ids, proba, classes):
    """Lưu xác suất test của một cách làm, để B6 ghép."""
    p = WORK / f"xs_{ten}.csv"
    d = pd.DataFrame(np.asarray(proba), columns=[str(c) for c in classes])
    d.insert(0, "id", list(ids))
    d.to_csv(p, index=False, encoding="utf-8")
    print("đã lưu", p)
    return p


def ghep_xac_suat(ten_va_trong_so):
    """ten_va_trong_so: {tên: trọng số}. Đọc work_b2/xs_<tên>.csv, cộng có trọng số. Trả về (ids, bảng xác suất)."""
    tong, ids = None, None
    for ten, w in ten_va_trong_so.items():
        d = pd.read_csv(WORK / f"xs_{ten}.csv", dtype={"id": str})
        ids = d["id"] if ids is None else ids
        P = d.drop(columns="id")
        tong = P * w if tong is None else tong.add(P * w, fill_value=0)
    tong = tong.div(sum(ten_va_trong_so.values()))
    return ids, tong


def ghi_nop(ids, du_doan, xac_suat=None, file_mau=None, cot_id="id", cot_nhan="label", ten="nop"):
    """Ghi file nộp theo đúng file mẫu của đề (nếu có). Ra work_b2/<ten>/submission.csv."""
    return bai2.write_submission(WORK / ten, list(ids), list(du_doan), xac_suat, cot_id, cot_nhan, file_mau)
