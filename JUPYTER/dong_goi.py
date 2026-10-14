"""ĐÓNG GÓI để mang lên JupyterHub (chạy trên MÁY NHÀ, máy Huệ):   python JUPYTER/dong_goi.py
Tạo trong thư mục Desktop\\GOI_JUPYTER:
  bai2.zip                (~30 MB)  chỉ KIT_BAI2 (bài 2), giải nén ra ~/B1/KIT_BAI2
  kit.zip                 (~550 MB) KIT_BAI1 + KIT_BAI2 + JUPYTER + du_lieu_tap_duot  -> giải nén thành ~/B1 trên máy chủ
  mo_hinh_bai2.zip        (~3,7 GB) mô hình bài 2: CLIP, DINOv2, ResNet/MobileNet, EasyOCR, e5, vietnamese-bi-encoder, MiniLM, opus-mt
  mo_hinh_nllb.zip        (~2,4 GB) mô hình dịch NLLB-200 (chỉ cần nếu đề là DỊCH và máy chủ không có mạng)
Chỉ cần hai file mô hình khi máy chủ KHÔNG có mạng.
"""
import os
import sys
import zipfile
from pathlib import Path

for _st in (sys.stdout, sys.stderr):
    try:
        _st.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
B1 = Path(__file__).resolve().parent.parent
HOME = Path.home()
OUT = HOME / "Desktop" / "GOI_JUPYTER"
OUT.mkdir(parents=True, exist_ok=True)
HF_SKIP_DIRS = {"onnx", "openvino"}      # bản xuất sang định dạng khác của mô hình, bài 2 không dùng
SKIP_DIRS = {"runs", "ket_qua", "cache", "__pycache__", ".git", "work_b2"}
HF = HOME / ".cache" / "huggingface" / "hub"
BAI2_HF = ["openai--clip-vit-base-patch32", "facebook--dinov2-small", "intfloat--multilingual-e5-small",
           "bkai-foundation-models--vietnamese-bi-encoder", "sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2",
           "Helsinki-NLP--opus-mt-zh-vi", "Helsinki-NLP--opus-mt-en-vi", "Helsinki-NLP--opus-mt-vi-en", "Helsinki-NLP--opus-mt-en-zh"]


def add_tree(z, src, arc, skip=SKIP_DIRS):
    n = 0
    main = (src / "refs" / "main").read_text().strip() if (src / "refs" / "main").exists() else None      # bản mô hình đang dùng
    for root, dirs, files in os.walk(src):
        if main and Path(root) == src / "snapshots":
            dirs[:] = [d for d in dirs if d == main]      # bỏ các bản tải về ở lần trước
        dirs[:] = [d for d in dirs if d not in skip and not d.startswith("blobs") and not (".cache" in arc.parts and d in HF_SKIP_DIRS)]
        for f in files:
            p = Path(root) / f
            if p.suffix == ".zip" or f.endswith(".incomplete") or f.endswith(".lock"):
                continue
            if ".cache" in arc.parts and (p.suffix in (".h5", ".msgpack", ".ot") or (f == "pytorch_model.bin" and (p.parent / "model.safetensors").exists())):
                continue      # bản trùng định dạng khác (TensorFlow, Flax, Rust, hoặc .bin khi đã có .safetensors)
            z.write(p, (arc / p.relative_to(src)).as_posix())
            n += 1
    return n


def make(name, items):
    p = OUT / name
    print(f"đang tạo {p} ...", flush=True)
    with zipfile.ZipFile(p, "w", zipfile.ZIP_STORED, allowZip64=True) as z:      # không nén: mô hình gần như không nén được, làm nhanh hơn
        for src, arc in items:
            if src.exists():
                print(f"   + {src}  ({add_tree(z, src, Path(arc))} file)", flush=True)
            else:
                print(f"   !!! không thấy {src} (bỏ qua)")
    print(f"[ok] {p}  ({p.stat().st_size / 1e9:.2f} GB)\n", flush=True)


which = sys.argv[1:] or ["kit", "goi_bai2", "bai2"]
if "kit" in which:
    make("kit.zip", [(B1 / d, f"B1/{d}") for d in ("KIT_BAI1", "KIT_BAI2", "JUPYTER", "du_lieu_tap_duot")])
if "goi_bai2" in which:      # gói nhỏ chỉ có bài 2 (giải nén ra cùng chỗ ~/B1/KIT_BAI2 như kit.zip)
    make("bai2.zip", [(B1 / "KIT_BAI2", "B1/KIT_BAI2")])
if "bai2" in which:
    make("mo_hinh_bai2.zip", [(HF / f"models--{m}", f".cache/huggingface/hub/models--{m}") for m in BAI2_HF]
         + [(HOME / ".cache" / "torch" / "hub" / "checkpoints", ".cache/torch/hub/checkpoints"), (HOME / ".EasyOCR" / "model", ".EasyOCR/model")])
if "nllb" in which:
    make("mo_hinh_nllb.zip", [(HF / "models--facebook--nllb-200-distilled-600M", ".cache/huggingface/hub/models--facebook--nllb-200-distilled-600M")])
print(f"Xong. Các file ở: {OUT}")
print("Muốn thêm NLLB (dịch):  python JUPYTER/dong_goi.py nllb")
