"""Tải trước (KHI CÒN MẠNG) các mô hình ảnh mà bài 2 dùng, để lúc thi chạy offline.
Chạy một lần:  python tai_mo_hinh_cv.py      (máy Huệ đã tải rồi; máy khác muốn dùng thì chạy lệnh này ở nhà)
Cần cài trước:  pip install ultralytics easyocr
"""
import os
import sys
from pathlib import Path

for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "YOLO_OFFLINE"):
    os.environ.pop(k, None)
for _st in (sys.stdout, sys.stderr):
    try:
        _st.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
HERE = Path(__file__).resolve().parent
W = HERE / "mo_hinh"
W.mkdir(exist_ok=True)


def step(name, fn):
    try:
        fn()
        print(f"[ok]  {name}", flush=True)
    except Exception as e:
        print(f"[LỖI] {name}: {type(e).__name__}: {str(e)[:200]}", flush=True)


def tv():
    import torchvision.models as M
    for w in (M.ResNet50_Weights.DEFAULT, M.ResNet18_Weights.DEFAULT, M.MobileNet_V3_Large_Weights.DEFAULT, M.EfficientNet_B0_Weights.DEFAULT):
        w.get_state_dict(progress=False)


def hf():
    from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
    CLIPModel.from_pretrained("openai/clip-vit-base-patch32"), CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    AutoModel.from_pretrained("facebook/dinov2-small"), AutoImageProcessor.from_pretrained("facebook/dinov2-small")


def yolo():
    import shutil
    from ultralytics import YOLO
    from ultralytics.utils.checks import check_font
    cwd = os.getcwd()
    os.chdir(W)      # ultralytics tải file .pt vào thư mục hiện tại
    try:
        for n in ("yolo11n.pt", "yolo11n-seg.pt"):
            if not (W / n).exists():
                YOLO(n)
    finally:
        os.chdir(cwd)
    check_font("Arial.ttf"), check_font("Arial.Unicode.ttf")      # phông chữ để vẽ nhãn lên ảnh
    for n in ("yolo11n.pt", "yolo11n-seg.pt"):
        assert (W / n).exists(), n


def ocr():
    import easyocr
    easyocr.Reader(["vi", "en"], gpu=False, verbose=False)
    easyocr.Reader(["ch_sim", "en"], gpu=False, verbose=False)


step("torchvision: ResNet50, ResNet18, MobileNetV3, EfficientNet-B0", tv)
step("CLIP + DINOv2", hf)
step("YOLO11n (khoanh hộp) + YOLO11n-seg (tô vùng)", yolo)
step("EasyOCR: Việt + Anh, Trung + Anh", ocr)
print("Xong.")
