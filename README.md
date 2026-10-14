# Bài 1 chung kết — Campus Courier nâng cấp

Bộ đồ nghề chạy tự động cho bài 1. Hướng dẫn đầy đủ: [KIT_BAI1/SO_TAY_NGAY_THI.md](KIT_BAI1/SO_TAY_NGAY_THI.md).

## Tải về

Repo dùng Git LFS cho file mô hình 470 MB. Phải tải bằng `git clone` (không dùng nút "Download ZIP", vì ZIP không kèm file LFS):

```
git lfs install
git clone <địa chỉ repo này>
```

Kiểm tra: file `KIT_BAI1/outputs/e5_small/onnx/model.onnx` phải nặng khoảng 470 MB.

## Cài thư viện (một lần)

```
pip install -r KIT_BAI1/requirements.txt
```

## Chạy

- Tập với bộ dữ liệu nhỏ (khoảng 10 phút): bấm đúp `KIT_BAI1/TAP_DUOT.bat`.
- Ngày thi: bấm đúp `KIT_BAI1/CHAY_BAI1.bat`, kéo thả thư mục dữ liệu mới vào cửa sổ, Enter.

Các file `.bat` ưu tiên Python ở `C:\Users\Thanh Hue\.venv\Scripts\python.exe` (máy của Huệ); trên máy khác chúng dùng lệnh `python` trong PATH.

## Trong repo có gì

| thư mục | nội dung |
|---|---|
| `KIT_BAI1/` | code (gốc: lời giải vòng private COURIER2 của MinnToet), mô hình đọc ảnh, công cụ chạy tự động, sổ tay |
| `KIT_BAI2/` | khung chung cho bài 2 (bảng, văn bản, ảnh, âm thanh, chia nhóm): bấm đúp `CHAY_BAI2.bat`; hướng dẫn ở `KIT_BAI2/SO_TAY_BAI2.md`. Các mô hình có sẵn (ResNet, CLIP, mô hình nhúng câu) KHÔNG nằm trong repo: chúng ở bộ nhớ đệm của máy Huệ, máy khác phải tự tải khi còn mạng |
| `du_lieu_tap_duot/` | 300 / 40 / 30 cảnh của dữ liệu vòng private, để tập |
