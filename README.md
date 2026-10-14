# Bộ đồ nghề chung kết Phenikaa AI Hackathon 2026

## 👉 Đọc hướng dẫn trước

**[Bấm vào đây để tải file Word hướng dẫn: KIT_BAI1/HUONG_DAN_THI.docx](KIT_BAI1/HUONG_DAN_THI.docx)**
→ trang mở ra thì bấm nút **Download** (biểu tượng mũi tên xuống ⬇, góc phải) → mở file bằng Word.

File Word gồm: tóm tắt nhanh ngày thi, sổ tay bài 1, sổ tay bài 2. Muốn đọc ngay trên web thì bấm:

- [Sổ tay bài 1](KIT_BAI1/SO_TAY_NGAY_THI.md)
- [Sổ tay bài 2](KIT_BAI2/SO_TAY_BAI2.md)

---

## Tải cả bộ về máy

Có hai cách. Không biết dùng `git` thì chọn cách 1.

### Cách 1: GitHub Desktop (bấm chuột, không gõ lệnh)

1. Tải và cài **GitHub Desktop**: https://desktop.github.com
2. Mở GitHub Desktop → **File** → **Clone repository** → tab **URL**.
3. Dán địa chỉ: `https://github.com/nguyenthithanhhue-13/B1`
4. Ô **Local path**: chọn chỗ lưu (vd `Downloads`) → bấm **Clone**. Đợi tải xong (khoảng 600 MB).
5. Lần sau muốn lấy bản mới nhất: mở GitHub Desktop → bấm **Fetch origin** → **Pull origin**.

### Cách 2: gõ lệnh

```
git lfs install
git clone https://github.com/nguyenthithanhhue-13/B1
```

Lấy bản mới nhất: vào thư mục `B1` rồi gõ `git pull`.

**Không dùng nút "Download ZIP"** (nút xanh **Code** → Download ZIP): ZIP không có file mô hình 470 MB nên **bài 1 không chạy được**. Kiểm tra: file `KIT_BAI1\outputs\e5_small\onnx\model.onnx` phải nặng khoảng 470 MB.

---

## Cài đặt trên máy KHÁC (không phải máy Huệ) — một lần, cần mạng

Máy Huệ đã cài sẵn mọi thứ, không cần làm bước này.

1. Cài **Python 3.13** từ https://www.python.org/downloads/ → khi cài **tick ô "Add python.exe to PATH"**.
2. Mở thư mục `KIT_BAI2` trong File Explorer → **bấm đúp `CAI_DAT_MAY_KHAC.bat`**.
3. Đợi 20–40 phút (tải thư viện và mô hình, khoảng 3 GB). Thấy `XONG` là xong. Có dòng `[LỖI]` thì bấm lại file đó lần nữa.

---

## Nếu BTC bắt chạy trên JupyterHub (máy chủ qua trình duyệt)

Xem **[JUPYTER/HUONG_DAN_JUPYTER.md](JUPYTER/HUONG_DAN_JUPYTER.md)**: không bấm `.bat`, mà dán lệnh vào Terminal của JupyterHub:

```
bash ~/B1/JUPYTER/cai_dat.sh                      # cài đặt, một lần
bash ~/B1/JUPYTER/bai1.sh <thư mục dữ liệu>       # bài 1
bash ~/B1/JUPYTER/hoc_lai.sh                      # học lại sau khi sửa code
cd ~/B1/KIT_BAI2 && python bai2.py                # bài 2 (yolo_mau.py, ocr_mau.py)
```

Bộ đồ nghề mang lên bằng `kit.zip` (tạo ở nhà bằng `python JUPYTER/dong_goi.py`, nằm ở `Desktop\GOI_JUPYTER`).

---

## Ngày thi (chạy trên máy mình): bấm file nào

Mọi file `.bat` đều **bấm đúp trong File Explorer** (không bấm trong VS Code: ở đó nó chỉ mở ra chữ, không chạy).

| việc | thư mục | bấm đúp |
|---|---|---|
| **Bài 1**: chạy cả bài | `KIT_BAI1` | `CHAY_BAI1.bat` → kéo thả thư mục dữ liệu vào cửa sổ đen → Enter |
| Bài 1: học lại sau khi sửa code | `KIT_BAI1` | `HOC_LAI.bat` |
| Bài 1: tập thử ở nhà (10–20 phút) | `KIT_BAI1` | `TAP_DUOT.bat` |
| **Bài 2**: bảng, văn bản, phân loại ảnh, âm thanh, chia nhóm, video, dịch | `KIT_BAI2` | `CHAY_BAI2.bat` |
| Bài 2: khoanh / đếm vật thể, tô vùng ảnh | `KIT_BAI2` | `CHAY_YOLO.bat` |
| Bài 2: đọc chữ trong ảnh | `KIT_BAI2` | `CHAY_OCR.bat` |
| Bài 2: chạy lại không phải trả lời lại | `KIT_BAI2` | kéo `ket_qua\<tên>\cau_hinh.json` thả lên `CHAY_LAI_BAI2.bat` (hoặc `CHAY_LAI_YOLO.bat`) |

File để nộp:

- Bài 1: `KIT_BAI1\runs\chungket\predictions.json`
- Bài 2: `KIT_BAI2\ket_qua\<tên lần chạy>\submission.csv`

Cách tính điểm: mỗi bài lấy **lần nộp cao nhất**, điểm cuối là **trung bình 2 bài** → nộp nhanh một bản có điểm cho cả hai bài trước, rồi mới cải tiến.

---

## Trong repo có gì

| thư mục | nội dung |
|---|---|
| `KIT_BAI1/` | bài 1: code (gốc là lời giải vòng private COURIER2 của MinnToet), mô hình đọc ảnh, công cụ chạy tự động, sổ tay, file Word hướng dẫn |
| `KIT_BAI2/` | bài 2: khung chung cho 9 kiểu bài (`bai2.py`, `yolo_mau.py`, `ocr_mau.py`), sổ tay, file cài đặt cho máy khác. Mô hình YOLO nằm trong `KIT_BAI2/mo_hinh/`; các mô hình khác tải bằng `CAI_DAT_MAY_KHAC.bat` |
| `JUPYTER/` | lệnh chạy trên JupyterHub, notebook, script đóng gói |
| `du_lieu_tap_duot/` | 300 / 40 / 30 cảnh của dữ liệu vòng private, để tập bài 1 |
