# CHẠY TRÊN JUPYTERHUB (máy chủ của BTC)

JupyterHub = máy chủ của BTC, dùng qua **trình duyệt**. Không bấm `.bat` được; thay vào đó **dán lệnh vào Terminal** rồi Enter. Màn hình chạy y hệt cửa sổ đen trên máy mình.

Mọi lệnh dưới đây: bôi đen → Ctrl + C → bấm vào Terminal → **Ctrl + Shift + V** (hoặc chuột phải → Paste) → **Enter**.

---

## Bước 0. Tối nay ở nhà: đóng gói (máy Huệ, cần làm một lần)

Mở Terminal trong VS Code (hoặc cửa sổ lệnh) tại thư mục `Phenikaa_Campus_Courier_2026`, gõ:

```
& "C:\Users\Thanh Hue\.venv\Scripts\python.exe" JUPYTER\dong_goi.py
```

Sẽ có thư mục **`Desktop\GOI_JUPYTER`** gồm `kit.zip` (bộ đồ nghề) và `mo_hinh_bai2.zip` (mô hình bài 2, chỉ cần nếu máy chủ không có mạng). Chép cả thư mục này vào **USB** để dự phòng.

---

## Bước 1. Đăng nhập và mở Terminal

1. Mở trình duyệt, vào địa chỉ BTC cho, đăng nhập bằng tài khoản của đội. (Nếu được hỏi chọn máy / "server options": chọn loại **có GPU**.)
2. Màn hình chính: bên trái là **cây thư mục** (giống File Explorer), bên phải là trang **Launcher**.
3. Trong Launcher bấm **Terminal** (hoặc menu **File → New → Terminal**). Cửa sổ đen hiện ra.

## Bước 2. Đưa bộ đồ nghề lên (chọn một cách)

**Cách A — upload `kit.zip` (chắc chắn chạy được):**

1. Ở cây thư mục bên trái, bấm biểu tượng **mũi tên lên (Upload)** ở phía trên → chọn `Desktop\GOI_JUPYTER\kit.zip`. Đợi thanh tiến trình chạy xong (550 MB).
2. Dán vào Terminal:

```
cd ~ && python -m zipfile -e kit.zip ~ && ls ~/B1
```

Thấy `JUPYTER  KIT_BAI1  KIT_BAI2  du_lieu_tap_duot` là được.

**Cách B — máy chủ có mạng và có git-lfs:**

```
cd ~ && git lfs install && git clone https://github.com/nguyenthithanhhue-13/B1
```

(Nếu bước 3 báo `thiếu ... model.onnx` thì dùng cách A.)

## Bước 3. Cài đặt (một lần, 5–20 phút)

```
bash ~/B1/JUPYTER/cai_dat.sh
```

Đọc các dòng nó in:

| dòng | nghĩa |
|---|---|
| `GPU: NVIDIA ...` | có GPU, tốt. `GPU: KHÔNG CÓ` cũng chạy được, chỉ chậm hơn |
| `[ok] thư viện đủ` | cài xong thư viện |
| `CÓ MẠNG -> tải các mô hình` | tự tải mô hình bài 2, đợi |
| `KHÔNG CÓ MẠNG` | upload thêm `mo_hinh_bai2.zip` vào thư mục nhà (như bước 2A) rồi chạy lại lệnh cài đặt |
| `[ok] mô hình đọc câu của bài 1 có đủ` | bài 1 sẵn sàng |
| `[LỖI] ...` | chụp màn hình phần chữ lỗi → hỏi AI của BTC (mẫu A trong sổ tay) |

Đề là DỊCH mà máy chủ không có mạng: ở nhà chạy thêm `JUPYTER\dong_goi.py nllb`, mang `mo_hinh_nllb.zip` lên, upload rồi chạy lại lệnh cài đặt.

## Bước 4. Bài 1

1. Tìm thư mục dữ liệu bài 1 BTC phát trong cây thư mục bên trái (nếu BTC phát file zip: upload lên rồi giải nén bằng `cd ~ && python -m zipfile -e <tên file>.zip ~/du_lieu`).
2. **Chuột phải vào thư mục dữ liệu → Copy Path.**
3. Trong Terminal gõ `bash ~/B1/JUPYTER/bai1.sh ` (có dấu cách ở cuối), rồi **dán** đường dẫn vừa copy, Enter. Ví dụ:

```
bash ~/B1/JUPYTER/bai1.sh du_lieu/chung_ket
```

4. Màn hình chạy y như trên máy nhà: `KHÁC`, `LỆCH`, bảng CHẨN ĐOÁN, `>>> FILE NỘP (bản nền)`, `>>> FILE NỘP (bản NÂNG CAO)`, `XONG TẤT CẢ`. Cách đọc và cách nộp: **giống hệt sổ tay bài 1**.
5. File nộp: `B1/KIT_BAI1/runs/chungket/predictions.json`. Trong cây thư mục bên trái đi tới file đó → **chuột phải → Download** → nộp lên trang của BTC.

**Lỡ đóng trình duyệt / mất mạng:** đăng nhập lại; tab Terminal cũ thường vẫn còn và vẫn chạy (menu **View → Running Terminals and Kernels**). Mọi chữ đã hiện cũng được ghi vào file `B1/KIT_BAI1/runs/man_hinh_chungket.txt` (bấm đúp để xem). Muốn xem tiếp trực tiếp: `tail -f ~/B1/KIT_BAI1/runs/man_hinh_chungket.txt` (thoát: Ctrl + C, việc chạy không bị dừng).

**Dừng hẳn việc đang chạy:** bấm vào Terminal, nhấn **Ctrl + C**.

## Bước 5. Sửa code và học lại

- Sửa code: trong cây thư mục, bấm đúp vào file `.py` (vd `B1/KIT_BAI1/nen.py`) → nó mở ra như VS Code → sửa → **Ctrl + S**.
  Sao lưu trước khi sửa: chuột phải file → **Duplicate** (tạo bản `nen-Copy1.py`).
- Học lại (thay cho `HOC_LAI.bat`):

```
bash ~/B1/JUPYTER/hoc_lai.sh
```

- Xem sổ điểm: bấm đúp `B1/KIT_BAI1/runs/chungket/nhat_ky_diem.txt`.

## Bước 6. Bài 2

Mở **thêm một Terminal** (File → New → Terminal) để bài 1 vẫn chạy ở Terminal kia:

```
cd ~/B1/KIT_BAI2
python bai2.py
```

(khoanh vật thể: `python yolo_mau.py`; đọc chữ: `python ocr_mau.py`)

Máy hỏi từng câu y như trên máy nhà; chỉ khác là **không kéo thả được**: thay vào đó chuột phải file / thư mục trong cây bên trái → **Copy Path** → dán vào Terminal → Enter.

File nộp: `B1/KIT_BAI2/ket_qua/<tên lần chạy>/submission.csv` → chuột phải → Download.

Chạy lại không phải trả lời lại:

```
cd ~/B1/KIT_BAI2 && python bai2.py --lai ket_qua/<tên lần chạy>/cau_hinh.json
```

## Có GPU thì nhanh hơn ở đâu

| phần | dùng GPU |
|---|---|
| bài 2: đọc ảnh bằng ResNet / CLIP / DINOv2, tinh chỉnh ảnh, YOLO, nhúng câu, dịch | **có** → nhanh hơn nhiều lần; tinh chỉnh có thể đặt nhiều phút hơn |
| bài 1 | phần lớn chạy CPU (đọc ảnh bằng onnx, mô hình cây); không sao, vẫn như máy nhà |

## Thêm: notebook

Thích bấm nút hơn gõ lệnh: mở `B1/JUPYTER/CHAY_TREN_JUPYTER.ipynb`, bấm vào từng ô rồi **Shift + Enter** lần lượt từ trên xuống. Bài 2 vẫn phải chạy trong Terminal (vì máy cần hỏi đáp).
