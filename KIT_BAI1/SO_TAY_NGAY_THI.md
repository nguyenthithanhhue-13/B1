# SỔ TAY BÀI 1 — Campus Courier nâng cấp

Không cần biết code. Làm lần lượt từ trên xuống.

Thư mục làm việc trên máy này: `C:\Users\Thanh Hue\Downloads\Phenikaa_Campus_Courier_2026\KIT_BAI1`

---

## PHẦN A. Làm bài (trường hợp bình thường)

### Bước 1. Lấy dữ liệu BTC phát

1. Tải file dữ liệu về. Nếu là file `.zip`: bấm chuột phải → **Extract All** → **Extract**.
2. Mở thư mục vừa giải nén, tìm tới thư mục có ba thư mục con tên **`train`**, **`validation`**, **`test`**. Đó là **thư mục bài mới**.
   (Trỏ nhầm ra thư mục bên ngoài một hai cấp cũng được, script tự tìm.)

### Bước 2. Chạy

1. Mở thư mục `KIT_BAI1` bằng File Explorer.
2. Bấm đúp file **`CHAY_BAI1.bat`**. Một cửa sổ đen hiện ra.
3. Dùng chuột **kéo thư mục bài mới thả vào cửa sổ đen**. Đường dẫn tự hiện ra.
4. Bấm **Enter**.
5. **Không tắt cửa sổ đen.** Cắm sạc, để máy chỗ thoáng.

### Bước 3. Theo dõi và nộp

Cửa sổ đen sẽ lần lượt hiện các dòng sau. Số ở đầu dòng là số phút đã trôi qua.

| dòng hiện ra | nghĩa là | em làm gì |
|---|---|---|
| `0 soi dữ liệu mới` và các dòng `KHÁC` | bài mới có gì lạ so với vòng trước | không làm gì, chỉ để biết |
| `0b soi dữ liệu đầu vào`: bảng `DỮ LIỆU CÓ VẤN ĐỀ` và `SO SÁNH ĐẶC TRƯNG` | kiểm tra dữ liệu BTC phát; dòng có chữ `LỆCH` = test khác validation ở điểm đó | đọc mục "Đọc bảng soi dữ liệu" ngay dưới bảng này |
| `1 đọc ảnh ... đang chạy nền` | máy đang đọc 1.500 ảnh | chờ |
| `2b học ...`, `2c học ...` | máy học lại trên bài mới | chờ |
| `chờ đọc ảnh validation xong` | đang chờ đọc ảnh | chờ (bước lâu nhất) |
| bảng `=== CHẨN ĐOÁN ===` | điểm tự chấm | xem dòng `(4) ... = ĐIỂM THẬT` |
| **`>>> FILE NỘP (bản nền): ...`** | có file nộp thứ nhất | **nộp ngay** (xem dưới) |
| `5 nâng cao ...` | máy tự học tiếp cho điểm cao hơn | chờ 25–45 phút |
| **`>>> FILE NỘP (bản NÂNG CAO): ...`** | có file nộp tốt hơn | **nộp lại** |
| `XONG TẤT CẢ` | xong bài 1 | chuyển sang bài 2 |

**Đọc bảng soi dữ liệu** (hiện ngay đầu, sau khoảng 10 giây):

- Phần `1. DỮ LIỆU CÓ VẤN ĐỀ`: nếu ghi "Không thấy vấn đề nào" thì yên tâm. Nếu có dòng `!!!` (thiếu ảnh, ảnh hỏng, câu rỗng, nhãn lạ, thiếu `scenes.json`...): dữ liệu tải về có thể bị lỗi → **tải lại / giải nén lại** rồi chạy lại. Vẫn còn thì báo BTC.
- Phần `2. SO SÁNH ĐẶC TRƯNG`: ba cột là trung bình của train, validation, test. Dòng ghi `LỆCH: test khác validation` nghĩa là bài test khó hơn / khác bài tự chấm ở điểm đó, nên **điểm trên web sẽ thấp hơn điểm tự chấm**. Không phải lỗi, không cần sửa gì; chỉ để không bất ngờ.
  - Dòng `câu: ...` lệch → phần hiểu câu sẽ sai nhiều hơn trên test.
  - Dòng `ảnh: ...` lệch → phần đọc bản đồ sẽ sai nhiều hơn trên test.
- Ví dụ thật: ở vòng private, câu trong test dài hơn validation (265 so với 233 ký tự), nhiều câu không dấu hơn (41% so với 34%); đó là một lý do điểm web (0,83–0,87) thấp hơn điểm tự chấm (0,93).

**Cách nộp:** vào trang nộp bài của BTC → chọn file
`C:\Users\Thanh Hue\Downloads\Phenikaa_Campus_Courier_2026\KIT_BAI1\runs\chungket\predictions.json`
Hai lần nộp đều là file này (cùng tên, lần sau nội dung mới). Mỗi lần nộp xong ghi lại điểm ra giấy.

**Thời gian** (đo trên dữ liệu vòng private):

| | khoảng |
|---|---|
| tới file nộp bản nền | 30–40 phút (riêng phần đọc 1.200 ảnh test chưa đo được, có thể lâu hơn) |
| tới file nộp bản nâng cao | thêm 25–45 phút |

**Điểm tự chấm đã đo trên dữ liệu vòng private:** bản nền 0,79 · bản nâng cao 0,90.
Điểm trên web sẽ **thấp hơn** điểm tự chấm (vòng private: tự chấm 0,93 nhưng web 0,83–0,87). Thấy thấp hơn là bình thường.

---

## PHẦN B. Khi có trục trặc

### B1. Máy quá nóng hoặc đơ

1. Bấm vào cửa sổ đen, nhấn **Ctrl + C**, rồi đóng cửa sổ.
2. Bấm chuột phải file `CHAY_BAI1.bat` → **Edit** (mở bằng Notepad).
3. Tìm dòng `if not exist "%PY%" set "PY=python"`. Thêm ngay **bên dưới** nó một dòng mới: `set PROCS=4`
4. Lưu (Ctrl + S), đóng Notepad, bấm đúp lại `CHAY_BAI1.bat`. Những bước đã xong không phải làm lại.

### B2. Lỡ tắt cửa sổ, máy tắt, mất điện

Bấm đúp lại `CHAY_BAI1.bat`, kéo thả lại thư mục bài mới, Enter. Bước nào đã xong sẽ hiện `đã có, bỏ qua`.

### B3. Hiện dòng `!!! BƯỚC LỖI`

Dòng đó có dạng: `!!! BƯỚC LỖI: <tên bước>. Log đầy đủ: C:\...\runs\chungket\log\<tên>.log`

1. Mở File Explorer, vào `KIT_BAI1\runs\chungket\log\`.
2. Bấm đúp file `.log` có tên giống trong dòng báo lỗi (mở bằng Notepad).
3. Nhấn **Ctrl + End** để xuống cuối file. Đọc **dòng cuối cùng**, tra bảng:

| dòng cuối có chữ | cách xử lý |
|---|---|
| `KHÔNG THẤY DỮ LIỆU` | kéo thả nhầm thư mục. Chạy lại, kéo đúng thư mục có `train`, `validation`, `test` |
| `Thiếu ... model.onnx` | thiếu file `KIT_BAI1\outputs\e5_small\onnx\model.onnx`. Chép lại từ bản sao lưu của `KIT_BAI1` |
| `No module named ...` | đang dùng nhầm bản Python. Phải chạy bằng `CHAY_BAI1.bat`, không tự gõ `python` |
| `MemoryError` | làm theo B1 nhưng dùng `set PROCS=3` |
| lỗi khác (`KeyError`, `TypeError`, `IndexError`, `AssertionError`...) | bài mới có thứ code chưa hiểu → sang **Phần C** |

**Lỗi ở bước `5 nâng cao` thì không sao:** file nộp bản nền vẫn còn và đã hợp lệ. Cứ để nguyên, làm bài 2.

### B4. Không có file nộp nào và không sửa được

Dùng AI của BTC theo Phần C, tối đa 2–3 lượt hỏi. Vẫn không được thì dừng, chuyển sang bài 2.

---

## PHẦN C. Dùng AI của BTC để sửa code (token có hạn)

**Chỉ dùng theo thứ tự ưu tiên này:**

1. Có `!!! BƯỚC LỖI` và **chưa có file nộp nào** → dùng ngay, mẫu A.
2. Đề có thứ mới làm điểm rất thấp → mẫu B.
3. Đã nộp bản nâng cao, bài 2 đã ổn, muốn thêm điểm → mẫu C.

### C1. Lấy đúng thứ cần dán cho AI (dán ít để tiết kiệm token)

**20 dòng cuối của file log:** mở file log (B3), Ctrl + End, dùng chuột bôi đen khoảng 20 dòng cuối, Ctrl + C.

**Đoạn code quanh chỗ lỗi:** trong 20 dòng đó, tìm dòng **dưới cùng** có dạng
`File "C:\...\KIT_BAI1\<tên file>.py", line <số>, in <tên hàm>`
→ ghi lại **tên file**, **số dòng**, **tên hàm**. Rồi:

1. Mở VS Code → **File → Open Folder** → chọn `KIT_BAI1`.
2. Nhấn **Ctrl + P**, gõ tên file (ví dụ `nen.py`), Enter.
3. Nhấn **Ctrl + G**, gõ số dòng, Enter. Con trỏ nhảy tới dòng lỗi.
4. Bôi đen cả hàm chứa dòng đó: từ dòng bắt đầu bằng `def <tên hàm>` tới hết hàm (trước dòng `def` kế tiếp). Ctrl + C.

**Không bao giờ dán cả file.** Riêng `src\nlp2.py` dài hơn 4.000 dòng.

### C2. Ba mẫu câu hỏi (chép nguyên, thay phần trong `< >`)

**Mẫu A — script báo lỗi:**

```
Tôi chạy một pipeline Python và bị lỗi. 20 dòng cuối của log:
<dán log>

Hàm chứa dòng lỗi (file <tên file>, dòng <số>):
<dán hàm>

Chỉ ra nguyên nhân và cách sửa ngắn nhất. Trả lời ngắn: chỉ đưa lại hàm đã sửa, đầy đủ, để tôi dán đè.
```

**Mẫu B — đề có thứ mới:**

```
Bài toán: đoán bước đi đầu tiên (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT) của robot trên bản đồ lưới, từ ảnh + câu tiếng Việt.
Code viết cho dữ liệu cũ. Dữ liệu mới khác ở:
<dán các dòng KHÁC ở đầu cửa sổ đen>

Một phần tử "mission" trong validation/scenes.json mới:
<dán phần "mission": {...} của một cảnh>

Ba hàm đọc dữ liệu trong nen.py:
<dán hàm true_legs, norm_for_parser, scene_feats>

Sửa ba hàm này để dùng được phần mới. Trả lời ngắn: chỉ đưa lại các hàm đã sửa, đầy đủ, để tôi dán đè.
```

Cách lấy một phần tử `mission`: trong VS Code mở file `validation\scenes.json` của bài mới, nhấn Ctrl + F, gõ `"mission"`, bôi đen từ `"mission": {` tới dấu `}` đóng của nó.

**Mẫu C — tăng điểm một robot yếu:**

```
Tôi đoán bước đi của robot bằng HistGradientBoosting trên đặc trưng của 4 hướng đi.
Robot <số> chỉ đúng <điểm> trên validation. Gợi ý của đề: "<dán dòng hint của robot đó trong robots.json>".

Hàm tạo đặc trưng:
<dán hàm scene_feats trong nen.py và hàm move_feats trong src/strat_ml.py>

Thêm 3-5 đặc trưng có khả năng giúp robot này nhất. Trả lời ngắn: chỉ đưa lại hàm đã sửa, đầy đủ.
```

Các hàm nằm ở đâu (mở file rồi Ctrl + F, gõ đúng chữ trong cột 2):

| hàm | gõ vào ô tìm | file |
|---|---|---|
| `true_legs` | `def true_legs` | `nen.py` |
| `norm_for_parser` | `def norm_for_parser` | `nen.py` |
| `scene_feats` | `def scene_feats` | `nen.py` |
| `move_feats` | `def move_feats` | `src\strat_ml.py` |

### C3. Dán bản sửa của AI vào code

1. **Sao lưu trước:** trong File Explorer, bấm vào file sắp sửa → Ctrl + C → Ctrl + V. Sẽ có thêm file `... - Copy.py`. Đó là bản gốc để quay lại.
2. Trong VS Code, bôi đen **đúng hàm cũ** (từ dòng `def ...` tới hết hàm).
3. Dán hàm mới của AI đè lên (Ctrl + V).
4. Kiểm tra dòng `def ...` nằm **sát lề trái giống như trước** (trong `nen.py` các hàm ở Phần C2 đều sát lề; riêng hàm nằm trong `class` thì thụt vào 4 dấu cách). Thụt lề sai là lỗi.
5. Lưu: **Ctrl + S**.

### C4. Chạy lại sau khi sửa

- **Sửa để hết lỗi (mẫu A):** bấm đúp `CHAY_BAI1.bat` như cũ.
- **Sửa cách đọc dữ liệu hoặc đặc trưng (mẫu B, C):** bấm đúp **`HOC_LAI.bat`**. Nó xóa phần đã học, giữ phần đã đọc ảnh, rồi học lại và ra file nộp mới.

### C5. Sửa xong mà tệ hơn hoặc lỗi nặng hơn

1. Xóa file vừa sửa.
2. Đổi tên file `... - Copy.py` về tên cũ.
3. Chạy lại như C4.
4. File nộp các lần trước vẫn còn trong `runs\chungket\`: `predictions_ban_nen.json` (bản nền) và các file có giờ như `predictions_1432.json`. Nộp lại file có điểm cao nhất đã ghi.

---

## PHẦN D. Đề đổi gì thì kết quả ra sao

Các dòng `KHÁC` ở đầu cửa sổ đen cho biết bài mới có gì lạ. Thấy `KHÁC` **vẫn cứ để script chạy**.

| đề đổi | kết quả | cần làm gì |
|---|---|---|
| dữ liệu mới, thói quen robot mới | có cả bản nền và bản nâng cao | không |
| số cảnh, số dòng test khác | có cả hai bản | không |
| thêm / bớt robot, robot có chân khác | chỉ có bản nền | không |
| câu có thêm thông tin có/không (vd hàng nặng) | chạy được, bản nền tự học thông tin đó | không |
| nhiều điểm ghé, kiểu chỉ địa điểm lạ | chạy được, điểm thấp hơn | nếu còn token: mẫu B |
| loại địa điểm mới, kiểu vẽ mới, loại đường mới | có thể lỗi hoặc điểm rất thấp | mẫu A nếu lỗi; không thì nộp bản có được |
| đáp án không còn là 0–3 | không làm được | hỏi AI một lượt theo mẫu B; không được thì bỏ |

---

## PHẦN E. Chia thời gian (5 giờ, 2 bài)

| phút | việc |
|---|---|
| 0–5 | một bạn làm Bước 1, 2 của Phần A. Các bạn còn lại đọc đề bài 2 |
| 5–40 | script tự chạy; thỉnh thoảng liếc xem có `!!! BƯỚC LỖI` không |
| ~40 | `FILE NỘP (bản nền)` → nộp, ghi điểm |
| ~65–90 | `FILE NỘP (bản NÂNG CAO)` → nộp lại, ghi điểm |
| còn lại | làm bài 2 |

Bài 1 chiếm máy khoảng một tiếng rưỡi. Nếu đội có máy thứ hai thì làm bài 2 trên máy đó.

---

## PHẦN F. Tập trước ở nhà (10 phút)

Bấm đúp **`TAP_DUOT.bat`** trong `KIT_BAI1`. Nó chạy y như lúc thi trên bộ dữ liệu nhỏ `du_lieu_tap_duot` (phải nằm cạnh `KIT_BAI1`).
Thấy `FILE NỘP (bản nền)`, rồi `FILE NỘP (bản NÂNG CAO)`, rồi `XONG TẤT CẢ` là đúng.
Tập xong, xóa thư mục `KIT_BAI1\runs\tap1`.

---

## PHẦN G. Ghi chú kỹ thuật (không cần đọc khi thi)

- Các file `.bat` gọi Python tại `C:\Users\Thanh Hue\.venv\Scripts\python.exe`. Đây là bản duy nhất trên máy có đủ thư viện; bản Python của miniconda thiếu `onnxruntime`.
- Nếu muốn gõ lệnh tay trong terminal của VS Code (đã bật sẵn `.venv`):
  `python kiem_tra_de.py "<bài mới>"` rồi `python tu_dong.py "<bài mới>" chungket`.
- `tu_dong.py` gọi: `kiem_tra_de.py` (cấu trúc dữ liệu) → `soi_du_lieu.py` (dữ liệu hỏng + so sánh đặc trưng train / validation / test, không huấn luyện gì trên test) → `src/run_cv.py` (đọc ảnh bằng mô hình có sẵn) → `nen.py fit` (bộ đọc câu + mỗi robot một mô hình HistGradientBoosting) → `nen.py test` → `nang_cao.py` (mỗi robot hai mô hình siêu tuyến tính của gói COURIER2 và trung bình của chúng, luật tham lam cho robot 9; chọn theo validation; chỉ chạy khi có đúng 10 robot 0–9) → `nen.py test`.
- Code gốc vòng private của MinnToet nằm nguyên trong `src/` và `train/`; hướng dẫn ở `README_COURIER2.md`. Phần luật chính xác làm tay (0,93) không tự động hóa được.
- `mo_hinh_vong_private\` chỉ để tham khảo; luật robot đổi thì chúng sai.
