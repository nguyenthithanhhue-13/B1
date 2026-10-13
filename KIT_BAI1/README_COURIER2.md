# Campus Courier 2026 - Private test (COURIER2) - mã nguồn huấn luyện và dự đoán

Gói này tái lập đúng bài nộp được chọn: `outputs/private_result/predictions.json` (bản p27, bảng xếp hạng công khai
**0,8661**).

Hệ thống gồm 3 phần chạy nối tiếp cho mỗi cảnh:

1. **CV**: đọc ảnh bản đồ -> đồ thị lưới (giao lộ, đoạn đường và trạng thái, bậc thang, một chiều), địa điểm, robot và
   hướng mũi, thời tiết. Dùng lại mô hình CV của bài COURIER.
2. **Đọc yêu cầu** (tiếng Việt): nơi giao (kể cả nơi chỉ mô tả qua bản đồ, đính chính / hủy đơn), điểm ghé, tham chiếu
   không gian, cờ gấp / dễ vỡ (có khôi phục câu gõ sai).
3. **Chiến thuật**: bước đi đầu tiên của 10 robot, mỗi robot một mô hình riêng (luật chính xác, mô hình học, bỏ phiếu).

## 1. Môi trường và dữ liệu

- Python 3.10, chạy CPU (Windows 11, 8 nhân, 24 GB RAM). Dự đoán không cần GPU.
- `pip install -r requirements.txt`. Các script huấn luyện mô hình chiến thuật và CNN cần thêm `torch` (có GPU thì nhanh hơn).
- Đặt `PYTHONUTF8=1`.
- Biến môi trường `COURIER_DATA` trỏ tới thư mục dữ liệu của bài COURIER2 (thư mục chứa `train/`, `validation/`, `test/`).
- Mọi lệnh dưới đây chạy từ thư mục gốc của gói (thư mục chứa README này).

## 2. Chuẩn bị bộ nhúng câu e5 (một lần)

```bash
python src/embed.py download
```

Lệnh này tải `onnx/model.onnx` của mô hình công khai `intfloat/multilingual-e5-small` (~470 MB, ~118 triệu tham số) ở phiên
bản cố định `614241f622f53c4eeff9890bdc4f31cfecc418b3` vào `outputs/e5_small/onnx/` và kiểm tra mã băm SHA-256. File này vượt
giới hạn 90 MB của zip nên không đóng gói; `tokenizer.json` đã có sẵn. Mô hình chỉ được dùng nguyên trạng, không huấn luyện lại.

## 3. Chạy lại bài nộp

```bash
export COURIER_DATA=/duong/dan/du_lieu_courier2
export PYTHONUTF8=1 PROCS=6
python src/run_cv.py test v1final       # CV trên 1200 ảnh test (~20-25 phút với 6 tiến trình)
python src/pipeline_p2.py test          # đọc yêu cầu + chiến thuật -> outputs/private_result/predictions.json (~5-10 phút)
```

`PROCS` là số tiến trình chạy CV song song (mặc định 11). Nếu bỏ qua lệnh `run_cv.py`, `pipeline_p2.py` tự chạy CV.

Đo trên validation bằng các mô hình chỉ học train (`*_trainonly`):

```bash
python src/run_cv.py validation v1final
python src/pipeline_p2.py eval
```

## 4. Vai trò từng file

Mỗi file có một khối mô tả ở đầu file ghi tác dụng, cách chạy, file đọc vào và ghi ra.

### 4.1. `src/` - dự đoán

| File | Tác dụng | Cách chạy |
|---|---|---|
| `pipeline_p2.py` | Ghép CV + đọc yêu cầu + chiến thuật; sinh file nộp, đo validation, học bộ đọc yêu cầu + HGB | `python src/pipeline_p2.py test` / `eval` / `fit <trainonly\|final>` |
| `run_cv.py` | Chạy CV cho cả một split, song song nhiều tiến trình; lưu đồ thị vào `cache/` | `python src/run_cv.py <split> v1final` |
| `cv_infer.py`, `cvfeat.py`, `mlp.py` | Suy luận CV cho một ảnh (MLP numpy + CNN ONNX), hàm xử lý ảnh, mạng MLP numpy | thư viện |
| `embed.py` | Bộ nhúng câu multilingual-e5-small (ONNX); tải mô hình | `python src/embed.py download` |
| `nlp.py`, `nlp_tagger.py`, `nlp_knowledge.py` | Chuẩn hóa câu, so khớp mờ; gán nhãn cụm địa điểm; kiến thức tiếng Việt viết tay | thư viện / dữ liệu |
| `nlp2.py` | Bộ đọc yêu cầu MissionParser2 (dùng lại từ bài COURIER, học lại trên dữ liệu mới) | thư viện |
| `nlp3.py` | Bộ đọc yêu cầu vòng private (MissionParser3): khôi phục câu cờ gõ sai, xử lý nơi giao mô tả qua bản đồ, gọi MissionParser2 | thư viện |
| `mapref.py` | Nhận diện "địa điểm nằm cao nhất", "nơi cạnh thư viện nhất"... và tìm vị trí trên bản đồ do CV đọc | thư viện |
| `strategy.py` | Chọn vị trí theo tham chiếu không gian (bắc / nam / gần / xa mốc...) | thư viện |
| `strat_ml.py` | Ghép mô hình chiến thuật cho từng robot theo `outputs/strategy_hybrid_final.json` và `outputs/rules_final.json` | thư viện |
| `rules.py` | Luật chính xác theo nhóm điều kiện (chi phí tuyến tính trên mặt Pareto các đường + thứ tự phá hòa) | thư viện |
| `hyper.py` | Mô hình siêu tuyến tính: trọng số chi phí = softplus(MLP(điều kiện cảnh)) trên mặt Pareto các đường | thư viện |
| `vin.py` | Bộ tìm đường có hàm chi phí học được (MLP chi phí từng đoạn + lặp giá trị) | thư viện |
| `common.py` | Hằng số, đường dẫn, đọc dữ liệu, chấm macro accuracy | thư viện |
| `cv_cnn.py`, `cv_edge.py`, `cv_swatch.py`, `cv_weather.py` | Lớp CNN mà `cv_infer.py` dùng; đồng thời là script học các CNN đó | xem 4.3 |

### 4.2. `train/` - huấn luyện mô hình chiến thuật và bộ đọc yêu cầu

| File | Tác dụng | Cách chạy |
|---|---|---|
| `lib.py` | Thư viện chung: đọc train / validation kèm thông tin đúng trong `scenes.json`, Dijkstra | thư viện |
| `pareto.py` | Liệt kê mặt Pareto các đường (số đoạn <= ngắn nhất + 5) cho mỗi cảnh, theo vector đặc trưng đường | `python train/pareto.py <train\|validation>` |
| `rules_pareto.py` | Đổi mặt Pareto sang ma trận đặc trưng cho bộ tìm luật | `python train/rules_pareto.py` |
| `rules_search.py` | Tìm luật chính xác cho một robot theo nhóm điều kiện (tìm kiếm chùm trên hạng tử chi phí + chọn thứ tự phá hòa) | `ENUM=5p FITSPLIT=trainval python train/rules_search.py <robot> <KEY> <CELL> 6 6` |
| `rules_config.py` | Gộp các luật đã tìm thành `outputs/rules_<chế độ>.json` | xem mục 7 |
| `hyper_train.py` | Học mô hình siêu tuyến tính cho một robot (torch) | `python train/hyper_train.py <robot> 1500 64 3 final` |
| `hyper_export.py` | Xuất checkpoint siêu tuyến tính -> `outputs/hyper*_<chế độ>.npz` | xem mục 7 |
| `vin_data.py` | Tạo dữ liệu cho bộ tìm đường có hàm chi phí học được | `python train/vin_data.py <train\|validation>` |
| `vin_train.py` | Học bộ tìm đường có hàm chi phí học được cho một robot (torch) | `python train/vin_train.py <robot> 15 64 3 final` |
| `vin_export.py` | Xuất checkpoint -> `outputs/vin_<chế độ>.npz` | xem mục 7 |
| `greedy_feats.py` | Đặc trưng một bước đi cho robot 9 (thư viện của `greedy_rules.py`) | thư viện |
| `greedy_rules.py` | Dò luật tham lam cho robot 9 theo nhóm đêm / gấp / dễ vỡ | `python train/greedy_rules.py nuf` |
| `ml4_train.py` | Học bộ phân loại HistGradientBoosting dự phòng cho robot 5 và 7 | `python train/ml4_train.py final 5 7` |
| `sent_inventory.py` | Tạo kho câu cờ chuẩn (câu gấp / dễ vỡ gặp >= 5 lần trong train + validation) cho bước khôi phục câu gõ sai | `python train/sent_inventory.py` |

### 4.3. `src/` - huấn luyện CV (giống bài COURIER)

| File | Tác dụng | Cách chạy |
|---|---|---|
| `cv_data.py` | Cắt mảnh ảnh huấn luyện từ ảnh + `scenes.json` | `python src/cv_data.py <split> [all\|strong]` |
| `cv_train.py` | Học 6 MLP của CV (`det`, `swatch`, `edge`, `align`, `node`, `weather`) | `python src/cv_train.py <dev\|final>` |
| `cv_hardneg.py` | Khai thác âm bản khó cho bộ dò `det` | `python src/cv_hardneg.py train dev [--append]` |
| `cv_cnn.py`, `cv_edge.py`, `cv_swatch.py`, `cv_weather.py` | Học 4 CNN (giao lộ, đoạn đường, chú giải, thời tiết) và xuất ONNX | `python src/cv_cnn.py final 8` ... |

## 5. Trọng số trong `outputs/` được lấy bằng cách nào

Tất cả trọng số chỉ học từ `train` và `validation` (ảnh, câu yêu cầu, `scenes.json`, nhãn). Test chỉ dùng để chạy dự đoán.
Bản `*_final` học trên train + validation và được dùng cho bài nộp; bản `*_trainonly` chỉ học train, dùng cho lệnh `eval`.

| File | Là gì | Lấy bằng cách nào |
|---|---|---|
| `models_v1final/*` | Trọng số CV: 6 MLP numpy (`.npz`) + 4 CNN (`.onnx`, kèm checkpoint `.pt` và số tham số `_params.txt`) | Giống từng byte `outputs/models_final/` của bài COURIER (học trên train + validation của bài COURIER, quy trình ở mục 7, bước 0). Dùng lại vì bản đồ vòng private cùng kiểu vẽ: trên validation mới, 98,3% bản đồ được đọc đúng hoàn toàn |
| `nlp3_final.pkl`, `nlp3_trainonly.pkl` | Bộ đọc yêu cầu (từ điển tên gọi, TF-IDF + hồi quy logistic, ngưỡng) | `python src/pipeline_p2.py fit final` (train + validation) / `fit trainonly` (train) |
| `strategy_ml_final.pkl`, `strategy_ml_trainonly.pkl` | HistGradientBoosting từng robot (tầng dự phòng cuối) | cùng lệnh `fit` ở trên |
| `rules_final.json`, `rules_trainonly.json` | Luật chính xác theo nhóm điều kiện của robot 0, 1, 2, 3, 4, 7, 8 và 2 bộ luật phụ của robot 6 (dùng để bỏ phiếu) | `train/pareto.py` -> `train/rules_pareto.py` -> `train/rules_search.py` -> `train/rules_config.py` (mục 7, bước 3) |
| `hyper_final.npz`, `hypergoal_final.npz`, `hypersel_final.npz` (+ `_trainonly`) | Mô hình siêu tuyến tính của robot 1-8, 3 bộ điều kiện: đầy đủ, theo loại đích, chọn riêng từng robot (ghi trong khóa `meta`) | `train/hyper_train.py` (torch, 3 hoặc 5 hạt giống) -> `train/hyper_export.py` (mục 7, bước 4) |
| `vin_final.npz`, `vin_trainonly.npz` | Bộ tìm đường có hàm chi phí học được của robot 1, 2, 4, 6, 8 (mạng nhỏ ẩn 64 và mạng lớn ẩn 128, mỗi loại 3 hạt giống) | `train/vin_data.py` -> `train/vin_train.py` -> `train/vin_export.py` (mục 7, bước 4) |
| `strategy_ml4_final.pkl`, `strategy_ml4_trainonly.pkl` | HistGradientBoosting dự phòng của robot 5 và 7 | `python train/ml4_train.py final 5 7` (đã kiểm tra: học lại cho ra mô hình giống hệt file trong gói) |
| `strategy_hybrid_final.json`, `strategy_hybrid.json` | Cấu hình chiến thuật từng robot: dùng mô hình nào, trọng số trộn siêu tuyến tính / tìm đường (`a` = 0,67), tham số luật tham lam robot 9 và các tầng dự phòng | xem ghi chú bên dưới |
| `sent_inventory.json` | Kho câu cờ chuẩn cho bước khôi phục câu gõ sai | `python train/sent_inventory.py` (đã kiểm tra: tạo lại giống hệt file trong gói) |
| `e5_small/tokenizer.json` | Tokenizer của multilingual-e5-small | đi kèm mô hình công khai, không huấn luyện; `model.onnx` tải bằng `python src/embed.py download` |
| `private_result/predictions.json` | Bài nộp p27 | kết quả của `python src/pipeline_p2.py test` |

Ghi chú về `strategy_hybrid*.json`:

- Mô hình chính của từng robot (khóa `method` ngoài cùng) được chọn theo độ chính xác trên validation của các bản
  `*_trainonly`: robot 0-4, 7, 8 dùng luật chính xác ở các nhóm điều kiện mà luật khớp dữ liệu học, còn lại dùng `hv` (trung
  bình log-xác suất của các mô hình siêu tuyến tính, trộn với bộ tìm đường theo trọng số `a`); robot 6 bỏ phiếu giữa 2 bộ
  luật phụ và `hv`; robot 9 dùng luật tham lam `greedy2`.
- Tham số `greedy2` của robot 9 lấy từ `python train/greedy_rules.py nuf` (học trên train, 4 lần khởi đầu ngẫu nhiên, hạt
  giống 0; ghi ra `cache/p2_greedy_exactfix_nuf.json`), chép vào mục `"9"` -> `"params"`. Đã kiểm tra: chạy lại cho đúng
  tham số trong gói ở cả 8 nhóm.
- Các khóa `fallback` là tầng dự phòng, chỉ chạy khi các mô hình phía trên không đưa ra được hướng đi (ví dụ không tìm được
  đường tới đích trên bản đồ CV đọc). Tham số của chúng (`cost`, `qcfg`, `lex`, `greedy`: phạt đông người, mái che, đi qua
  địa điểm, rẽ trái / phải / quay đầu) được dò theo từng tọa độ trên lưới giá trị nhỏ để khớp nhãn train (bản `_final`: train +
  validation) trong giai đoạn nghiên cứu, rồi ghi cố định vào file JSON. Đã đếm trên test: các tầng `cost`, `lex`, `greedy`
  không quyết định dòng nào; `qcfg` chỉ là đặc trưng của ml4, được dùng ở 30/12000 dòng (robot 5 và 7 trong 15 cảnh mà mô
  hình chính không cho kết quả); HGB `strategy_ml_final.pkl` quyết định 30/12000 dòng (robot 0 và 3 trong 15 cảnh đó).

## 6. Thư mục `cache/` dùng để làm gì

`cache/` chứa **kết quả trung gian** mà các script tự ghi ra để lần chạy sau không phải tính lại những bước tốn thời gian
(CV trên 1200 ảnh mất ~20 phút; liệt kê mặt Pareto, tạo dữ liệu học mất hàng chục phút). Trong gói, `cache/` để trống: đó
là trạng thái bình thường, các file sẽ tự xuất hiện khi chạy lệnh ở mục 3 và 7.

| File trong `cache/` | Script ghi ra | Nội dung | Được dùng lại bởi |
|---|---|---|---|
| `det_<split>_v1final.pkl` | `run_cv.py` | kết quả bộ dò giao lộ của từng ảnh | `run_cv.py` (lần sau bỏ qua bước dò) |
| `world_<split>_v1final.pkl` | `run_cv.py` | đồ thị bản đồ đọc từ mỗi ảnh (đầu ra của CV) | `pipeline_p2.py` |
| `e5_cache.pkl` | `embed.py` | vector e5 của các câu đã nhúng | `embed.py` |
| `p2_stratfeats_<split>.pkl` | `pipeline_p2.py fit` | đặc trưng bước đi để học HGB | `pipeline_p2.py fit` |
| `p2_paretotu_<split>_<0\|1>.pkl` | `train/pareto.py` | mặt Pareto các đường (0: robot thường, 1: robot 4 đi được bậc thang) | `rules_pareto.py`, `hyper_train.py` |
| `p3_enum5p_<split>_<0\|1>.pkl` | `train/rules_pareto.py` | ma trận đặc trưng đường cho bộ tìm luật | `rules_search.py` |
| `p3_search_r<robot>_<KEY>_<CELL>[_tv].json` | `train/rules_search.py` | luật tìm được của một robot | `rules_config.py` |
| `p2_hyper<tag>_r<robot>[_final].pt` | `train/hyper_train.py` | checkpoint mô hình siêu tuyến tính | `hyper_export.py` |
| `p2_vin_<split>.npz` | `train/vin_data.py` | dữ liệu học bộ tìm đường | `vin_train.py` |
| `p2_vin2<tag>_r<robot>_s<i>[_final].pt` | `train/vin_train.py` | checkpoint bộ tìm đường | `vin_export.py` |
| `p2_greedy_exactfix_<KEY>.json` | `train/greedy_rules.py` | tham số luật tham lam robot 9 | chép tay vào `strategy_hybrid*.json` |
| `p2_mlfeats_<split>.pkl` | `train/ml4_train.py` | đặc trưng bước đi cho bộ phân loại ml4 | `ml4_train.py` |
| `cvdata_*.npz`, `cvdet_*.npz`, `cvcnn_*.npz`, `cvstrong_*.npz`, `cvhard_*.npz` | `cv_data.py`, `cv_hardneg.py` | mảnh ảnh huấn luyện CV | `cv_train.py`, các script CNN |

Lưu ý: script thấy file cache đã có thì dùng lại, không tính lại. Khi đổi trọng số hoặc sửa mã, hãy xóa file cache tương
ứng (hoặc xóa cả thư mục `cache/`, lúc nào cũng an toàn) để kết quả được tính mới.

## 7. Huấn luyện lại

Chạy từ thư mục gốc của gói, theo thứ tự. Lệnh dưới đây tạo bản `*_final` (học train + validation). Bản `*_trainonly` dùng
cùng lệnh nhưng bỏ chữ `final` ở cuối lệnh học, đổi `final` thành `trainonly` ở lệnh xuất, và bỏ `FITSPLIT=trainval`, `SUF=_tv`
ở bước 3.

**Bước 0 - CV** (tùy chọn; giống bài COURIER, cần dữ liệu bài COURIER; kết quả ở `outputs/models_final/`, đổi tên thành
`outputs/models_v1final/`):

```bash
python src/cv_data.py train ; python src/cv_data.py validation ; python src/cv_data.py train strong
python src/cv_train.py dev
python src/cv_hardneg.py train dev ; python src/cv_train.py dev det
python src/cv_hardneg.py train dev --append ; python src/cv_train.py dev det
python src/cv_train.py final
python src/cv_cnn.py final 8 ; python src/cv_edge.py final 6 ; python src/cv_swatch.py final 12 ; python src/cv_weather.py final 14
```

**Bước 1 - bộ đọc yêu cầu + HGB + kho câu cờ:**

```bash
python train/sent_inventory.py
python src/pipeline_p2.py fit final
```

**Bước 2 - mặt Pareto các đường** (dùng thông tin đúng trong `scenes.json` của train / validation):

```bash
python train/pareto.py train
python train/pareto.py validation
python train/rules_pareto.py
```

**Bước 3 - luật chính xác** -> `outputs/rules_final.json`:

```bash
ENUM=5p FITSPLIT=trainval python train/rules_search.py 0 w wu 6 6
ENUM=5p FITSPLIT=trainval python train/rules_search.py 2 cnw - 6 6
ENUM=5p FITSPLIT=trainval python train/rules_search.py 3 cwv - 6 6
ENUM=5p FITSPLIT=trainval python train/rules_search.py 4 wuf - 6 6
ENUM=5p FITSPLIT=trainval REFINE=0 python train/rules_search.py 1 nwu - 6 6
ENUM=5p FITSPLIT=trainval REFINE=0 python train/rules_search.py 7 cfw - 6 6
ENUM=5p FITSPLIT=trainval REFINE=0 python train/rules_search.py 8 nfv - 6 6
ENUM=5p FITSPLIT=trainval REFINE=0 python train/rules_search.py 6 nwf - 6 6
ENUM=5p FITSPLIT=trainval REFINE=0 python train/rules_search.py 6 nuf - 6 6
SUF=_tv THR=0.95 PARTIAL=7,8 python train/rules_config.py outputs/rules_final.json \
    0:w:wu 1:nwu:- 2:cnw:- 3:cwv:- 4:wuf:- 7:cfw:- 8:nfv:- 6@nwf:nwf:- 6@nuf:nuf:-
```

`KEY` là nhóm điều kiện mà luật được tìm riêng (mỗi chữ một điều kiện: `n` đêm, `w` mưa, `u` gấp, `f` dễ vỡ, `v` có điểm ghé,
`c` cụm loại đích, `g` loại đích), `CELL` là nhóm có thứ tự phá hòa riêng (`-` = một thứ tự chung). Với robot 7, 8, nhóm nào
luật khớp dữ liệu học dưới 95% thì được đánh dấu bỏ và dùng mô hình học thay luật.

**Bước 4 - mô hình siêu tuyến tính và bộ tìm đường** (torch):

```bash
for r in 1 2 3 4 5 6 7 8; do
  python train/hyper_train.py $r 1500 64 3 final
  COND=goal HTAG=goal python train/hyper_train.py $r 1500 64 3 final
done
COND=rain+night+urg HTAG=sel python train/hyper_train.py 1 1500 64 5 final
COND=rain+night+goal HTAG=sel python train/hyper_train.py 2 1500 64 5 final
COND=rain+goal HTAG=sel python train/hyper_train.py 3 1500 64 5 final
COND=rain+urg+frag+goal HTAG=sel python train/hyper_train.py 4 1500 64 5 final
COND=night HTAG=sel python train/hyper_train.py 5 1500 64 5 final
COND=night+urg+frag+via+mapg+goal HTAG=sel python train/hyper_train.py 6 1500 64 5 final
COND=rain+night+urg+frag+goal HTAG=sel python train/hyper_train.py 7 1500 64 5 final
COND=night+urg+frag+via+goal HTAG=sel python train/hyper_train.py 8 1500 64 5 final
python train/hyper_export.py final 1:full:SRLB 2:full:SRLB 3:full:SRLB 4:full:SRLB 5:full:SRLB 6:full:SRLB 7:full:SRLB 8:full:SRLB
OUTTAG=goal python train/hyper_export.py final 1:goal:SRLB:goal 2:goal:SRLB:goal 3:goal:SRLB:goal 4:goal:SRLB:goal \
    5:goal:SRLB:goal 6:goal:SRLB:goal 7:goal:SRLB:goal 8:goal:RSLB:goal
OUTTAG=sel python train/hyper_export.py final 1:rain+night+urg:SRLB:sel 2:rain+night+goal:RLSB:sel 3:rain+goal:SRLB:sel \
    4:rain+urg+frag+goal:SRLB:sel 5:night:SRLB:sel 6:night+urg+frag+via+mapg+goal:SRLB:sel \
    7:rain+night+urg+frag+goal:SRLB:sel 8:night+urg+frag+via+goal:SRLB:sel

python train/vin_data.py train
python train/vin_data.py validation
for r in 1 2 4 6; do python train/vin_train.py $r 15 64 3 final; done
for r in 1 4 8; do VTAG=big python train/vin_train.py $r 25 128 3 final; done
python train/vin_export.py final 1:,big 2: 4:,big 6: 8:big
```

`COND` là các điều kiện cảnh đưa vào MLP sinh trọng số (`full` = tất cả, `goal` = loại đích); `SRLB`, `RSLB`, `RLSB` là thứ tự
phá hòa (thẳng / phải / trái / quay đầu) ghi vào khóa `meta`. Lệnh xuất đã được kiểm tra là xuất lại đúng từng số của các file
`.npz` trong gói từ checkpoint đã học. Mô hình torch phụ thuộc hạt giống ngẫu nhiên và phần cứng, nên học lại có thể lệch
nhẹ; bài nộp dùng đúng các file trong `outputs/`.

**Bước 5 - robot 9 và bộ phân loại ml4:**

```bash
python train/greedy_rules.py nuf          # tham số -> cache/p2_greedy_exactfix_nuf.json, chép vào "9" -> "params" của strategy_hybrid*.json
python train/ml4_train.py final 5 7
```

Hai lệnh này đã được chạy lại và cho đúng tham số R9 và mô hình ml4 đang có trong `outputs/`.

## 8. Kết quả kiểm tra

- Chạy lại từ cache trống (CV + đọc yêu cầu + chiến thuật) bằng mã trong gói: `predictions.json` trùng 12000/12000 dòng với
  bài nộp p27.
- Validation end-to-end (CV thật, bộ đọc yêu cầu + chiến thuật chỉ học train): 0,9347. Riêng kho câu cờ
  `sent_inventory.json` được tạo từ train + validation, nên con số này có thể hơi lạc quan.
- Tổng tham số khi dự đoán ~132 triệu (e5-small ~118 triệu + CV ~13,5 triệu + mô hình chiến thuật), dưới giới hạn 200
  triệu. Không gọi dịch vụ AI bên ngoài; test chỉ dùng để chạy dự đoán.
