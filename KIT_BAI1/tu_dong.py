"""MỘT LỆNH cho ngày thi: từ thư mục dữ liệu mới -> file nộp predictions.json + bảng chẩn đoán.

Cách chạy:
  python tu_dong.py <thư mục dữ liệu mới (chứa train/ validation/ test/)> [tên lượt chạy=moi]
Tùy chọn (biến môi trường): PROCS = số tiến trình đọc ảnh (mặc định = số nhân CPU - 3, chừa nhân cho bước học; máy nóng thì đặt PROCS=4).

Việc script làm, theo thứ tự (bước nào đã xong thì lần chạy sau bỏ qua; muốn làm lại thì xóa thư mục runs/<tên>):
  0. soi dữ liệu mới khác vòng private ở đâu (kiem_tra_de.py)                     ~10 giây
  1. chạy NỀN: đọc ảnh validation rồi test bằng mô hình CV có sẵn (run_cv.py)     validation ~4 phút, test ~16 phút (8 nhân)
  2. học lại bộ đọc câu + mô hình chiến thuật trên dữ liệu mới (nen.py fit)       ~2 phút mỗi chế độ
  3. chẩn đoán trên validation: khâu nào mất điểm (nen.py chandoan)               ~2 phút
  4. ghi file nộp BẢN NỀN: runs/<tên>/predictions.json  -> NỘP NGAY                ~3 phút sau khi đọc xong ảnh test
  5. nâng cao: học mô hình mạnh hơn cho từng robot (nang_cao.py)                  ~20 phút
  6-7. chẩn đoán lại và ghi đè file nộp bằng BẢN NÂNG CAO -> NỘP LẠI
Đặt NANG_CAO=0 nếu chỉ muốn bản nền.
Mọi kết quả nằm trong runs/<tên>/ ; log từng bước trong runs/<tên>/log/.
"""
import json
import os
import shutil
import subprocess
import sys
import time
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
    p = _P(str(p).strip().strip('"')).resolve()
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
if len(sys.argv) < 2:
    sys.exit(__doc__)
DATA = tim_du_lieu(sys.argv[1])
NAME = sys.argv[2] if len(sys.argv) > 2 else "moi"
WORK = HERE / "runs" / NAME
LOG = WORK / "log"
LOG.mkdir(parents=True, exist_ok=True)
for sp in ("train", "validation", "test"):
    if not (DATA / sp / "observations.json").exists():
        sys.exit(f"Không thấy {DATA / sp / 'observations.json'}: kiểm tra lại đường dẫn dữ liệu.")
if not (HERE / "outputs" / "e5_small" / "onnx" / "model.onnx").exists():
    sys.exit("Thiếu outputs/e5_small/onnx/model.onnx (470 MB). Có mạng thì chạy: python src/embed.py download")

ENV = dict(os.environ, COURIER_DATA=str(DATA), COURIER_WORK=str(WORK), PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
           PYTHONWARNINGS="ignore", PROCS=os.environ.get("PROCS", str(max(1, (os.cpu_count() or 2) - 3))))
T0 = time.time()


def clock():
    return f"[{(time.time() - T0) / 60:5.1f} phút]"


def step(name, args, done=None, show=12, must=True):
    """Chạy một bước; in vài dòng cuối của log. done: file kết quả, có rồi thì bỏ qua."""
    if done is not None and Path(done).exists():
        print(f"{clock()} {name}: đã có, bỏ qua", flush=True)
        return True
    print(f"{clock()} {name} ...", flush=True)
    log = LOG / (name.split(":")[0].replace(" ", "_") + ".log")
    with open(log, "w", encoding="utf-8") as f:
        code = subprocess.call([sys.executable] + args, cwd=HERE, env=ENV, stdout=f, stderr=subprocess.STDOUT)
    lines = [l for l in log.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip() and "Warning" not in l and "warn" not in l]
    for l in lines[-show:]:
        print("      " + l[:400])
    if code != 0:
        print(f"{clock()} !!! BƯỚC LỖI: {name}. Log đầy đủ: {log}", flush=True)
        if must:
            print("      Mở log, đọc dòng CUỐI, tra bảng lỗi ở SO_TAY_NGAY_THI.md mục 5. Sửa xong chạy lại đúng lệnh cũ.")
            sys.exit(1)
    return code == 0


print(f"Dữ liệu: {DATA}\nKết quả: {WORK}\nSố tiến trình đọc ảnh: {ENV['PROCS']}\n", flush=True)

# 0. soi đề
step("0 soi dữ liệu mới", ["kiem_tra_de.py", str(DATA)], show=40, must=False)

step("0b soi dữ liệu đầu vào (vấn đề + test có khác validation không)", ["soi_du_lieu.py", str(DATA)], show=50, must=False)

# 1. đọc ảnh chạy nền (validation trước, test sau)
cv_log = open(LOG / "1_doc_anh.log", "w", encoding="utf-8")
SPLITS = ("validation",) if os.environ.get("CHI_DO") == "1" else ("validation", "test")      # CHI_DO=1: chỉ đo trên validation, không làm file nộp
cv_code = ("import subprocess,sys;"
           f"[subprocess.call([sys.executable,'src/run_cv.py',s,'v1final']) for s in {SPLITS!r}]")
need_cv = not all((WORK / "cache" / f"world_{s}_v1final.pkl").exists() for s in SPLITS)
cv_proc = subprocess.Popen([sys.executable, "-c", cv_code], cwd=HERE, env=ENV, stdout=cv_log, stderr=subprocess.STDOUT) if need_cv else None
print(f"{clock()} 1 đọc ảnh validation + test: {'đang chạy nền' if need_cv else 'đã có'}", flush=True)


def wait_cv(split):
    path = WORK / "cache" / f"world_{split}_v1final.pkl"
    said = False
    while not path.exists():
        if cv_proc is None or cv_proc.poll() is not None:
            if path.exists():
                break
            print(f"{clock()} !!! Đọc ảnh {split} không ra kết quả. Xem {LOG / '1_doc_anh.log'}")
            sys.exit(1)
        if not said:
            print(f"{clock()} chờ đọc ảnh {split} xong ...", flush=True)
            said = True
        time.sleep(10)
    print(f"{clock()} đọc ảnh {split}: xong", flush=True)


# 2. học lại trên dữ liệu mới
step("2a kho câu gấp - dễ vỡ", ["train/sent_inventory.py"], done=WORK / "outputs" / "sent_inventory.json", show=2)
step("2b học (chỉ train, để đo)", ["nen.py", "fit", "trainonly"], done=WORK / "outputs" / "nen_trainonly.pkl", show=5)
step("2c học (train + validation, để nộp)", ["nen.py", "fit", "final"], done=WORK / "outputs" / "nen_final.pkl", show=5)

# 3. chẩn đoán
wait_cv("validation")
step("3 chẩn đoán validation", ["nen.py", "chandoan"], show=11, must=False)

if os.environ.get("CHI_DO") == "1":
    print(f"{clock()} CHI_DO=1: dừng sau chẩn đoán, không ghi file nộp.")
    sys.exit(0)

def self_score():
    """Điểm tự chấm trên validation của lần chẩn đoán gần nhất (dòng (4) trong bảng CHẨN ĐOÁN); None nếu chưa có."""
    try:
        return float(json.loads((WORK / "outputs" / "chan_doan.json").read_text(encoding="utf-8"))["diem"]["cn"])
    except Exception:
        return None


def fmt(x):
    return "chưa đo được" if x is None else f"{x:.4f}"


S_NEN = self_score()


# 4. file nộp
def write_submission(label):
    """Chạy dự đoán trên test, kiểm tra định dạng, chép ra runs/<tên>/predictions.json (+ một bản có giờ)."""
    if not step(f"{label} ghi file nộp", ["nen.py", "test"], show=2, must=label.startswith("4")):
        return False
    src = WORK / "outputs" / "predictions_nen.json"
    n_rows = len(json.loads((DATA / "test" / "observations.json").read_text(encoding="utf-8")))
    preds = json.loads(src.read_text(encoding="utf-8"))
    ok = len(preds) == n_rows and all(isinstance(x, int) and 0 <= x <= 3 for x in preds)
    if not ok:
        print(f"{clock()} !!! SAI ĐỊNH DẠNG ({len(preds)} dòng / test có {n_rows} dòng): đừng nộp file này.")
        return False
    stamp = f"predictions_{time.strftime('%H%M')}.json"
    shutil.copy(src, WORK / "predictions.json")
    shutil.copy(src, WORK / stamp)
    with open(WORK / "nhat_ky_diem.txt", "a", encoding="utf-8") as f:      # sổ ghi: mỗi file nộp đã tạo + điểm tự chấm của nó
        f.write(f"{time.strftime('%H:%M')}  {'bản nền' if label.startswith('4') else 'bản nâng cao'}  điểm tự chấm {fmt(self_score())}  file {stamp}\n")
    print(f"\n{clock()} >>> FILE NỘP ({'bản nền' if label.startswith('4') else 'bản NÂNG CAO'}): {WORK / 'predictions.json'}   ({len(preds)} dòng, HỢP LỆ)\n", flush=True)
    return True


wait_cv("test")
write_submission("4")
print(f"      Điểm tự chấm của bản nền: {fmt(S_NEN)}")
print("      SỐ LẦN NỘP CÓ HẠN: còn nhiều lượt thì nộp file này ngay để có điểm; chỉ còn ít lượt (3 trở xuống) thì CHỜ bản nâng cao rồi nộp một lần.")
print("      Script đang tự học tiếp mô hình mạnh hơn (25-45 phút); đừng tắt cửa sổ này.", flush=True)

# 5-7. nâng cao: mô hình chiến thuật mạnh hơn cho từng robot, rồi ghi lại file nộp
if os.environ.get("NANG_CAO", "1") == "1":
    shutil.copy(WORK / "predictions.json", WORK / "predictions_ban_nen.json")
    step("5 nâng cao (học mô hình mạnh hơn cho từng robot)", ["nang_cao.py"], done=WORK / "outputs" / "chon.json", show=18, must=False)
    if (WORK / "outputs" / "chon.json").exists():
        step("6 chẩn đoán lại validation", ["nen.py", "chandoan"], show=11, must=False)
        if write_submission("7"):
            s_nc = self_score()
            shutil.copy(WORK / "predictions.json", WORK / "predictions_nang_cao.json")
            if S_NEN is not None and s_nc is not None and s_nc < S_NEN + 0.003:
                shutil.copy(WORK / "predictions_ban_nen.json", WORK / "predictions.json")
                print(f"      Bản nâng cao KHÔNG hơn bản nền trên validation ({fmt(S_NEN)} -> {fmt(s_nc)}).")
                print("      => predictions.json đã được TRẢ VỀ BẢN NỀN. Nếu đã nộp bản nền rồi thì KHÔNG cần nộp lại (đỡ tốn lượt).")
            else:
                print(f"      Bản nâng cao TỐT HƠN trên validation: {fmt(S_NEN)} -> {fmt(s_nc)}.  => NỘP file predictions.json này.")
                print("      (bản nền vẫn còn ở predictions_ban_nen.json)")
    else:
        print(f"{clock()} Không có bản nâng cao: giữ file nộp bản nền.")
print(f"{clock()} XONG TẤT CẢ. Các file nộp đã tạo và điểm tự chấm của từng file: {WORK / 'nhat_ky_diem.txt'}")
