"""NÂNG CAO (tu_dong.py tự gọi sau khi đã có file nộp đầu tiên): học mô hình chiến thuật mạnh hơn cho từng robot.

Việc làm:
  1. liệt kê các đường đi tốt của mỗi cảnh (train/pareto.py)                              ~2-3 phút
  2. mỗi robot: học 2 mô hình "siêu tuyến tính" (train/hyper_train.py; cần torch)          tổng ~20 phút khi SONG_SONG=2
  3. robot 9 ("chỉ nhìn một bước"): dò luật tham lam (train/greedy_rules.py)               vài giây
  4. chọn cho từng robot mô hình đúng nhất trên validation (nen.py chon) -> outputs/chon.json
Robot nào mô hình mới không hơn mô hình nền thì giữ mô hình nền. Bước nào lỗi thì bỏ qua bước đó, file nộp cũ vẫn còn.

Chỉ áp dụng khi dữ liệu có đúng 10 robot (0..9) như các vòng trước; khác thì script tự dừng và giữ bản nền.
Cách chạy tay (tu_dong.py tự đặt hai biến môi trường):  python nang_cao.py
Tùy chọn: SEEDS (mặc định 2), SONG_SONG = số robot học cùng lúc (mặc định 2).
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

for _st in (sys.stdout, sys.stderr):
    try:
        _st.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
HERE = Path(__file__).resolve().parent
if not (os.environ.get("COURIER_DATA") and os.environ.get("COURIER_WORK")):
    sys.exit("Thiếu COURIER_DATA / COURIER_WORK. Chạy qua tu_dong.py.")
DATA, WORK = Path(os.environ["COURIER_DATA"]), Path(os.environ["COURIER_WORK"])
(WORK / "cache").mkdir(parents=True, exist_ok=True)
(WORK / "outputs").mkdir(parents=True, exist_ok=True)
ENV = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONWARNINGS="ignore",
           PYTHONPATH=os.pathsep.join([str(HERE / "src"), str(HERE / "train"), os.environ.get("PYTHONPATH", "")]))
SEEDS = os.environ.get("SEEDS", "2")
PAR = int(os.environ.get("SONG_SONG", "2"))
T0 = time.time()


def say(msg):
    print(f"[{(time.time() - T0) / 60:5.1f} phút] {msg}", flush=True)


def run(script, *args, env=None):
    """Chạy một script trong train/ với thư mục làm việc = WORK (các script đó ghi vào cache/ và outputs/ tương đối)."""
    r = subprocess.run([sys.executable, str(HERE / script), *map(str, args)], cwd=WORK, env=dict(ENV, **(env or {})),
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.returncode, (r.stdout or "") + (r.stderr or "")


# ---- chỉ áp dụng cho dạng 10 robot quen thuộc ----
rows = json.loads((DATA / "train" / "observations.json").read_text(encoding="utf-8"))
ids = sorted({r["robot_id"] for r in rows}, key=str)
if ids != list(range(10)) or len(rows) % 10:
    say(f"Dữ liệu có robot {ids}: khác dạng 10 robot (0..9) nên KHÔNG chạy phần nâng cao. Giữ bản nền.")
    sys.exit(0)
try:
    import torch      # noqa: F401
except Exception:
    say("Máy không có torch nên không học được mô hình nâng cao. Giữ bản nền.")
    sys.exit(0)

# ---- 1. các đường đi tốt ----
for split in ("train", "validation"):
    if not (WORK / "cache" / f"p2_paretotu_{split}_1.pkl").exists():
        say(f"1. liệt kê đường đi ({split}) ...")
        code, out = run("train/pareto.py", split)
        if code != 0:
            say("!!! lỗi ở bước liệt kê đường đi: " + out.strip().splitlines()[-1][:300] + "\n      Giữ bản nền.")
            sys.exit(0)

# ---- 2. mô hình siêu tuyến tính cho từng robot: 2 bộ điều kiện (đầy đủ, theo loại đích) x 2 chế độ ----
PAT = re.compile(r"order=(\w+): train ([\d.]+) validation ([\d.]+)")
SETS = [("", "full"), ("goal", "goal")]      # (tag file, điều kiện COND)
memo_path = WORK / "cache" / "nang_cao_orders.json"
memo = json.loads(memo_path.read_text(encoding="utf-8")) if memo_path.exists() else {}


def learn(job):
    tag, cond, mode, r = job
    key = f"{tag}|{mode}|{r}"
    pt = WORK / "cache" / f"p2_hyper{tag}_r{r}{'_final' if mode == 'final' else ''}.pt"
    if key in memo and pt.exists():      # đã học ở lần chạy trước
        return job, tuple(memo[key]), ""
    code, out = run("train/hyper_train.py", r, 1500, 64, SEEDS, *(["final"] if mode == "final" else []), env={"COND": cond, "HTAG": tag})
    m = PAT.search(out)
    return job, (m.group(1), float(m.group(3))) if (code == 0 and m) else None, out


say(f"2. học mô hình siêu tuyến tính: 10 robot x 2 bộ điều kiện x 2 chế độ ({PAR} việc cùng lúc) ...")
jobs = [(tag, cond, mode, r) for tag, cond in SETS for mode in ("trainonly", "final") for r in range(10)]
res = {}
with ThreadPoolExecutor(PAR) as ex:
    for (tag, cond, mode, r), got, out in ex.map(learn, jobs):
        if got:
            res[(tag, mode, r)] = got
            memo[f"{tag}|{mode}|{r}"] = list(got)
            memo_path.write_text(json.dumps(memo), encoding="utf-8")
            if mode == "trainonly":
                say(f"   robot {r} ({cond}): validation {got[1]:.3f}")
        else:
            say(f"   !!! robot {r} ({cond}, {mode}) không học được: " + (out.strip().splitlines() or ["?"])[-1][:200])
for tag, cond in SETS:
    for mode in ("trainonly", "final"):
        specs = [f"{r}:{cond}:{res[(tag, mode, r)][0]}" + (f":{tag}" if tag else "") for r in range(10)
                 if (tag, mode, r) in res and (tag, "trainonly", r) in res and (tag, "final", r) in res]
        (WORK / "outputs" / f"hyper{tag}_{mode}.npz").unlink(missing_ok=True)
        if specs:
            code, out = run("train/hyper_export.py", mode, *specs, env={"OUTTAG": tag})
            if code != 0:
                say(f"!!! không xuất được mô hình ({cond}, {mode}): " + out.strip().splitlines()[-1][:300])

# ---- 3. luật tham lam cho robot 9 ----
say("3. dò luật tham lam cho robot 9 ...")
code, out = run("train/greedy_rules.py", "nuf")
gp = WORK / "cache" / "p2_greedy_exactfix_nuf.json"
if code == 0 and gp.exists():
    shutil.copy(gp, WORK / "outputs" / "greedy.json")
    say("   " + out.strip().splitlines()[-1][:200])
else:
    say("   !!! không dò được luật tham lam (bỏ qua): " + (out.strip().splitlines() or ["?"])[-1][:200])

# ---- 4. chọn mô hình tốt nhất cho từng robot ----
say("4. chọn mô hình tốt nhất cho từng robot ...")
code, out = run("nen.py", "chon")
print("\n".join("      " + l for l in out.strip().splitlines()[-14:]))
if code != 0:
    (WORK / "outputs" / "chon.json").unlink(missing_ok=True)
    say("!!! bước chọn lỗi: giữ bản nền.")
say("NÂNG CAO xong.")
