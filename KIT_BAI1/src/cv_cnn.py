"""CNN phân loại giao lộ (thường / robot + 4 hướng mũi / 10 loại địa điểm / không phải giao lộ).
Cách chạy (học):  python src/cv_cnn.py <dev|final> <số vòng>      (cần torch; đọc cache/cvcnn_*.npz)
Cách chạy (xuất ONNX):  python src/cv_cnn.py <chế độ> export      Ghi: outputs/models_<chế độ>/node_cnn.pt, node_cnn.onnx.
Lớp NodeCNN được cv_infer.py dùng khi dự đoán (chỉ cần onnxruntime).
"""

import os
import sys
import time

import numpy as np

from common import *

H, W = 56, 72
N_CLASSES = 16


def to_input(X):
    n = len(X)
    col = X[:, :3024].reshape(n, 28, 36, 3).repeat(2, axis=1).repeat(2, axis=2).transpose(0, 3, 1, 2)
    gray = X[:, 3024:].reshape(n, 1, H, W)
    return np.concatenate([col, gray], axis=1).astype(np.float32) / 255.0 - 0.5


def build_net():
    import torch.nn as nn

    def block(a, b):
        return nn.Sequential(
            nn.Conv2d(a, b, 3, padding=1),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
            nn.Conv2d(b, b, 3, padding=1),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
        )

    return nn.Sequential(
        block(4, 32),
        nn.MaxPool2d(2),
        block(32, 64),
        nn.MaxPool2d(2),
        block(64, 96),
        nn.MaxPool2d(2),
        block(96, 128),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.2),
        nn.Linear(128, N_CLASSES),
    )


def main(mode, epochs=8):
    print("thiết bị:", "cuda" if __import__("torch").cuda.is_available() else "cpu")
    import torch

    torch.manual_seed(0)
    tr = np.load(CACHE / "cvcnn_train.npz")
    va = np.load(CACHE / "cvcnn_validation.npz")
    X, Y = tr["cn_x"], tr["cn_y"]
    Xv, Yv = va["cn_x"], va["cn_y"]
    X, Y = with_strong(X, Y, "cn", mode, extra=[(Xv, Yv)] if mode.startswith("final") else [])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = build_net()
    init_from(net, "node_cnn.pt", mode)
    net = net.to(dev)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"CNN giao lộ: {n_params:,} tham số; {len(X)} mẫu huấn luyện", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = epochs * (len(X) // 128 + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(os.environ.get("MAX_LR", "3e-3")), total_steps=steps)
    (OUT / f"models_{mode}").mkdir(exist_ok=True)
    lossf = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    rng = np.random.default_rng(0)
    xv = torch.from_numpy(to_input(Xv))
    yv = torch.from_numpy(Yv.astype(np.int64))

    def evaluate():
        net.eval()
        ok = 0
        with torch.no_grad():
            for i in range(0, len(xv), 512):
                ok += (net(xv[i : i + 512].to(dev)).argmax(1).cpu() == yv[i : i + 512]).sum().item()
        net.train()
        return ok / len(xv)

    t0 = time.time()
    for ep in range(epochs):
        perm = rng.permutation(len(X))
        tot = 0.0
        for i in range(0, len(X), 128):
            if (i // 128) % 300 == 0:
                print(f"      ep {ep + 1} bước {i // 128}/{len(X) // 128} ({time.time() - t0:.0f}s)", flush=True)
            idx = np.sort(perm[i : i + 128])
            if len(idx) < 32:
                continue
            xb = torch.from_numpy(to_input(X[idx])).to(dev)
            yb = torch.from_numpy(Y[idx].astype(np.int64)).to(dev)
            dx, dy = rng.integers(-2, 3, 2)
            xb = torch.roll(xb, shifts=(int(dy), int(dx)), dims=(2, 3))
            loss = lossf(net(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        print(
            f"   epoch {ep + 1}/{epochs} loss {tot / len(X):.4f} val acc {evaluate():.4f}  ({time.time() - t0:.0f}s)",
            flush=True,
        )
        torch.save({k: v.cpu() for k, v in net.state_dict().items()}, OUT / f"models_{mode}" / "node_cnn.pt")
    export(mode)


def export(mode):
    import torch

    net = build_net()
    net.load_state_dict(torch.load(OUT / f"models_{mode}" / "node_cnn.pt"))
    net.eval()
    out = OUT / f"models_{mode}" / "node_cnn.onnx"
    torch.onnx.export(
        net,
        torch.zeros(2, 4, H, W),
        str(out),
        input_names=["x"],
        output_names=["logits"],
        dynamic_axes={"x": {0: "n"}, "logits": {0: "n"}},
        opset_version=17,
        dynamo=False,
    )
    (OUT / f"models_{mode}" / "node_cnn_params.txt").write_text(str(sum(p.numel() for p in net.parameters())))
    print("đã lưu", out)


class NodeCNN:
    def __init__(self, path):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.n_classes = int(self.sess.get_outputs()[0].shape[-1])

    def predict_proba_all(self, X):
        z = self.sess.run(None, {"x": to_input(np.asarray(X))})[0]
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def predict_proba(self, X):
        p = self.predict_proba_all(X)[:, :15]
        return p / p.sum(1, keepdims=True)

    def p_not_node(self, X):
        p = self.predict_proba_all(X)
        return p[:, 15] if p.shape[1] > 15 else np.zeros(len(p))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "export":
        export(sys.argv[1])
    else:
        main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3)
