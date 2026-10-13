"""Bộ nhúng câu multilingual-e5-small (ONNX, chạy CPU) dùng trong bộ đọc yêu cầu; vector đã tính được nhớ trong cache/e5_cache.pkl.
Cách chạy (tải mô hình một lần, ~470 MB, phiên bản cố định, kiểm tra SHA-256):  python src/embed.py download
Đọc: outputs/e5_small/onnx/model.onnx, outputs/e5_small/tokenizer.json.
"""

import pickle
import sys

import numpy as np

from common import CACHE, MODELS

MODEL_DIR = MODELS / "e5_small"
REPO = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
MODEL_SHA256 = "ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665"


def download():
    import hashlib
    from huggingface_hub import hf_hub_download

    MODEL_DIR.mkdir(exist_ok=True)
    p = hf_hub_download(REPO, "onnx/model.onnx", revision=REVISION, local_dir=MODEL_DIR)
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    print("đã tải", p, "sha256 khớp" if h == MODEL_SHA256 else "SAI sha256: " + h)


class Embedder:
    def __init__(self):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
        self.tok.enable_padding(pad_id=1, pad_token="<pad>")
        self.tok.enable_truncation(max_length=64)
        so = ort.SessionOptions()
        so.intra_op_num_threads = 8
        self.sess = ort.InferenceSession(str(MODEL_DIR / "onnx" / "model.onnx"), so, providers=["CPUExecutionProvider"])
        self.names = [i.name for i in self.sess.get_inputs()]
        self.cache_path = CACHE / "e5_cache.pkl"
        self.cache = pickle.load(open(self.cache_path, "rb")) if self.cache_path.exists() else {}
        self._dirty = 0

    def encode(self, texts, batch=64, prefix="query: "):
        texts = [prefix + t for t in texts]
        todo = sorted({t for t in texts if t not in self.cache})
        for i in range(0, len(todo), batch):
            chunk = todo[i : i + batch]
            enc = self.tok.encode_batch(chunk)
            ids = np.array([e.ids for e in enc], np.int64)
            mask = np.array([e.attention_mask for e in enc], np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.names:
                feed["token_type_ids"] = np.zeros_like(ids)
            h = self.sess.run(None, feed)[0]
            m = mask[..., None].astype(np.float32)
            v = (h * m).sum(1) / m.sum(1)
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            for t, x in zip(chunk, v):
                self.cache[t] = x.astype(np.float32)
            self._dirty += len(chunk)
        return np.array([self.cache[t] for t in texts])

    def save(self):
        if self._dirty:
            import os

            tmp = self.cache_path.with_suffix(f".{os.getpid()}.tmp")
            with open(tmp, "wb") as f:
                pickle.dump(self.cache, f)
            try:
                os.replace(tmp, self.cache_path)
            except OSError:
                os.remove(tmp)
            self._dirty = 0


if __name__ == "__main__":
    if sys.argv[1:] == ["download"]:
        download()
    e = Embedder()
    v = e.encode(["thư viện", "nơi mượn sách", "bãi đỗ xe", "gấp lắm", "khẩn trương lên", "không vội đâu"])
    print(np.round(v @ v.T, 2))
    e.save()
