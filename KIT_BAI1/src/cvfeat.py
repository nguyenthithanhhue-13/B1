"""Hàm xử lý ảnh dùng chung cho CV: đọc ảnh RGB, cắt và chuẩn hóa mảnh ảnh, đặc trưng màu / kiểu nét. Thư viện."""

import cv2
import numpy as np
from PIL import Image

cv2.setNumThreads(1)

PAD = 64
DET_WIN = 24
EDGE_W, EDGE_H = 48, 24
LOOKS = ["none", "normal", "crowded", "covered", "closed"]
ROAD_KINDS = ["normal", "crowded", "covered"]
LEGEND_KINDS = [
    "library",
    "dorm",
    "sports",
    "clinic",
    "canteen",
    "parking",
    "lecture",
    "lab",
    "office",
    "gate",
    "normal",
    "crowded",
    "covered",
    "closed",
    "stairs",
    "oneway",
    "robot",
    "weather",
    "none",
]
K_NONE = 18
ALIGN_STEP, ALIGN_BINS = 0.015, 6
ALIGN_MAX = 0.095


def read_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def grid_unit(rc, xy):
    rc = np.asarray(rc, float)
    xy = np.asarray(xy, float)
    A = np.c_[rc[:, 1], rc[:, 0], np.ones(len(rc))]
    coef, *_ = np.linalg.lstsq(A, xy, rcond=None)
    return float(min(np.hypot(*coef[0]), np.hypot(*coef[1])))


def degrade(img, rng):
    out = img
    if rng.random() < 0.8:
        out = cv2.GaussianBlur(out, (0, 0), float(rng.uniform(0.6, 1.0)))
    if rng.random() < 0.8:
        q = int(rng.integers(40, 88))
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
        out = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return out


class Img:

    def __init__(self, rgb):
        self.h, self.w = rgb.shape[:2]
        self.full = cv2.copyMakeBorder(rgb, PAD, PAD, PAD, PAD, cv2.BORDER_REPLICATE)
        half = cv2.resize(rgb, (self.w // 2, self.h // 2), interpolation=cv2.INTER_AREA)
        self.half = cv2.copyMakeBorder(half, PAD, PAD, PAD, PAD, cv2.BORDER_REPLICATE)
        quarter = cv2.resize(rgb, (self.w // 4, self.h // 4), interpolation=cv2.INTER_AREA)
        self.quarter = cv2.copyMakeBorder(quarter, PAD, PAD, PAD, PAD, cv2.BORDER_REPLICATE)
        self.gray = cv2.cvtColor(self.full, cv2.COLOR_RGB2GRAY)

    def _clip(self, x, y):
        return min(max(x, -PAD // 2), self.w + PAD // 2), min(max(y, -PAD // 2), self.h + PAD // 2)

    def det_patch(self, x, y):
        x, y = self._clip(x, y)
        gx, gy = int(round(x / 4)), int(round(y / 4))
        r = DET_WIN // 2
        cx, cy = gx * 2 + PAD, gy * 2 + PAD
        a = self.half[cy - r : cy + r, cx - r : cx + r]
        qx, qy = gx + PAD, gy + PAD
        b = self.quarter[qy - r : qy + r, qx - r : qx + r]
        return np.concatenate([a.reshape(-1), b.reshape(-1)])

    def _warp(self, src, cx, cy, ux, uy, span_w, span_h, out_w, out_h):
        nx, ny = -uy, ux
        sw, sh = span_w / out_w, span_h / out_h
        M = np.array(
            [
                [ux * sw, nx * sh, cx + PAD - span_w / 2 * ux - span_h / 2 * nx],
                [uy * sw, ny * sh, cy + PAD - span_w / 2 * uy - span_h / 2 * ny],
            ],
            np.float32,
        )
        interp = cv2.INTER_AREA if max(sw, sh) > 1.3 else cv2.INTER_LINEAR
        if interp == cv2.INTER_AREA:
            k = max(sw, sh)
            x0, y0 = int(cx + PAD - span_w), int(cy + PAD - span_w)
            x0, y0 = max(x0, 0), max(y0, 0)
            sub = src[y0 : int(cy + PAD + span_w) + 1, x0 : int(cx + PAD + span_w) + 1]
            sub = cv2.GaussianBlur(sub, (0, 0), 0.4 * k)
            M = M.copy()
            M[0, 2] -= x0
            M[1, 2] -= y0
            return cv2.warpAffine(
                sub, M, (out_w, out_h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE
            )
        return cv2.warpAffine(
            src, M, (out_w, out_h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE
        )

    def node_crop(self, x, y, unit):
        x, y = self._clip(x, y)
        g = self._warp(self.gray, x, y, 1.0, 0.0, 1.0 * unit, 0.78 * unit, 72, 56)
        c = self._warp(self.full, x, y, 1.0, 0.0, 1.0 * unit, 0.78 * unit, 36, 28)
        return np.concatenate([c.reshape(-1), g.reshape(-1)])

    def edge_crop(self, a, b, unit):
        ax, ay = a
        bx, by = b
        d = np.hypot(bx - ax, by - ay) + 1e-6
        ux, uy = (bx - ax) / d, (by - ay) / d
        out = self._warp(self.full, (ax + bx) / 2, (ay + by) / 2, ux, uy, 0.6 * unit, 0.3 * unit, EDGE_W, EDGE_H)
        return out.reshape(-1)

    def label_offset(self, centers):
        return float(np.median([self.label_start(x, y) - x for x, y in centers]))

    def swatch_crop(self, x, y, offset=None):
        x, y = self._clip(x, y)
        cx, cy = int(round(x / 2)) + PAD, int(round(y / 2)) + PAD
        a = self.half[cy - 12 : cy + 12, cx - 18 : cx + 18].reshape(-1)
        xs = x + (offset if offset is not None else self.label_start(x, y) - x)
        fx, fy = int(round(x)) + PAD, int(round(y)) + PAD
        g = self.gray[fy - 14 : fy + 14, fx - 32 : fx + 32].reshape(-1)
        return np.concatenate([a, self.label_patch(xs, y).reshape(-1), g])

    def label_patch(self, xs, y):
        fx, fy = int(round(xs)) + PAD, int(round(y)) + PAD
        fx = min(max(fx, 20), self.gray.shape[1] - 60)
        reg = self.gray[fy - 20 : fy + 21, fx - 1 : fx + 300].astype(np.int16)
        bg = np.median(reg[:, :60])
        ink = np.abs(reg - bg) > 45

        def extent(w):
            rows = np.nonzero(ink[:, : max(int(w), 8)].any(1))[0]
            if len(rows) == 0:
                return 12, 28
            mid = rows[np.argmin(np.abs(rows - 20))]
            lo = hi = mid
            rs = set(rows.tolist())
            while lo - 1 in rs or lo - 2 in rs:
                lo -= 1 if lo - 1 in rs else 2
            while hi + 1 in rs or hi + 2 in rs:
                hi += 1 if hi + 1 in rs else 2
            return lo, hi

        lo, hi = extent(45)
        h = max(hi - lo + 1, 8)
        lo, hi = extent(3.5 * h)
        h = float(min(max(hi - lo + 1, 8), 30))
        cx = xs - 0.2 * h + 5.6 * h
        cyc = y - 20 + (lo + hi) / 2.0
        return self._warp(self.gray, cx, cyc, 1.0, 0.0, 11.2 * h, 1.3 * h, 144, 16)

    def label_start(self, x, y):
        fx, fy = int(round(x)) + PAD, int(round(y)) + PAD
        band = self.gray[fy - 9 : fy + 10, fx + 10 : fx + 120].astype(np.int16)
        if band.shape[1] < 20:
            return x + 30
        bg = np.median(band)
        ink = (np.abs(band - bg) > 45).sum(0) >= 1
        gap = 0
        seen_gap = False
        for i, v in enumerate(ink):
            if not v:
                gap += 1
                if gap >= 3:
                    seen_gap = True
            else:
                if seen_gap:
                    return x + 10 + i
                gap = 0
        return x + 30

    def ring_color(self, x, y, r0, r1):
        r = int(r1) + 1
        fx, fy = int(round(x)) + PAD, int(round(y)) + PAD
        sub = self.full[fy - r : fy + r + 1, fx - r : fx + r + 1].astype(np.float32)
        ys, xs = np.mgrid[-r : r + 1, -r : r + 1]
        d = np.hypot(xs, ys)
        m = (d >= r0) & (d <= r1)
        return np.median(sub[m], axis=0)

    def weather_crop(self, x, y):
        x, y = self._clip(x, y)
        cx, cy = int(round(x / 2)) + PAD, int(round(y / 2)) + PAD
        return self.half[cy - 20 : cy + 20, cx - 20 : cx + 20].reshape(-1)

    def dense_windows(self):
        r = DET_WIN // 2
        hq, wq = self.h // 4, self.w // 4
        region = self.half[PAD - r : PAD + 2 * hq + r, PAD - r : PAD + 2 * wq + r]
        v = np.lib.stride_tricks.sliding_window_view(region, (DET_WIN, DET_WIN), axis=(0, 1))
        v = v[::2, ::2].transpose(0, 1, 3, 4, 2)[:hq, :wq]
        regq = self.quarter[PAD - r : PAD + hq + r, PAD - r : PAD + wq + r]
        q = np.lib.stride_tricks.sliding_window_view(regq, (DET_WIN, DET_WIN), axis=(0, 1)).transpose(0, 1, 3, 4, 2)[
            :hq, :wq
        ]
        return v, q, 4
