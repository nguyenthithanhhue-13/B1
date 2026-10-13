"""Bộ đọc yêu cầu vòng private (MissionParser3): khôi phục câu cờ gõ sai, thay cụm mô tả qua bản đồ bằng tên loại ở vị trí đó,
đọc phần còn lại bằng MissionParser2, chốt đích đúng vị trí. Thư viện; học / đọc thông qua src/pipeline_p2.py.
"""

import pickle
import re

import mapref
from nlp2 import MissionParser2, resolve_with_map, CANONICAL_ACC, PLACE_TYPES, _unaccent

SENT_RESTORE = True
_INV = None


def _load_inv():
    global _INV
    if _INV is None:
        import json
        from common import OUT

        p = OUT / "sent_inventory.json"
        _INV = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
    return _INV


def restore_sentences(text):
    from nlp import dl_distance

    inv = _load_inv()
    if not inv:
        return text
    keys = {k for k, _ in inv}
    parts = re.split(r"(?<=[.!?])(\s+)", text)
    out = []
    for s in parts:
        core = s.strip()
        body = core.rstrip(".!?")
        key = re.sub(r"\s+", " ", _unaccent(body.lower())).strip()
        if len(key) < 6 or key in keys:
            out.append(s)
            continue
        thr = 1 if len(key) < 12 else (2 if len(key) < 25 else 3)
        hits = {k: c for k, c in inv if abs(len(k) - len(key)) <= thr and dl_distance(key, k, thr) <= thr}
        if len(hits) == 1:
            k, c = next(iter(hits.items()))
            new = (k if core.isascii() else c) + core[len(body) :]
            out.append(s.replace(core, new, 1) if core else s)
        else:
            out.append(s)
    return "".join(out)


def _name(t, accented):
    v = CANONICAL_ACC[t]
    return v if accented else _unaccent(v)


def _subst(text, span, t):
    a, b = span
    return text[:a] + _name(t, not text.isascii()) + text[b:]


_KP = r"(?:khong|kong|khng|hkong|kohng|khogn|khog)\s+(?:phai|hai|pahi|phi|pai|phia|phair|phaii)"
_END = r"(?=\s*(?:[.,;:!?)(]|$)|\s+(?:giup|nhe|nha|dang|nho|truoc|roi|nhung)\b)"


def is_mapref_mission(m):
    g = m.get("goal_ref")
    return bool(g) and (g["kind"] in mapref.KINDS_MOST or g["kind"] == "anchor_near")


class MissionParser3:
    def fit(self, missions, verbose=False):
        self.vocab = mapref.build_vocab([m["text"] for m in missions])
        mapref.VOCAB = self.vocab
        ms, skipped = [], 0
        for m in missions:
            if is_mapref_mission(m):
                f = mapref.find(m["text"], self.vocab)
                if not f or not f["span"]:
                    skipped += 1
                    continue
                ms.append(dict(m, text=_subst(m["text"], f["span"], m["goal"]), goal_ref=None))
            else:
                ms.append(m)
        if verbose:
            print("nlp3: câu mô tả qua bản đồ không tìm được cụm (bỏ khi học):", skipped)
        import nlp2 as _n2
        from nlp import norm, tokens
        from collections import defaultdict
        from nlp_knowledge import PLACES

        self.lex_kb = getattr(self, "lex_kb", __import__("os").environ.get("NLP_LEX", "canon"))
        PERSONS = {"thủ thư", "đội bóng", "bếp trưởng", "lớp trưởng", "văn thư"}
        kb = defaultdict(set)
        for t, lst in PLACES.items():
            extra = (
                lst
                if self.lex_kb == "full"
                else [a for a in lst if a not in PERSONS] if self.lex_kb == "places" else []
            )
            for a in extra + [CANONICAL_ACC[t]]:
                kb[" ".join(tokens(norm(a)))].add(t)
        kb = {a: next(iter(ts)) for a, ts in kb.items() if a and len(ts) == 1}
        orig = _n2.mine_lexicon
        _n2.mine_lexicon = lambda missions: {**kb, **orig(missions)}
        try:
            self.p2 = MissionParser2().fit(ms, verbose=verbose)
        finally:
            _n2.mine_lexicon = orig
        return self

    def type_anchor(self, text, f, one):
        a, b = f["anchor_span"]
        mini = ("Giao hàng tới " if not text.isascii() else "Giao hang toi ") + text[a:b] + "."
        r = self.p2.parse(mini, set(one))
        if r.get("goal_known") and r["goal"] in one:
            return r["goal"]
        d = r["goal_dist"]
        return max(one, key=lambda t: d[PLACE_TYPES.index(t)])

    def parse(self, text, present=None, landmarks=None):
        if SENT_RESTORE and __import__("os").environ.get("NO_RESTORE") != "1":
            text = restore_sentences(text)
        mapref.VOCAB = self.vocab
        f = mapref.find(text, self.vocab)
        if f and f["span"] and landmarks:
            at = None
            if f["kind"] == "anchor_near":
                one = [t for t, v in landmarks.items() if len(v) == 1]
                at = self.type_anchor(text, f, one) if one else None
            pos = mapref.resolve(f["kind"], landmarks, at) if (f["kind"] != "anchor_near" or at) else None
            if pos:
                gt = next(t for t, v in landmarks.items() if pos in v)
                r = dict(self.p2.parse(_subst(text, f["span"], gt), present))
                r.update(goal=gt, goal_ref=("pos", pos), goal_known=True, mapref=(f["kind"], at, pos))
                return r
        r = dict(self.p2.parse(text, present))
        if f:
            r["mapref"] = (f["kind"], None, None)
        return r

    def save(self, path):
        pickle.dump(self, open(path, "wb"))

    @staticmethod
    def load(path):
        return pickle.load(open(path, "rb"))


def resolve3(m, landmarks, world=None):
    if m.get("goal_ref") and m["goal_ref"][0] == "pos":
        pos = m["goal_ref"][1]
        out = resolve_with_map(dict(m, goal_ref=None, goal_known=True), landmarks, world)
        out["goal"], out["goal_ref"] = m["goal"], ("pos", pos)
        return out
    return resolve_with_map(m, landmarks, world)
