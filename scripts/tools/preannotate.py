import os
import json
import re

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def find_spans(text, substr):
    spans = []
    start = 0
    s = str(substr)
    while True:
        i = text.find(s, start)
        if i == -1:
            break
        spans.append((i, i + len(s)))
        start = i + len(s)
    return spans

def preannotate_item(text, cfg):
    ents = []
    for sk in cfg.get("skills", []) or []:
        for s, e in find_spans(text.lower(), str(sk).lower()):
            ents.append({"type": "SKILL", "start": s, "end": e})
    for c in cfg.get("certs", []) or []:
        for s, e in find_spans(text.lower(), str(c).lower()):
            ents.append({"type": "CERT_TYPE", "start": s, "end": e})
    for l in cfg.get("languages", []) or []:
        for s, e in find_spans(text.lower(), str(l).lower()):
            ents.append({"type": "LANG_LEVEL", "start": s, "end": e})
    for d in cfg.get("domains", []) or []:
        for s, e in find_spans(text.lower(), str(d).lower()):
            ents.append({"type": "DOMAIN", "start": s, "end": e})
    for m in cfg.get("methods", []) or []:
        for s, e in find_spans(text.lower(), str(m).lower()):
            ents.append({"type": "METHOD", "start": s, "end": e})
    prof = re.findall(r"(熟练|精通|掌握)\s*([A-Za-z0-9#\-\+]{2,})", text)
    for p in prof[:20]:
        s = text.find("".join(p))
        if s != -1:
            ents.append({"type": "SKILL_PROFICIENCY", "start": s, "end": s + len("".join(p))})
    metrics = re.findall(r"(提升|降低|增长|减少)\s*(\d+[\.]?\d*\s*%?)", text)
    for m in metrics[:20]:
        s = text.find("".join(m))
        if s != -1:
            ents.append({"type": "RESULT_METRIC", "start": s, "end": s + len("".join(m))})
    return ents

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    src = os.path.join(base_dir, "data", "processed", "resumes_for_annotation.json")
    out = os.path.join(base_dir, "data", "processed", "entity_train_pre.json")
    cfg = os.path.join(base_dir, "config", "matching.json")
    data = load_json(src)
    conf = load_json(cfg)
    rows = []
    for i, it in enumerate(data):
        t = str(it.get("text", ""))
        rows.append({"id": i, "text": t, "entities": preannotate_item(t, conf)})
    save_json(out, rows)

if __name__ == "__main__":
    main()