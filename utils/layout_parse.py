import pdfplumber

def layout_aware_parse(pdf_path: str):
    text_blocks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            words = page.extract_words(extra_attrs=["size"]) or []
            for w in words:
                txt = w.get("text", "").strip()
                if not txt:
                    continue
                x0 = float(w.get("x0", 0.0))
                y0 = float(w.get("top", 0.0))
                fs = float(w.get("size", 10.0))
                text_blocks.append({"text": txt, "x0": x0, "y0": y0, "font_size": fs})
    text_blocks.sort(key=lambda x: (-x["font_size"], -x["y0"], x["x0"]))
    sorted_text = "\n".join([b["text"] for b in text_blocks])
    return sorted_text, text_blocks

def index_backfill(sorted_text: str, text_blocks, llm_response: dict):
    result = {}
    lines = (sorted_text or "").split("\n")
    for key, line_range in (llm_response or {}).items():
        try:
            start, end = map(int, str(line_range).split("-"))
            result[key] = "\n".join(lines[max(start-1, 0):max(end, 0)])
        except Exception:
            result[key] = ""
    return result