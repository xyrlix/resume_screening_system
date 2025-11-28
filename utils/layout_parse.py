try:
    import pdfplumber
except ImportError:
    pdfplumber = None
from typing import List, Dict, Any

def layout_aware_parse(pdf_path: str):
    if pdfplumber is None:
        raise ImportError("pdfplumber is not installed")
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

def parse_document_layout(document_path: str) -> Dict[str, Any]:
    """解析文档布局的简单实现"""
    # 使用现有的layout_aware_parse函数作为基础实现
    try:
        if document_path.lower().endswith('.pdf'):
            text, elements = layout_aware_parse(document_path)
            return {
                "success": True,
                "text": text,
                "elements": elements,
                "page_count": 1  # layout_aware_parse没有返回页数，这里简化处理
            }
        else:
            return {
                "success": False,
                "error": "Unsupported document type"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def extract_layout_elements(layout_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从布局数据中提取布局元素"""
    if layout_data.get("success"):
        return layout_data.get("elements", [])
    return []
