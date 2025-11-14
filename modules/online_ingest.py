import os
import json
import requests
from bs4 import BeautifulSoup
from utils.clean_tools import clean_text
from modules.debias import mask_sensitive, collect_fairness_tags
from utils.logger import get_logger

def fetch_text(url: str, headers: dict | None = None, cookies: dict | None = None) -> str:
    logger = get_logger("online")
    try:
        h = headers or {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=h, cookies=cookies, timeout=8)
        if r.status_code != 200:
            logger.info(f"fetch_fail status={r.status_code} url={url}")
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        for s in soup.find_all(["script", "style"]):
            s.extract()
        txt = soup.get_text("\n")
        return txt
    except Exception:
        logger.info(f"fetch_error url={url}")
        return ""

def ingest_urls(urls: list[str]) -> list[dict]:
    logger = get_logger("online")
    out = []
    for u in urls or []:
        raw = fetch_text(u)
        if not raw:
            continue
        cleaned = clean_text(raw)
        masked = mask_sensitive(cleaned)
        fair = collect_fairness_tags(cleaned)
        out.append({"url": u, "text": cleaned, "masked_text": masked, "fairness_tags": fair, "entities": []})
        logger.info(f"ingested url_len={len(u)} text_len={len(cleaned)}")
    if out:
        os.makedirs(os.path.join("data", "processed"), exist_ok=True)
        p = os.path.join("data", "processed", "online_resumes.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        logger.info(f"saved count={len(out)} path={p}")
    return out