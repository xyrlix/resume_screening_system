from typing import List, Dict
import requests
from bs4 import BeautifulSoup
from utils.logger import get_logger
from utils.clean_tools import clean_text
from modules.debias import mask_sensitive, collect_fairness_tags

def _parse_cookie_str(cookie: str) -> Dict[str, str]:
    out = {}
    for part in (cookie or "").split(";"):
        kv = part.strip().split("=", 1)
        if len(kv) == 2:
            k = kv[0].strip()
            v = kv[1].strip()
            if k:
                out[k] = v
    return out

def fetch_linkedin(urls: List[str], cookie: str) -> List[Dict]:
    logger = get_logger("linkedin")
    sess = requests.Session()
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.linkedin.com/"}
    cookies = _parse_cookie_str(cookie)
    out: List[Dict] = []
    for u in urls or []:
        try:
            r = sess.get(u, headers=headers, cookies=cookies, timeout=8)
            if r.status_code != 200:
                logger.info(f"linkedin_fail status={r.status_code} url={u}")
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            for s in soup.find_all(["script", "style"]):
                s.extract()
            txt = soup.get_text("\n")
            cleaned = clean_text(txt)
            masked = mask_sensitive(cleaned)
            fair = collect_fairness_tags(cleaned)
            out.append({"url": u, "text": cleaned, "masked_text": masked, "fairness_tags": fair})
            logger.info(f"linkedin_ingested url_len={len(u)} text_len={len(cleaned)}")
        except Exception:
            logger.info(f"linkedin_error url={u}")
    return out