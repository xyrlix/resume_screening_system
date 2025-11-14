import requests
from bs4 import BeautifulSoup
from utils.clean_tools import clean_text
from modules.debias import mask_sensitive, collect_fairness_tags
from utils.logger import get_logger

def _parse_cookie_str(cookie: str) -> dict:
    jar = {}
    for part in (cookie or "").split(";"):
        if "=" in part:
            k, v = part.strip().split("=", 1)
            jar[k.strip()] = v.strip()
    return jar

def fetch_bosszhipin(urls: list[str], cookie: str) -> list[dict]:
    logger = get_logger("boss")
    sess = requests.Session()
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.zhipin.com/"}
    cookies = _parse_cookie_str(cookie)
    out = []
    for u in urls or []:
        try:
            r = sess.get(u, headers=headers, cookies=cookies, timeout=8)
            if r.status_code != 200:
                logger.info(f"boss_fail status={r.status_code} url={u}")
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            for s in soup.find_all(["script", "style"]):
                s.extract()
            txt = soup.get_text("\n")
            cleaned = clean_text(txt)
            masked = mask_sensitive(cleaned)
            fair = collect_fairness_tags(cleaned)
            out.append({"url": u, "text": cleaned, "masked_text": masked, "fairness_tags": fair})
            logger.info(f"boss_ingested url_len={len(u)} text_len={len(cleaned)}")
        except Exception:
            logger.info(f"boss_error url={u}")
    return out