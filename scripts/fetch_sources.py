"""Fetch the pages listed in sources.yaml, save clean text, record fetch date and hash.

Setup:  pip install requests beautifulsoup4 pyyaml
Run:    python fetch_sources.py

Saves docs/<id>.txt and writes fetch_date and sha256 (of the saved text) back into sources.yaml.
Re-run before deploy: if a hash changes, the page changed and the rules need re-checking.
"""
import hashlib
from datetime import date
from pathlib import Path

import requests
import yaml
from bs4 import BeautifulSoup

SOURCES = Path("sources.yaml")
DOCS = Path("docs")


def page_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "form", "noscript"]):
        tag.decompose()
    main = soup.find("main") or soup.body or soup
    lines = [ln.strip() for ln in main.get_text("\n").splitlines()]
    return "\n".join(ln for ln in lines if ln)


def main() -> None:
    config = yaml.safe_load(SOURCES.read_text(encoding="utf-8"))
    DOCS.mkdir(exist_ok=True)
    for src in config["sources"]:
        resp = requests.get(src["url"], timeout=30, headers={"User-Agent": "mortgage-checker-portfolio"})
        resp.raise_for_status()
        text = page_text(resp.text)
        (DOCS / f"{src['id']}.txt").write_text(text, encoding="utf-8")
        new_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if src.get("sha256") and src["sha256"] != new_hash:
            print(f"CHANGED: {src['id']} (hash differs from last fetch) - re-check the rules")
        src["fetch_date"] = date.today().isoformat()
        src["sha256"] = new_hash
        print(f"saved docs/{src['id']}.txt  ({len(text)} chars)")
    SOURCES.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
