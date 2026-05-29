"""Build a tiny client-side search index for the LindbladTTN docs.

Walks every *.html file in this docs/ folder, extracts the title, all
section headings (h1/h2/h3) and the visible body text, and writes
docs/search-index.json. `main.js` fetches this file from the nav search
bar and filters it in the browser.
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

DOCS = Path(__file__).resolve().parent
OUT  = DOCS / "search-index.json"

SKIP_PAGES: set[str] = {"build_search_index.py", "build_notebooks.py"}

# Match <script>...</script> and <style>...</style>
RE_SCRIPT = re.compile(r"<script\b[^>]*>.*?</script>", re.DOTALL | re.IGNORECASE)
RE_STYLE  = re.compile(r"<style\b[^>]*>.*?</style>",   re.DOTALL | re.IGNORECASE)
RE_NAV    = re.compile(r"<nav\b[^>]*>.*?</nav>",        re.DOTALL | re.IGNORECASE)
RE_FOOTER = re.compile(r"<footer\b[^>]*>.*?</footer>",  re.DOTALL | re.IGNORECASE)
RE_TAG    = re.compile(r"<[^>]+>")
RE_WS     = re.compile(r"\s+")
RE_TITLE  = re.compile(r"<title>(.*?)</title>", re.DOTALL | re.IGNORECASE)
RE_H      = re.compile(r"<h([1-3])\b[^>]*>(.*?)</h\1>", re.DOTALL | re.IGNORECASE)


def strip(text: str) -> str:
    text = RE_SCRIPT.sub(" ", text)
    text = RE_STYLE.sub(" ", text)
    text = RE_NAV.sub(" ", text)
    text = RE_FOOTER.sub(" ", text)
    text = RE_TAG.sub(" ", text)
    text = html.unescape(text)
    return RE_WS.sub(" ", text).strip()


def page_title(text: str, fallback: str) -> str:
    m = RE_TITLE.search(text)
    if not m:
        return fallback
    title = html.unescape(RE_TAG.sub("", m.group(1))).strip()
    return title.split("·")[0].strip() or fallback


def sections(text: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for m in RE_H.finditer(text):
        heading = html.unescape(RE_TAG.sub("", m.group(2))).strip()
        out.append({"heading": heading})
    return out


def main() -> None:
    index: list[dict] = []
    for path in sorted(DOCS.glob("*.html")):
        raw = path.read_text(encoding="utf-8", errors="ignore")
        title = page_title(raw, fallback=path.stem)
        headings = sections(raw)
        body = strip(raw)
        # Drop super-short pages (likely redirects)
        if len(body) < 60:
            continue
        # Single combined section: heading list + full body. Cheap and works.
        index.append({
            "url": path.name,
            "title": title,
            "sections": [
                {
                    "heading": " / ".join(h["heading"] for h in headings[:6]),
                    "text": body[:4000],
                }
            ],
        })

    OUT.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  wrote {OUT.relative_to(DOCS.parent)}  ({len(index)} pages)")


if __name__ == "__main__":
    main()
