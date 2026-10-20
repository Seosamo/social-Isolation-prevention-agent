"""Bounded HTML collection. Crawled documents are candidates, not verified offers."""
from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.error import HTTPError
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener
from urllib.robotparser import RobotFileParser

USER_AGENT = "IsolationPolicyBot/0.1"
HOSTS = {"gangnam.go.kr", "www.gangnam.go.kr", "chuncheon.go.kr", "www.chuncheon.go.kr"}
KEYWORDS = re.compile(r"1\s*인\s*가구|고독사|사회적?\s*고립|안부\s*확인")
EXCLUDED = re.compile(r"채용|입찰|민간위탁|수탁기관")
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}


def official(url):
    p = urlparse(url)
    return p.scheme == "https" and p.hostname in HOSTS and p.port in (None, 443) and not p.username


class SafeRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not official(newurl):
            raise ValueError("Redirect outside municipal allowlist")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class Page(HTMLParser):
    """Extract only configured content containers; navigation cannot become evidence."""
    def __init__(self, tokens):
        super().__init__(convert_charrefs=True)
        self.tokens = set(tokens)
        self.stack = []
        self.parts, self.titles, self.links = [], [], []
        self.anchor = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = set(attrs.get("class", "").split()) | {attrs.get("id", "")}
        active = bool(self.stack and self.stack[-1][1]) or bool(classes & self.tokens)
        hidden = tag in {"script", "style", "nav", "footer"} or bool(self.stack and self.stack[-1][2])
        if tag == "a" and active and not hidden:
            self.anchor = [attrs.get("href", ""), []]
        if tag not in VOID:
            self.stack.append((tag, active, hidden))
        if tag in {"p", "div", "br", "li", "tr"} and active:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag == "a" and self.anchor is not None:
            self.links.append((self.anchor[0], " ".join(self.anchor[1])))
            self.anchor = None
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        if self.stack and not self.stack[-1][2]:
            if self.stack[-1][1]:
                self.parts.append(data)
            if any(t[0] == "title" for t in self.stack):
                self.titles.append(data)
            if self.anchor is not None:
                self.anchor[1].append(data.strip())

    @property
    def text(self):
        return "\n".join(" ".join(line.split()) for line in "".join(self.parts).splitlines() if line.strip())


class MunicipalClient:
    def __init__(self):
        self.opener = build_opener(SafeRedirect())
        self.robots = {}
        self.last_request = 0.0

    def _read(self, url):
        if not official(url):
            raise ValueError("URL outside municipal allowlist")
        time.sleep(max(0, 1.0 - (time.monotonic() - self.last_request)))
        try:
            with self.opener.open(Request(url, headers={"User-Agent": USER_AGENT}), timeout=15) as response:
                body = response.read(2_000_001)
                if len(body) > 2_000_000:
                    raise ValueError("Policy response exceeds size limit")
                content_type = response.headers.get_content_type()
                if content_type not in {"text/html", "text/plain"}:
                    raise ValueError("Only HTML/text supported; attachment requires separate review")
                return body.decode(response.headers.get_content_charset() or "utf-8")
        finally:
            self.last_request = time.monotonic()

    def fetch(self, url):
        origin = "https://" + (urlparse(url).netloc or "")
        if origin not in self.robots:
            robot = RobotFileParser()
            try:
                text = self._read(origin + "/robots.txt")
            except HTTPError as error:
                if error.code != 404:
                    raise
                text = "User-agent: *\nAllow: /"
            robot.parse(text.splitlines())
            self.robots[origin] = robot
        robot = self.robots[origin]
        if not robot.can_fetch(USER_AGENT, url):
            raise ValueError("Collection disallowed by robots.txt")
        delay = robot.crawl_delay(USER_AGENT) or 1
        time.sleep(max(0, delay - (time.monotonic() - self.last_request)))
        return self._read(url)


def collect(config, fetch=None):
    settings = config.values.get("policies", {})
    source_path = config.root / settings.get("sources_file", "references/policy_sources.json")
    sources = json.loads(source_path.read_text(encoding="utf-8"))
    fetch = fetch or MunicipalClient().fetch
    rows, errors, seen = [], [], set()
    checked_at = datetime.now(timezone.utc).isoformat()
    for source in sources:
        urls = []
        discovered_titles = {}
        for listing in source.get("list_urls", []):
            try:
                page = Page(source["list_tokens"])
                page.feed(fetch(listing))
                for href, title in page.links:
                    url = urldefrag(urljoin(listing, href))[0]
                    if official(url) and re.search(source["detail_pattern"], url) and KEYWORDS.search(title) and not EXCLUDED.search(title):
                        urls.append(url)
                        discovered_titles[url] = " ".join(title.split())
            except Exception as error:
                errors.append({"source_id": source["source_id"], "url": listing, "error": str(error)})
        urls.extend(source.get("detail_urls", []))
        for url in list(dict.fromkeys(urls))[:settings.get("max_pages_per_source", 10)]:
            try:
                if not official(url):
                    raise ValueError("URL outside municipal allowlist")
                if url in seen:
                    continue
                seen.add(url)
                page = Page(source["body_tokens"])
                page.feed(fetch(url))
                text = page.text
                title = discovered_titles.get(url) or (text.splitlines()[0] if text else "")
                if len(text) < 80 or not KEYWORDS.search(text):
                    raise ValueError("Relevant policy body missing; parser or page requires review")
                eligibility = " / ".join(line for line in text.splitlines() if re.search(r"모집대상|참여대상|지원대상|신청대상|지원자격", line))[:400]
                rows.append({"resource_id": hashlib.sha256(url.encode()).hexdigest()[:16],
                             "source_id": source["source_id"], "region": source["region"],
                             "title": title, "url": url, "checked_at": checked_at,
                             "content_hash": hashlib.sha256(text.encode()).hexdigest(),
                             "excerpt": text[:800], "status": "needs_verification",
                             "eligibility": ("원문 대상 안내: " + eligibility + "; 상세 조건 확인 필요") if eligibility else "원문에서 거주지·연령·소득·가구 조건 확인 필요",
                             "recruitment": "모집 기간·현재 접수 여부 미확인"})
            except Exception as error:
                errors.append({"source_id": source["source_id"], "url": url, "error": str(error)})
    directory = config.output / "policies"
    directory.mkdir(parents=True, exist_ok=True)
    # A failed refresh must not erase the last successful collection.
    path = directory / "catalog.json"
    if rows:
        temporary = directory / "catalog.tmp"
        temporary.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    result = {"checked_at": checked_at, "collected": len(rows), "errors": errors,
              "previous_catalog_preserved": not rows and path.exists()}
    (directory / "collection_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
