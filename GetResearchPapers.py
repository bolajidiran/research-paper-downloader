from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import quote_plus, urljoin

import requests
from bs4 import BeautifulSoup

try:
    import PyPDF2
except Exception:
    PyPDF2 = None

try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    from scholarly import scholarly
except Exception:
    scholarly = None


REQUEST_TIMEOUT = 45
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


@dataclass
class PaperRecord:
    title: str
    authors: List[str] = field(default_factory=list)
    abstract: str = ""
    source: str = ""
    year: str = "Unknown"
    url: str = ""
    pdf_url: str = ""
    category: str = ""
    doi: str = ""
    venue: str = ""
    relevance_score: float = 0.0
    relevance_breakdown: Dict[str, float] = field(default_factory=dict)
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> str:
        title = normalize_text(self.title)
        author_key = "|".join(normalize_text(a) for a in self.authors)
        return f"{title}|{author_key}"


def normalize_text(value: str) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9\-_ .]+", "", name or "paper")
    cleaned = cleaned.strip()
    return cleaned[:180] if cleaned else "paper"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def prompt_for_storage_dir() -> Path:
    while True:
        value = input("Enter the exact directory to store downloaded papers: ").strip()
        if not value:
            print("Storage directory cannot be empty. Please try again.")
            continue
        path = Path(value).expanduser()
        ensure_dir(path)
        return path


def download_file(url: str, destination: Path) -> bool:
    if not url:
        return False
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT}, stream=True)
        r.raise_for_status()
        ensure_dir(destination.parent)
        with open(destination, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as exc:
        print(f"Download failed for {url}: {exc}")
        return False


def decode_inverted_index(data: Optional[Dict[str, Any]]) -> str:
    if not data:
        return ""
    text = {}
    for token, positions in data.items():
        text[int(token)] = token
    chars = [""] * (max(text.keys(), default=0) + 1)
    for idx, token in text.items():
        chars[idx] = token
    return "".join(chars).replace("###", " ").strip()


def fetch_text(url: str, timeout: int = REQUEST_TIMEOUT) -> str:
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        r.raise_for_status()
        return r.text
    except Exception as exc:
        print(f"Request failed: {url} -> {exc}")
        return ""


def extract_keywords(query: str) -> List[str]:
    """Extract individual keywords from query, prioritizing multi-word phrases."""
    stop_words = {"the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with", "by", "from", "is", "are", "was", "were", "be", "been"}
    query_lower = query.lower()
    
    # First, try to extract meaningful multi-word phrases
    phrases = []
    words = query_lower.split()
    
    # Look for 2-3 word phrases
    for i in range(len(words) - 1):
        phrase = " ".join(words[i:i+2])
        if not any(sw in phrase for sw in stop_words):
            phrases.append(phrase)
    
    # Add individual keywords (length > 3 to avoid short words)
    individual_words = [w for w in words if len(w) > 3 and w not in stop_words]
    
    # Combine and deduplicate
    all_keywords = phrases + individual_words
    seen = set()
    result = []
    for kw in all_keywords:
        if kw not in seen:
            result.append(kw)
            seen.add(kw)
    
    return result[:10]  # Limit to top 10 keywords


def calculate_relevance_score(record: PaperRecord, query: str) -> Tuple[float, Dict[str, float]]:
    """
    Calculate relevance score with detailed breakdown.
    Weights:
      - Exact phrase match in title: 1.0
      - Multi-word phrase in title: 0.8
      - Single keyword in title: 0.6
      - Exact phrase in abstract: 0.6
      - Multi-word phrase in abstract: 0.4
      - Single keyword in abstract: 0.2
      - Author relevance bonus: 0.1
      - Venue/journal quality bonus: 0.05
    """
    keywords = extract_keywords(query)
    if not keywords:
        return 0.5, {"no_keywords": 0.5}
    
    breakdown: Dict[str, float] = {}
    title_lower = record.title.lower()
    abstract_lower = record.abstract.lower()
    
    title_score = 0.0
    abstract_score = 0.0
    author_score = 0.0
    venue_score = 0.0
    
    matched_phrases = set()
    
    # Check for phrase matches (2+ word combinations)
    for i in range(len(keywords) - 1):
        phrase = keywords[i] + " " + keywords[i + 1]
        if phrase in title_lower:
            title_score += 0.8
            matched_phrases.add(phrase)
        elif phrase in abstract_lower:
            abstract_score += 0.4
            matched_phrases.add(phrase)
    
    # Check for individual keyword matches
    for keyword in keywords:
        if keyword in matched_phrases:
            continue
        
        # Title matches (highest weight)
        if keyword in title_lower:
            title_score += 0.6
        # Abstract matches (medium weight)
        elif keyword in abstract_lower:
            abstract_score += 0.2
    
    # Author relevance bonus
    authors_text = " ".join(record.authors).lower()
    for keyword in keywords:
        if keyword in authors_text and len(keyword) > 4:
            author_score += 0.05
    
    # Venue/journal quality bonus (if venue is mentioned in query)
    if record.venue:
        venue_lower = record.venue.lower()
        for keyword in keywords:
            if keyword in venue_lower:
                venue_score += 0.05
    
    breakdown["title_score"] = min(title_score, 1.0)
    breakdown["abstract_score"] = min(abstract_score, 1.0)
    breakdown["author_score"] = min(author_score, 0.2)
    breakdown["venue_score"] = min(venue_score, 0.1)
    
    # Weighted combination
    total_score = (
        breakdown["title_score"] * 0.50 +
        breakdown["abstract_score"] * 0.35 +
        breakdown["author_score"] * 0.10 +
        breakdown["venue_score"] * 0.05
    )
    
    final_score = min(max(total_score, 0.0), 1.0)
    breakdown["final_score"] = final_score
    
    return final_score, breakdown


def strict_filter_records(records: List[PaperRecord], query: str) -> List[PaperRecord]:
    """Filter records to only include those with keywords in title or abstract."""
    keywords = extract_keywords(query)
    if not keywords:
        return records

    filtered = []
    for record in records:
        title_lower = record.title.lower()
        abstract_lower = record.abstract.lower()
        
        # Check if at least 50% of keywords appear in title+abstract
        keyword_count = sum(1 for kw in keywords if kw in title_lower or kw in abstract_lower)
        if keyword_count >= len(keywords) * 0.5:
            filtered.append(record)

    return filtered


def sort_by_relevance(records: List[PaperRecord]) -> List[PaperRecord]:
    """Sort records by relevance score in descending order."""
    return sorted(records, key=lambda r: r.relevance_score, reverse=True)


def normalize_record(record: PaperRecord) -> PaperRecord:
    record.title = " ".join(record.title.split())
    record.authors = [a.strip() for a in record.authors if a and a.strip()]
    if not record.authors:
        record.authors = ["Unknown author"]
    record.abstract = " ".join(record.abstract.split()) if record.abstract else ""
    return record


def deduplicate_records(records: Sequence[PaperRecord]) -> List[PaperRecord]:
    seen = set()
    unique: List[PaperRecord] = []
    for record in records:
        record = normalize_record(record)
        fp = record.fingerprint()
        if fp in seen:
            continue
        seen.add(fp)
        unique.append(record)
    return unique


def get_search_log_path(base_dir: Path) -> Path:
    """Get the path for the search log file."""
    log_dir = base_dir / ".search_logs"
    ensure_dir(log_dir)
    return log_dir / "search_history.jsonl"


def log_search(base_dir: Path, query: str, source: str, results_count: int, strict_match: bool, min_relevance: float, filters: Dict[str, Any]):
    """Log the search to a JSONL file."""
    log_path = get_search_log_path(base_dir)
    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "query": query,
        "source": source,
        "results_count": results_count,
        "strict_match": strict_match,
        "min_relevance": min_relevance,
        "filters": filters,
    }
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as exc:
        print(f"Warning: Could not write to search log: {exc}")


def get_search_history(base_dir: Path, limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve recent searches from log."""
    log_path = get_search_log_path(base_dir)
    if not log_path.exists():
        return []
    
    history = []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        history.append(entry)
                    except json.JSONDecodeError:
                        continue
    except Exception as exc:
        print(f"Warning: Could not read search log: {exc}")
    
    return history[-limit:]


def print_search_history(base_dir: Path):
    """Print recent search history."""
    history = get_search_history(base_dir)
    if not history:
        print("No search history available.")
        return
    
    print("\n=== Recent Search History ===")
    for i, entry in enumerate(history[-10:], start=1):
        print(f"\n[{i}] {entry.get('timestamp', 'Unknown')}")
        print(f"    Query: {entry.get('query', 'N/A')}")
        print(f"    Source: {entry.get('source', 'N/A')}")
        print(f"    Results: {entry.get('results_count', 0)}")
        print(f"    Strict Match: {entry.get('strict_match', False)}")
        print(f"    Min Relevance: {entry.get('min_relevance', 0.0):.0%}")


def save_metadata_json(record: PaperRecord, file_path: Path):
    payload = {
        "title": record.title,
        "authors": record.authors,
        "abstract": record.abstract,
        "source": record.source,
        "year": record.year,
        "url": record.url,
        "pdf_url": record.pdf_url,
        "category": record.category,
        "doi": record.doi,
        "venue": record.venue,
        "relevance_score": record.relevance_score,
        "relevance_breakdown": record.relevance_breakdown,
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "raw_metadata": record.raw_metadata,
    }
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_text_fallback(record: PaperRecord, file_path: Path):
    text = (
        f"Title: {record.title}\n"
        f"Authors: {', '.join(record.authors)}\n"
        f"Source: {record.source}\n"
        f"Year: {record.year}\n"
        f"URL: {record.url}\n"
        f"PDF URL: {record.pdf_url or 'Not available'}\n"
        f"DOI: {record.doi or 'Not available'}\n"
        f"Venue: {record.venue or 'Not available'}\n"
        f"Relevance Score: {record.relevance_score:.2%}\n"
        f"Relevance Breakdown: {json.dumps(record.relevance_breakdown, indent=2)}\n\n"
        f"Abstract:\n{record.abstract or 'No abstract available.'}\n"
    )
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)


def get_source_folder(base_dir: Path, source: str, year: str) -> Path:
    source_name = source.lower().replace(" ", "_").replace("/", "_")
    folder = base_dir / source_name / year.strip() if year and year.isdigit() else base_dir / source_name / "unknown"
    ensure_dir(folder)
    return folder


def store_record(record: PaperRecord, base_dir: Path) -> Path:
    record = normalize_record(record)
    year_dir = get_source_folder(base_dir, record.source, record.year if record.year.isdigit() else "unknown")

    safe_base = sanitize_filename(record.title)
    pdf_path = year_dir / f"{safe_base}.pdf"
    meta_path = year_dir / f"{safe_base}.json"
    txt_path = year_dir / f"{safe_base}.txt"

    if record.pdf_url:
        success = download_file(record.pdf_url, pdf_path)
        if success:
            save_metadata_json(record, meta_path)
            return pdf_path

    save_text_fallback(record, txt_path)
    save_metadata_json(record, meta_path)
    return txt_path


def parse_arxiv(query: str, limit: int = 10) -> List[PaperRecord]:
    encoded = quote_plus(query)
    url = f"https://export.arxiv.org/api/query?search_query=all:{encoded}&start=0&max_results={limit}"
    xml = fetch_text(url)
    if not xml:
        return []
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        records: List[PaperRecord] = []
        for entry in root.findall("a:entry", ns)[:limit]:
            title = entry.findtext("a:title", default="", namespaces=ns).replace("\\n", " ").strip()
            authors = []
            for author in entry.findall("a:author", ns):
                name = author.findtext("a:name", default="", namespaces=ns)
                if name:
                    authors.append(name.strip())
            summary = entry.findtext("a:summary", default="", namespaces=ns).strip()
            published = entry.findtext("a:published", default="", namespaces=ns)
            pdf_url = ""
            url_value = ""
            for link in entry.findall("a:link", ns):
                href = link.get("href", "")
                rel = link.get("rel")
                title_attr = link.get("title")
                if title_attr == "pdf":
                    pdf_url = href
                if rel == "alternate":
                    url_value = href
            year = published[:4] if published else "Unknown"
            
            record = PaperRecord(
                title=title or "Untitled",
                authors=authors or ["Unknown author"],
                abstract=summary,
                source="arXiv",
                year=year,
                url=url_value or pdf_url,
                pdf_url=pdf_url,
                category="arXiv",
                doi="",
                venue="arXiv",
            )
            record.relevance_score, record.relevance_breakdown = calculate_relevance_score(record, query)
            records.append(record)
        return records
    except Exception as exc:
        print(f"Failed to parse arXiv: {exc}")
        return []


def parse_pubmed_search_json(data: Dict[str, Any], query: str, limit: int) -> List[PaperRecord]:
    records: List[PaperRecord] = []
    ids = data.get("esearchresult", {}).get("idlist", [])[:limit]
    for pmid in ids:
        try:
            summary_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={pmid}&retmode=json"
            summary = fetch_text(summary_url)
            if not summary:
                continue
            sdata = json.loads(summary)
            doc = sdata.get("result", {}).get(str(pmid), {})
            title = doc.get("title", "Untitled")
            authors = [a.get("name", "") for a in doc.get("authors", []) if a.get("name")]
            abstract = doc.get("abstract", "")
            pubdate = doc.get("pubdate", "")
            year = pubdate[:4] if isinstance(pubdate, str) and len(pubdate) >= 4 else "Unknown"
            
            record = PaperRecord(
                title=title,
                authors=authors or ["Unknown author"],
                abstract=abstract,
                source="PubMed",
                year=year,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                pdf_url="",
                category="PubMed",
                doi=doc.get("articleids", [{}])[0].get("value", "") if doc.get("articleids") else "",
                venue=doc.get("fulljournalname", ""),
                raw_metadata=doc,
            )
            record.relevance_score, record.relevance_breakdown = calculate_relevance_score(record, query)
            records.append(record)
        except Exception as exc:
            print(f"PubMed item {pmid} failed: {exc}")
    return records


def search_pubmed(query: str, limit: int = 10) -> List[PaperRecord]:
    encoded = quote_plus(query)
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmode=json&retmax={limit}&term={encoded}"
    text = fetch_text(url)
    if not text:
        return []
    try:
        data = json.loads(text)
        return parse_pubmed_search_json(data, query, limit)
    except Exception as exc:
        print(f"PubMed search failed: {exc}")
        return []


def parse_biorxiv_html(html: str, query: str, limit: int) -> List[PaperRecord]:
    """Improved bioRxiv HTML parsing with better metadata extraction."""
    soup = BeautifulSoup(html, "html.parser")
    results: List[PaperRecord] = []
    seen: set[str] = set()

    # Target biorxiv article result containers
    articles = soup.select("div.highwire-article-container") or soup.select("article.result") or soup.select("div[data-test*='result']")
    
    for article_elem in articles[:limit * 2]:
        try:
            # Extract title and URL
            title_elem = article_elem.select_one("h2 a, h3 a, a[href*='/content/']")
            if not title_elem:
                continue
            
            title = " ".join(title_elem.get_text(" ", strip=True).split())
            href = title_elem.get("href", "")
            
            if not href or href in seen or len(title) < 10:
                continue
            seen.add(href)

            # Extract authors with better parsing
            authors = []
            author_elems = article_elem.select("span.nlm-person-name, span.contrib-author, a[rel='author']")
            for auth in author_elems[:10]:
                name = auth.get_text(strip=True)
                if name and len(name) > 2 and "@" not in name:
                    authors.append(name)

            # Extract abstract
            abstract = ""
            abstract_elem = article_elem.select_one("p.summary, p.abstract, div.summary")
            if abstract_elem:
                abstract = " ".join(abstract_elem.get_text(" ", strip=True).split())[:500]

            # Extract publication date and year
            year = "Unknown"
            date_elem = article_elem.select_one("span.published-date, time, span.date, div.published")
            if date_elem:
                date_text = date_elem.get_text(strip=True)
                year_match = re.search(r"(20\d{2})", date_text)
                if year_match:
                    year = year_match.group(1)

            # Extract DOI if available
            doi = ""
            doi_elem = article_elem.select_one("a[href*='doi.org']")
            if doi_elem:
                doi_href = doi_elem.get("href", "")
                doi_match = re.search(r"10\.\d{4,9}/[^/\s]+", doi_href)
                if doi_match:
                    doi = doi_match.group(0)

            full_url = f"https://www.biorxiv.org{href}" if not href.startswith("http") else href
            
            record = PaperRecord(
                title=title[:250],
                authors=authors or ["Unknown author"],
                abstract=abstract,
                source="bioRxiv",
                year=year,
                url=full_url,
                pdf_url=f"{full_url}.full.pdf" if "/content/" in full_url else "",
                category="bioRxiv",
                doi=doi,
                venue="bioRxiv",
            )
            record.relevance_score, record.relevance_breakdown = calculate_relevance_score(record, query)
            results.append(record)
            
            if len(results) >= limit:
                break
        except Exception as e:
            continue

    return results


def search_biorxiv(query: str, limit: int = 10) -> List[PaperRecord]:
    encoded = quote_plus(query)
    url = f"https://www.biorxiv.org/search/{encoded}"
    html = fetch_text(url)
    if not html:
        return []
    return parse_biorxiv_html(html, query, limit)


def parse_medrxiv_html(html: str, query: str, limit: int) -> List[PaperRecord]:
    """Improved medRxiv HTML parsing with better metadata extraction."""
    soup = BeautifulSoup(html, "html.parser")
    results: List[PaperRecord] = []
    seen: set[str] = set()

    articles = soup.select("div.highwire-article-container") or soup.select("article.result") or soup.select("div[data-test*='result']")
    
    for article_elem in articles[:limit * 2]:
        try:
            title_elem = article_elem.select_one("h2 a, h3 a, a[href*='/content/']")
            if not title_elem:
                continue
            
            title = " ".join(title_elem.get_text(" ", strip=True).split())
            href = title_elem.get("href", "")
            
            if not href or href in seen or len(title) < 10:
                continue
            seen.add(href)

            authors = []
            author_elems = article_elem.select("span.nlm-person-name, span.contrib-author, a[rel='author']")
            for auth in author_elems[:10]:
                name = auth.get_text(strip=True)
                if name and len(name) > 2 and "@" not in name:
                    authors.append(name)

            abstract = ""
            abstract_elem = article_elem.select_one("p.summary, p.abstract, div.summary")
            if abstract_elem:
                abstract = " ".join(abstract_elem.get_text(" ", strip=True).split())[:500]

            year = "Unknown"
            date_elem = article_elem.select_one("span.published-date, time, span.date, div.published")
            if date_elem:
                date_text = date_elem.get_text(strip=True)
                year_match = re.search(r"(20\d{2})", date_text)
                if year_match:
                    year = year_match.group(1)

            doi = ""
            doi_elem = article_elem.select_one("a[href*='doi.org']")
            if doi_elem:
                doi_href = doi_elem.get("href", "")
                doi_match = re.search(r"10\.\d{4,9}/[^/\s]+", doi_href)
                if doi_match:
                    doi = doi_match.group(0)

            full_url = f"https://www.medrxiv.org{href}" if not href.startswith("http") else href
            
            record = PaperRecord(
                title=title[:250],
                authors=authors or ["Unknown author"],
                abstract=abstract,
                source="medRxiv",
                year=year,
                url=full_url,
                pdf_url=f"{full_url}.full.pdf" if "/content/" in full_url else "",
                category="medRxiv",
                doi=doi,
                venue="medRxiv",
            )
            record.relevance_score, record.relevance_breakdown = calculate_relevance_score(record, query)
            results.append(record)
            
            if len(results) >= limit:
                break
        except Exception:
            continue

    return results


def search_medrxiv(query: str, limit: int = 10) -> List[PaperRecord]:
    encoded = quote_plus(query)
    url = f"https://www.medrxiv.org/search/{encoded}"
    html = fetch_text(url)
    if not html:
        return []
    return parse_medrxiv_html(html, query, limit)


def parse_ieee_html(html: str, query: str, limit: int) -> List[PaperRecord]:
    """Improved IEEE Xplore HTML parsing with better metadata extraction."""
    soup = BeautifulSoup(html, "html.parser")
    records: List[PaperRecord] = []
    seen: set[str] = set()

    # IEEE typically uses specific result row classes
    result_items = soup.select("div.result-item, article[data-test*='result'], tr[data-test*='result']")
    if not result_items:
        result_items = soup.select("a[href*='document/']")

    for item in result_items[:limit * 5]:
        try:
            # Find title link
            title_link = item.select_one("a") if item.name == "a" else item.select_one("a[href*='document/']")
            if not title_link:
                continue

            href = title_link.get("href", "")
            title = " ".join(title_link.get_text(" ", strip=True).split())

            if not href or href in seen or len(title) < 10:
                continue
            seen.add(href)

            # Extract authors with better parsing
            authors = []
            author_container = item.select("span.author, a.author, .contrib-author, span[data-test*='author']")
            for auth_elem in author_container[:10]:
                auth_text = auth_elem.get_text(strip=True)
                if auth_text and len(auth_text) > 2 and not auth_text.endswith(";"):
                    authors.append(auth_text.rstrip(";"))

            # Extract abstract
            abstract = ""
            abstract_elem = item.select_one("p.abstract, span.abstract, div[class*='abstract']")
            if abstract_elem:
                abstract = " ".join(abstract_elem.get_text(" ", strip=True).split())[:500]

            # Extract year and publication date
            year = "Unknown"
            date_elem = item.select_one("span.publish-date, span.date, time, div.date")
            if date_elem:
                date_text = date_elem.get_text(strip=True)
                year_match = re.search(r"(20\d{2})", date_text)
                if year_match:
                    year = year_match.group(1)

            # Extract DOI
            doi = ""
            doi_elem = item.select_one("a[href*='doi.org']")
            if doi_elem:
                doi_text = doi_elem.get_text(strip=True)
                doi_match = re.search(r"10\.\d{4,9}/[^/\s]+", doi_text)
                if doi_match:
                    doi = doi_match.group(0)

            # Extract venue/conference
            venue = ""
            venue_elem = item.select_one("span.conference, span.journal, div.publication")
            if venue_elem:
                venue = venue_elem.get_text(strip=True)[:100]

            full_url = f"https://ieeexplore.ieee.org{href}" if not href.startswith("http") else href

            record = PaperRecord(
                title=title[:250],
                authors=authors or ["Unknown author"],
                abstract=abstract,
                source="IEEE Xplore",
                year=year,
                url=full_url,
                pdf_url="",  # IEEE requires authentication for PDFs
                category="IEEE Xplore",
                doi=doi,
                venue=venue or "IEEE Xplore",
            )
            record.relevance_score, record.relevance_breakdown = calculate_relevance_score(record, query)
            records.append(record)

            if len(records) >= limit:
                break
        except Exception:
            continue

    return records


def search_ieee(query: str, limit: int = 10) -> List[PaperRecord]:
    encoded = quote_plus(query)
    url = f"https://ieeexplore.ieee.org/search/searchresult.jsp?queryText={encoded}&rowsPerPage={limit}"
    html = fetch_text(url)
    if not html:
        return []
    return parse_ieee_html(html, query, limit)


def parse_springer_html(html: str, query: str, limit: int) -> List[PaperRecord]:
    """Improved SpringerLink HTML parsing with better metadata extraction."""
    soup = BeautifulSoup(html, "html.parser")
    records: List[PaperRecord] = []
    seen: set[str] = set()

    result_items = soup.select("article[data-test*='result'], div.result-item, li[data-test*='result']")
    if not result_items:
        result_items = soup.select("a[href*='/article/']")

    for item in result_items[:limit * 5]:
        try:
            title_link = item.select_one("a") if item.name == "a" else item.select_one("a[href*='/article/']")
            if not title_link:
                continue

            href = title_link.get("href", "")
            title = " ".join(title_link.get_text(" ", strip=True).split())

            if not href or href in seen or len(title) < 10:
                continue
            seen.add(href)

            # Extract authors
            authors = []
            author_elems = item.select("a[data-test*='author'], span.author, a[rel='author']")
            for auth_elem in author_elems[:10]:
                auth_text = auth_elem.get_text(strip=True)
                if auth_text and len(auth_text) > 2:
                    authors.append(auth_text)

            # Extract abstract
            abstract = ""
            abstract_elem = item.select_one("p.abstract, p.summary, div[class*='abstract']")
            if abstract_elem:
                abstract = " ".join(abstract_elem.get_text(" ", strip=True).split())[:500]

            # Extract year
            year = "Unknown"
            date_elem = item.select_one("span.date, time, span[class*='date']")
            if date_elem:
                date_text = date_elem.get_text(strip=True)
                year_match = re.search(r"(20\d{2})", date_text)
                if year_match:
                    year = year_match.group(1)

            # Extract DOI
            doi = ""
            doi_elem = item.select_one("a[href*='doi.org']")
            if doi_elem:
                doi_text = doi_elem.get_text(strip=True)
                doi_match = re.search(r"10\.\d{4,9}/[^/\s]+", doi_text)
                if doi_match:
                    doi = doi_match.group(0)

            # Extract journal/venue
            venue = ""
            venue_elem = item.select_one("span.journal, div.journal, p.journal")
            if venue_elem:
                venue = venue_elem.get_text(strip=True)[:100]

            full_url = f"https://link.springer.com{href}" if not href.startswith("http") else href

            record = PaperRecord(
                title=title[:250],
                authors=authors or ["Unknown author"],
                abstract=abstract,
                source="SpringerLink",
                year=year,
                url=full_url,
                pdf_url="",  # Springer requires authentication
                category="SpringerLink",
                doi=doi,
                venue=venue or "SpringerLink",
            )
            record.relevance_score, record.relevance_breakdown = calculate_relevance_score(record, query)
            records.append(record)

            if len(records) >= limit:
                break
        except Exception:
            continue

    return records


def search_springer(query: str, limit: int = 10) -> List[PaperRecord]:
    encoded = quote_plus(query)
    url = f"https://link.springer.com/search?query={encoded}"
    html = fetch_text(url)
    if not html:
        return []
    return parse_springer_html(html, query, limit)


def parse_sciencedirect_html(html: str, query: str, limit: int) -> List[PaperRecord]:
    """Improved ScienceDirect HTML parsing with better metadata extraction."""
    soup = BeautifulSoup(html, "html.parser")
    records: List[PaperRecord] = []
    seen: set[str] = set()

    result_items = soup.select("article[data-test*='result'], div.result-item")
    if not result_items:
        result_items = soup.select("a[href*='/science/article/']")

    for item in result_items[:limit * 5]:
        try:
            title_link = item.select_one("a") if item.name == "a" else item.select_one("a[href*='/science/article/']")
            if not title_link:
                continue

            href = title_link.get("href", "")
            title = " ".join(title_link.get_text(" ", strip=True).split())

            if not href or href in seen or len(title) < 10:
                continue
            seen.add(href)

            # Extract authors
            authors = []
            author_elems = item.select("span.author, a.author, div[data-test*='author']")
            for auth_elem in author_elems[:10]:
                auth_text = auth_elem.get_text(strip=True)
                if auth_text and len(auth_text) > 2:
                    authors.append(auth_text)

            # Extract abstract
            abstract = ""
            abstract_elem = item.select_one("p.abstract, div.abstract, span[class*='abstract']")
            if abstract_elem:
                abstract = " ".join(abstract_elem.get_text(" ", strip=True).split())[:500]

            # Extract year
            year = "Unknown"
            date_elem = item.select_one("span.date, time, span.publication-date")
            if date_elem:
                date_text = date_elem.get_text(strip=True)
                year_match = re.search(r"(20\d{2})", date_text)
                if year_match:
                    year = year_match.group(1)

            # Extract DOI
            doi = ""
            doi_elem = item.select_one("a[href*='doi.org'], span.doi")
            if doi_elem:
                doi_text = doi_elem.get_text(strip=True)
                doi_match = re.search(r"10\.\d{4,9}/[^/\s]+", doi_text)
                if doi_match:
                    doi = doi_match.group(0)

            # Extract journal
            venue = ""
            venue_elem = item.select_one("span.journal, div.journal, p.journal")
            if venue_elem:
                venue = venue_elem.get_text(strip=True)[:100]

            full_url = f"https://www.sciencedirect.com{href}" if not href.startswith("http") else href

            record = PaperRecord(
                title=title[:250],
                authors=authors or ["Unknown author"],
                abstract=abstract,
                source="ScienceDirect",
                year=year,
                url=full_url,
                pdf_url="",  # ScienceDirect requires authentication
                category="ScienceDirect",
                doi=doi,
                venue=venue or "ScienceDirect",
            )
            record.relevance_score, record.relevance_breakdown = calculate_relevance_score(record, query)
            records.append(record)

            if len(records) >= limit:
                break
        except Exception:
            continue

    return records


def search_sciencedirect(query: str, limit: int = 10) -> List[PaperRecord]:
    encoded = quote_plus(query)
    url = f"https://www.sciencedirect.com/search?qs={encoded}&show=100"
    html = fetch_text(url)
    if not html:
        return []
    return parse_sciencedirect_html(html, query, limit)


def search_google_scholar(query: str, limit: int = 10) -> List[PaperRecord]:
    if scholarly is None:
        print("scholarly is not installed. Install with: pip install scholarly")
        return []
    try:
        results = []
        for item in scholarly.search_pubs(query):
            if len(results) >= limit:
                break
            title = item.get("title", "Untitled")
            authors = item.get("author", []) or ["Unknown author"]
            abstract = item.get("abstract", "No abstract available.")
            year = str(item.get("year", "Unknown"))
            url = item.get("pub_url", "")
            pdf_url = item.get("eprint_url", "") or ""
            
            record = PaperRecord(
                title=title,
                authors=authors,
                abstract=abstract,
                source="Google Scholar",
                year=year,
                url=url,
                pdf_url=pdf_url,
                category="Google Scholar",
                doi=item.get("doi", ""),
                venue=item.get("venue", "") or "Google Scholar",
                raw_metadata=item,
            )
            record.relevance_score, record.relevance_breakdown = calculate_relevance_score(record, query)
            results.append(record)
        return results
    except Exception as exc:
        print(f"Google Scholar failed: {exc}")
        return []


def search_openalex(query: str, limit: int = 10) -> List[PaperRecord]:
    encoded = quote_plus(query)
    url = f"https://api.openalex.org/works?search={encoded}&per-page={limit}&select=id,display_name,publication_year,authors,primary_location,doi,url,abstract_inverted_index,concepts"
    text = fetch_text(url)
    if not text:
        return []
    try:
        data = json.loads(text)
        records: List[PaperRecord] = []
        for item in data.get("results", [])[:limit]:
            title = item.get("display_name", "Untitled")
            authors = [a.get("author", {}).get("display_name", "") for a in item.get("authors", []) if a.get("author", {}).get("display_name")]
            abstract = decode_inverted_index(item.get("abstract_inverted_index"))
            year = str(item.get("publication_year", "Unknown"))
            primary = item.get("primary_location", {})
            venue = primary.get("source", {}).get("display_name", "") or "OpenAlex"
            url_value = item.get("primary_location", {}).get("landing_page_url", "") or item.get("url", "")
            doi = item.get("doi", "").replace("https://doi.org/", "") if item.get("doi") else ""
            
            record = PaperRecord(
                title=title,
                authors=authors or ["Unknown author"],
                abstract=abstract,
                source="OpenAlex",
                year=year,
                url=url_value,
                pdf_url="",
                category="OpenAlex",
                doi=doi,
                venue=venue,
                raw_metadata=item,
            )
            record.relevance_score, record.relevance_breakdown = calculate_relevance_score(record, query)
            records.append(record)
        return records
    except Exception as exc:
        print(f"OpenAlex search failed: {exc}")
        return []


def search_semantic_scholar(query: str, limit: int = 10) -> List[PaperRecord]:
    encoded = quote_plus(query)
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded}&limit={limit}&fields=title,abstract,authors,year,url,externalIds,venue"
    text = fetch_text(url)
    if not text:
        return []
    try:
        data = json.loads(text)
        records: List[PaperRecord] = []
        for item in data.get("data", [])[:limit]:
            authors = [a.get("name", "") for a in item.get("authors", []) if a.get("name")]
            abstract = item.get("abstract", "")
            year = str(item.get("year", "Unknown"))
            url_value = item.get("url", "")
            doi = item.get("externalIds", {}).get("DOI", "") if item.get("externalIds") else ""
            venue = item.get("venue", "") or "Semantic Scholar"
            
            record = PaperRecord(
                title=item.get("title", "Untitled"),
                authors=authors or ["Unknown author"],
                abstract=abstract,
                source="Semantic Scholar",
                year=year,
                url=url_value,
                pdf_url="",
                category="Semantic Scholar",
                doi=doi,
                venue=venue,
                raw_metadata=item,
            )
            record.relevance_score, record.relevance_breakdown = calculate_relevance_score(record, query)
            records.append(record)
        return records
    except Exception as exc:
        print(f"Semantic Scholar failed: {exc}")
        return []


def search_source(source: str, query: str, limit: int = 10) -> List[PaperRecord]:
    source_key = source.lower().replace(" ", "_")
    if source_key == "arxiv":
        return parse_arxiv(query, limit)
    if source_key == "pubmed":
        return search_pubmed(query, limit)
    if source_key == "biorxiv":
        return search_biorxiv(query, limit)
    if source_key == "medrxiv":
        return search_medrxiv(query, limit)
    if source_key == "ieee":
        return search_ieee(query, limit)
    if source_key == "springer":
        return search_springer(query, limit)
    if source_key == "sciencedirect":
        return search_sciencedirect(query, limit)
    if source_key in ("google_scholar", "scholar"):
        return search_google_scholar(query, limit)
    if source_key == "openalex":
        return search_openalex(query, limit)
    if source_key == "semantic_scholar":
        return search_semantic_scholar(query, limit)
    return []


def search_all(query: str, limit: int = 10) -> List[PaperRecord]:
    sources = [
        ("arXiv", "arxiv"),
        ("bioRxiv", "biorxiv"),
        ("medRxiv", "medrxiv"),
        ("PubMed", "pubmed"),
        ("Google Scholar", "google_scholar"),
        ("OpenAlex", "openalex"),
        ("Semantic Scholar", "semantic_scholar"),
        ("IEEE Xplore", "ieee"),
        ("SpringerLink", "springer"),
        ("ScienceDirect", "sciencedirect"),
    ]
    results: List[PaperRecord] = []
    for _, key in sources:
        try:
            results.extend(search_source(key, query, limit))
        except Exception as exc:
            print(f"Search failed in source {key}: {exc}")
    return results


def filter_records(
    records: Sequence[PaperRecord],
    author: Optional[str] = None,
    title: Optional[str] = None,
    category: Optional[str] = None,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
) -> List[PaperRecord]:
    filtered: List[PaperRecord] = []
    for r in records:
        author_text = " ".join(r.authors).lower()
        title_text = r.title.lower()
        category_text = r.category.lower()

        if author and author.lower() not in author_text:
            continue
        if title and title.lower() not in title_text:
            continue
        if category and category.lower() not in category_text:
            continue

        try:
            year_value = int(r.year)
        except ValueError:
            year_value = None

        if year_start is not None and (year_value is None or year_value < year_start):
            continue
        if year_end is not None and (year_value is None or year_value > year_end):
            continue

        filtered.append(r)
    return filtered


def load_queries(batch_file: str) -> List[str]:
    try:
        with open(batch_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as exc:
        print(f"Unable to read batch file: {exc}")
        return []


def print_results(records: Sequence[PaperRecord]) -> None:
    if not records:
        print("No papers found.")
        return

    for i, record in enumerate(records, start=1):
        print(f"\n[{i}] {record.title}")
        print(f"Authors: {', '.join(record.authors) if record.authors else 'Unknown'}")
        print(f"Source: {record.source} | Year: {record.year} | Venue: {record.category}")
        print(f"Relevance Score: {record.relevance_score:.1%} | Title: {record.relevance_breakdown.get('title_score', 0):.1%} | Abstract: {record.relevance_breakdown.get('abstract_score', 0):.1%}")
        print(f"URL: {record.url or 'Not available'}")
        print(f"PDF URL: {record.pdf_url or 'Not available'}")
        if record.doi:
            print(f"DOI: {record.doi}")
        if record.abstract:
            print(f"Abstract: {record.abstract[:350]}..." if len(record.abstract) > 350 else f"Abstract: {record.abstract}")


def metadata_from_pdf(pdf_path: Path) -> Dict[str, Any]:
    data: Dict[str, Any] = {"title": "", "authors": [], "abstract": "", "year": "", "doi": "", "venue": ""}
    if not pdf_path.exists():
        return data

    if PyPDF2 is not None:
        try:
            reader = PyPDF2.PdfReader(str(pdf_path))
            info = reader.metadata or {}
            data["title"] = str(info.get("/Title", "") or "")
            data["authors"] = [x.strip() for x in str(info.get("/Author", "") or "").split(";") if x.strip()]
            text = "\n".join(page.extract_text() or "" for page in list(reader.pages)[:20])
            if not data["title"]:
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                if lines:
                    data["title"] = lines[0][:200]
            if not data["abstract"]:
                match = re.search(r"(?is)abstract\s*[:\-]?\s*(.*?)(?:\n\s*(?:keywords?|introduction|1\.)\b)", text)
                if match:
                    data["abstract"] = match.group(1).strip()
                else:
                    m2 = re.search(r"(?is)abstract\s*[:\-]?\s*(.+)", text)
                    if m2:
                        data["abstract"] = m2.group(1).strip()
            if not data["year"]:
                year_match = re.search(r"(19|20)\d{2}", text)
                if year_match:
                    data["year"] = year_match.group(0)
            doi_match = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", text, re.I)
            if doi_match:
                data["doi"] = doi_match.group(0)
        except Exception as exc:
            print(f"PDF metadata parse failed for {pdf_path}: {exc}")

    if (not data["title"] or not data["abstract"]) and pdfplumber is not None:
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                chunk = "\n".join(page.extract_text() or "" for page in pdf.pages[:10])
                if not data["title"]:
                    line = chunk.splitlines()[0].strip() if chunk.splitlines() else ""
                    data["title"] = line[:200]
                if not data["abstract"]:
                    match = re.search(r"(?is)abstract\s*[:\-]?\s*(.*?)(?:\n\s*(?:keywords?|introduction|1\.)\b)", chunk)
                    if match:
                        data["abstract"] = match.group(1).strip()
        except Exception:
            pass

    return data


def parse_pdf_and_update(record: PaperRecord, pdf_path: Path):
    meta = metadata_from_pdf(pdf_path)
    if meta.get("title"):
        record.title = meta["title"]
    if meta.get("abstract"):
        record.abstract = meta["abstract"]
    if meta.get("doi"):
        record.doi = meta["doi"]
    if meta.get("year"):
        record.year = meta["year"]
    if meta.get("authors"):
        record.authors = meta["authors"]


def main():
    parser = argparse.ArgumentParser(description="Download research papers from multiple sources with advanced relevance filtering and search logging.")
    parser.add_argument("--query", help="Search query. Example: 'transformer medical imaging'")
    parser.add_argument("--batch-file", help="Text file with one query per line.")
    parser.add_argument("--source", choices=[
        "all", "arxiv", "biorxiv", "medrxiv", "pubmed", "ieee", "springer", "sciencedirect",
        "google_scholar", "openalex", "semantic_scholar", "scholar"
    ], default="all")
    parser.add_argument("--max-results", type=int, default=10)
    parser.add_argument("--download", action="store_true", help="Download PDFs when available.")
    parser.add_argument("--storage-dir", help="Exact directory to save results. If omitted, prompt user.")
    parser.add_argument("--author", default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--category", default=None)
    parser.add_argument("--year-start", type=int, default=None)
    parser.add_argument("--year-end", type=int, default=None)
    parser.add_argument("--strict-match", action="store_true", help="Only keep papers with keywords in title or abstract.")
    parser.add_argument("--min-relevance", type=float, default=0.0, help="Minimum relevance score (0.0-1.0) to include results.")
    parser.add_argument("--show-history", action="store_true", help="Show recent search history and exit.")
    args = parser.parse_args()

    if args.storage_dir:
        base_dir = Path(args.storage_dir).expanduser()
    else:
        base_dir = prompt_for_storage_dir()

    ensure_dir(base_dir)

    # Handle history display
    if args.show_history:
        print_search_history(base_dir)
        return

    queries: List[str] = []
    if args.batch_file:
        queries = load_queries(args.batch_file)
    elif args.query:
        queries = [args.query]
    else:
        print("Please provide either --query or --batch-file.")
        sys.exit(1)

    all_records: List[PaperRecord] = []
    for query in queries:
        print(f"\nSearching for: '{query}'")
        if args.source == "all":
            records = search_all(query, limit=args.max_results)
        else:
            records = search_source(args.source, query, limit=args.max_results)
        
        # Apply strict matching if requested
        if args.strict_match:
            print(f"  Applying strict keyword matching...")
            records = strict_filter_records(records, query)
        
        # Filter by relevance score
        if args.min_relevance > 0.0:
            records = [r for r in records if r.relevance_score >= args.min_relevance]
            print(f"  Filtered to relevance score >= {args.min_relevance:.0%}")
        
        # Apply other filters
        filtered = filter_records(
            records,
            author=args.author,
            title=args.title,
            category=args.category,
            year_start=args.year_start,
            year_end=args.year_end,
        )
        all_records.extend(filtered)
        
        # Log search
        log_search(
            base_dir,
            query=query,
            source=args.source,
            results_count=len(filtered),
            strict_match=args.strict_match,
            min_relevance=args.min_relevance,
            filters={
                "author": args.author,
                "title": args.title,
                "category": args.category,
                "year_start": args.year_start,
                "year_end": args.year_end,
            }
        )

    unique_records = deduplicate_records(all_records)
    
    # Sort by relevance score
    unique_records = sort_by_relevance(unique_records)
    
    print(f"\nFound {len(unique_records)} unique papers")
    print_results(unique_records)

    if args.download:
        print("\nDownloading and organizing results...")
        for record in unique_records:
            saved_path = store_record(record, base_dir)
            if saved_path.suffix.lower() == ".pdf":
                try:
                    parse_pdf_and_update(record, saved_path)
                    save_metadata_json(record, saved_path.with_suffix(".json"))
                except Exception as exc:
                    print(f"Could not update metadata for {saved_path}: {exc}")
            print(f"Saved: {saved_path}")

    print(f"\nStorage directory: {base_dir}")
    print(f"Search history: {get_search_log_path(base_dir)}")
    print(f"Tip: Run with --show-history to view recent searches")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(0)
