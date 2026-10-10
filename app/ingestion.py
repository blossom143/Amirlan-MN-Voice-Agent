"""
Khan Bank Automated Data Ingestion Pipeline (Scrape & Structuring)
===================================================================
Automated crawler, parser, and document normalizer for Khan Bank knowledge base.
Crawls official Khan Bank portal pages (Personal & Business banking), converts
HTML/text/PDF content into clean structured Markdown with explicit YAML frontmatter.
"""

import os
import re
import sys
import json
import time
import argparse
from typing import Dict, List, Optional, Any
from datetime import datetime

# Optional HTTP and parsing libraries
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False


# Category heuristic mappings from URL path / title
CATEGORY_RULES = {
    "deposit": ("deposit_inquiry", "individual"),
    "saving": ("deposit_inquiry", "individual"),
    "хадгаламж": ("deposit_inquiry", "individual"),
    "loan": ("loan_inquiry", "individual"),
    "mortgage": ("loan_inquiry", "individual"),
    "зээл": ("loan_inquiry", "individual"),
    "ипотек": ("loan_inquiry", "individual"),
    "card": ("card_inquiry", "individual"),
    "карт": ("card_inquiry", "individual"),
    "digital": ("digital_banking", "individual"),
    "app": ("digital_banking", "individual"),
    "цахим": ("digital_banking", "individual"),
    "fee": ("fees_and_tariffs", "all"),
    "tariff": ("fees_and_tariffs", "all"),
    "шимтгэл": ("fees_and_tariffs", "all"),
    "branch": ("branch_and_atm", "all"),
    "atm": ("branch_and_atm", "all"),
    "салбар": ("branch_and_atm", "all"),
    "remittance": ("remittance_and_forex", "all"),
    "swift": ("remittance_and_forex", "all"),
    "гуйвуулга": ("remittance_and_forex", "all"),
    "валют": ("remittance_and_forex", "all"),
    "contact": ("contact_and_support", "all"),
    "лавлах": ("contact_and_support", "all"),
    "business": ("business_banking", "business"),
    "байгууллага": ("business_banking", "business")
}


class KhanBankIngestionWorker:
    """
    Automated data ingestion worker for Khan Bank website and documents.
    """

    def __init__(self, output_dir: Optional[str] = None):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.kb_dir = output_dir or os.path.join(self.base_dir, "knowledge_base")
        os.makedirs(self.kb_dir, exist_ok=True)
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "mn,en-US,en;q=0.9"
        }

    def infer_metadata(self, url: str, title: str, text: str) -> Dict[str, str]:
        """Infers category and target audience from URL path, title, and body keywords."""
        haystack = f"{url.lower()} {title.lower()} {text[:500].lower()}"
        category = "general_inquiry"
        audience = "individual"

        for key, (cat, aud) in CATEGORY_RULES.items():
            if key in haystack:
                category = cat
                audience = aud
                break

        return {
            "category": category,
            "target_audience": audience,
            "last_updated": datetime.now().strftime("%Y-%m-%d")
        }

    def clean_html_to_markdown(self, html_content: str, source_url: str) -> Dict[str, Any]:
        """
        Strips boilerplate navigations, ads, scripts and converts main article into structured Markdown.
        """
        title = "Хаан Банкны албан ёсны мэдээлэл"
        clean_text = ""

        if BS4_AVAILABLE:
            soup = BeautifulSoup(html_content, "html.parser")
            
            # Extract title
            title_tag = soup.find("h1") or soup.find("title")
            if title_tag:
                title = title_tag.get_text().strip()

            # Remove unwanted tags
            for s in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
                s.decompose()

            # Preserve structured headers and tables
            body_parts = []
            main_node = soup.find("main") or soup.find("article") or soup.body or soup

            for elem in main_node.descendants:
                if elem.name in ["h1", "h2", "h3"]:
                    heading_text = elem.get_text().strip()
                    if heading_text:
                        body_parts.append(f"\n\n### [SECTION] {heading_text}\n")
                elif elem.name == "p":
                    p_text = elem.get_text().strip()
                    if p_text and len(p_text) > 10:
                        body_parts.append(f"{p_text}\n")
                elif elem.name in ["ul", "ol"]:
                    items = [li.get_text().strip() for li in elem.find_all("li") if li.get_text().strip()]
                    if items:
                        list_str = "\n".join([f"- {item}" for item in items])
                        body_parts.append(f"\n{list_str}\n")
                elif elem.name == "table":
                    # Convert table to markdown table format
                    rows = elem.find_all("tr")
                    table_md = []
                    for row_idx, r in enumerate(rows):
                        cols = [c.get_text().strip().replace("\n", " ") for c in r.find_all(["th", "td"])]
                        if cols:
                            table_md.append("| " + " | ".join(cols) + " |")
                            if row_idx == 0:
                                table_md.append("| " + " | ".join(["---"] * len(cols)) + " |")
                    if table_md:
                        body_parts.append("\n" + "\n".join(table_md) + "\n")

            clean_text = "".join(body_parts)
        else:
            # Fallback regex-based cleaning
            clean_text = re.sub(r"<script.*?</script>", "", html_content, flags=re.DOTALL)
            clean_text = re.sub(r"<style.*?</style>", "", clean_text, flags=re.DOTALL)
            clean_text = re.sub(r"<[^>]+>", " ", clean_text)
            clean_text = re.sub(r"\s+", " ", clean_text).strip()

        meta = self.infer_metadata(source_url, title, clean_text)

        return {
            "title": title,
            "url": source_url,
            "category": meta["category"],
            "last_updated": meta["last_updated"],
            "target_audience": meta["target_audience"],
            "content": clean_text.strip()
        }

    def format_document(self, doc_data: Dict[str, Any]) -> str:
        """
        Formats normalized document with explicit YAML frontmatter and structured headers.
        """
        yaml_frontmatter = (
            f"---\n"
            f'title: "{doc_data.get("title", "Мэдээлэл")}"\n'
            f'url: "{doc_data.get("url", "https://www.khanbank.com")}"\n'
            f'category: "{doc_data.get("category", "general_inquiry")}"\n'
            f'last_updated: "{doc_data.get("last_updated", "2026-10-01")}"\n'
            f'target_audience: "{doc_data.get("target_audience", "individual")}"\n'
            f"---\n\n"
        )
        body = f"# {doc_data.get('title', '')}\n\n{doc_data.get('content', '')}\n"
        return yaml_frontmatter + body

    def save_normalized_document(self, filename: str, doc_data: Dict[str, Any]) -> str:
        """Saves normalized markdown file into knowledge base directory."""
        formatted_md = self.format_document(doc_data)
        out_path = os.path.join(self.kb_dir, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(formatted_md)
        return out_path

    def crawl_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Fetches a single page and returns normalized document representation."""
        if not REQUESTS_AVAILABLE:
            print("requests library not installed.")
            return None

        try:
            resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code == 200:
                return self.clean_html_to_markdown(resp.text, url)
            else:
                print(f"Failed to fetch {url}: HTTP {resp.status_code}")
                return None
        except Exception as e:
            print(f"Exception fetching {url}: {e}")
            return None

    def list_existing_documents(self) -> List[str]:
        """Lists all existing normalized knowledge base documents."""
        if not os.path.exists(self.kb_dir):
            return []
        return sorted([
            f for f in os.listdir(self.kb_dir)
            if f.endswith(".md") or f.endswith(".json")
        ])


def main():
    parser = argparse.ArgumentParser(description="Khan Bank Automated Ingestion Pipeline")
    parser.add_argument("--crawl", type=str, help="Single URL to crawl and ingest")
    parser.add_argument("--list", action="store_true", help="List all indexed knowledge files")
    args = parser.parse_args()

    worker = KhanBankIngestionWorker()

    if args.list:
        files = worker.list_existing_documents()
        print(f"Found {len(files)} knowledge files in {worker.kb_dir}:")
        for f in files:
            print(f" - {f}")
    elif args.crawl:
        print(f"Crawling {args.crawl}...")
        doc = worker.crawl_url(args.crawl)
        if doc:
            filename = re.sub(r"[^\w\-]", "_", doc["title"][:30]) + ".md"
            saved = worker.save_normalized_document(filename, doc)
            print(f"Saved: {saved}")
        else:
            print("Crawl failed.")
    else:
        print("Usage: python ingestion.py --list OR python ingestion.py --crawl <url>")


if __name__ == "__main__":
    main()
