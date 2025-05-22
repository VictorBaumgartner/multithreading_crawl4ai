import httpx
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from typing import List, Dict
import xml.etree.ElementTree as ET
import gzip
import io
import csv
import re
import asyncio
from datetime import datetime
import os
from urllib.parse import urlparse, urljoin
from crawl4ai import AsyncWebCrawler

app = FastAPI()

# User-Agent to identify our scraper
HEADERS = {"User-Agent": "SitemapScraperWithCrawl4AI/1.0 (https://example.com)"}
SITEMAP_ROBOTS_REGEX = re.compile(r"Sitemap:\s*(.*?)(?:\n|$)", re.IGNORECASE)
MAX_SITEMAP_DEPTH = 5
TIMEOUT_SECONDS = 30

CRAWL4AI_DEFAULT_CONFIG = {
    "headless": True,
    "page_timeout": 30000,
    "user_agent": HEADERS["User-Agent"],
    "accept_downloads": False,
    "capture_console_logs": False,
    "capture_network_traffic": False,
    "ignore_pages": True,
}

async def is_valid_url(url: str) -> bool:
    """Validate if a URL is properly formatted."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

async def parse_xml_content_for_sitemap_entries(xml_content: bytes, base_url: str, depth: int) -> List[Dict]:
    """Parses a sitemap XML content and extracts URLs along with their lastmod dates."""
    entries = []
    try:
        root = ET.fromstring(xml_content)
        sitemap_ns = "{http://www.sitemaps.org/schemas/sitemap/0.9}"

        if "sitemapindex" in root.tag:
            if depth >= MAX_SITEMAP_DEPTH:
                print(f"Max sitemap recursion depth reached for {base_url}")
                return entries
            sitemap_urls = []
            for sitemap in root.findall(f"{sitemap_ns}sitemap"):
                loc = sitemap.find(f"{sitemap_ns}loc")
                if loc is not None and loc.text and await is_valid_url(loc.text):
                    sitemap_urls.append(loc.text)

            tasks = [fetch_and_parse_sitemap_content(url, depth + 1) for url in sitemap_urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, list):
                    entries.extend(result)

        elif "urlset" in root.tag:
            for url_elem in root.findall(f"{sitemap_ns}url"):
                loc_elem = url_elem.find(f"{sitemap_ns}loc")
                lastmod_elem = url_elem.find(f"{sitemap_ns}lastmod")

                url = loc_elem.text if loc_elem is not None and loc_elem.text else None
                lastmod = lastmod_elem.text if lastmod_elem is not None and lastmod_elem.text else ""

                if url and await is_valid_url(url):
                    entries.append({"url": url, "lastmod": lastmod})
        else:
            print(f"Unknown sitemap root tag: {root.tag} for {base_url}")

    except ET.ParseError as e:
        print(f"XML parsing error for {base_url}: {e}")
    except Exception as e:
        print(f"Unexpected error during sitemap XML parsing for {base_url}: {e}")
    return entries

async def fetch_and_parse_sitemap_content(sitemap_url: str, depth: int = 0) -> List[Dict]:
    """Fetches and parses sitemap content, handling various content types."""
    print(f"Fetching sitemap: {sitemap_url}")
    content = None
    content_type = None

    try:
        async with AsyncWebCrawler() as crawler:
            response = await crawler.arun(
                url=sitemap_url,
                **CRAWL4AI_DEFAULT_CONFIG,
                timeout=TIMEOUT_SECONDS * 1000
            )

            if response.success:
                content_type = response.headers.get('Content-Type', '').lower()
                content = response.raw_content or response.html.encode('utf-8') if response.html else None

            if not content:
                print(f"No content received for {sitemap_url}")
                return []

        # Handle different content types
        if 'text/html' in content_type or content.strip().startswith(b'<!DOCTYPE html>') or content.strip().startswith(b'<html'):
            print(f"Received HTML content for {sitemap_url}, not an XML sitemap")
            return []

        # Handle gzip compression
        if (sitemap_url.endswith(('.gz', '.xml.gz')) or 
            'application/x-gzip' in content_type or 
            'application/gzip' in content_type or 
            content.startswith(b'\x1f\x8b')):
            try:
                content = gzip.decompress(content)
                print(f"Decompressed gzipped content for {sitemap_url}")
            except Exception as e:
                print(f"Failed to decompress gzip content for {sitemap_url}: {e}")
                return []

        return await parse_xml_content_for_sitemap_entries(content, sitemap_url, depth)

    except Exception as e:
        print(f"Error fetching {sitemap_url}: {e}")
        return []

async def get_sitemaps_from_robots_txt_crawl4ai(domain_url: str) -> List[str]:
    """Extracts sitemap URLs from robots.txt."""
    robots_urls = [
        f"{domain_url.rstrip('/')}/robots.txt",
        f"{domain_url.rstrip('/')}/robot.txt"  # Some sites use non-standard names
    ]
    sitemap_urls = []

    for robots_url in robots_urls:
        print(f"Checking robots.txt: {robots_url}")
        try:
            async with AsyncWebCrawler() as crawler:
                response = await crawler.arun(
                    url=robots_url,
                    **CRAWL4AI_DEFAULT_CONFIG,
                    timeout=TIMEOUT_SECONDS * 1000
                )
                if response.success and response.html:
                    content = response.html
                    for line in content.splitlines():
                        match = SITEMAP_ROBOTS_REGEX.match(line.strip())
                        if match:
                            sitemap_url = match.group(1).strip()
                            if await is_valid_url(sitemap_url):
                                sitemap_urls.append(sitemap_url)
                            else:
                                print(f"Invalid sitemap URL found in robots.txt: {sitemap_url}")
                else:
                    print(f"No content in robots.txt for {robots_url}")
        except Exception as e:
            print(f"Error fetching robots.txt {robots_url}: {e}")

    return list(set(sitemap_urls))

async def find_and_parse_sitemaps_crawl4ai(domain_url: str) -> List[dict]:
    """Orchestrates finding and parsing sitemaps for a domain."""
    all_entries = {}

    # Expanded list of common sitemap locations
    potential_sitemap_urls = [
        f"{domain_url.rstrip('/')}/sitemap.xml",
        f"{domain_url.rstrip('/')}/sitemap_index.xml",
        f"{domain_url.rstrip('/')}/sitemap.xml.gz",
        f"{domain_url.rstrip('/')}/sitemap_index.xml.gz",
        f"{domain_url.rstrip('/')}/sitemap/sitemap.xml",
        f"{domain_url.rstrip('/')}/sitemap/sitemap_index.xml",
        f"{domain_url.rstrip('/')}/sitemap.php",
        f"{domain_url.rstrip('/')}/sitemap.txt",
    ]

    # Add sitemaps from robots.txt
    robots_sitemaps = await get_sitemaps_from_robots_txt_crawl4ai(domain_url)
    potential_sitemap_urls.extend(robots_sitemaps)

    # Normalize and deduplicate URLs
    potential_sitemap_urls = list(set([urljoin(domain_url, url) for url in potential_sitemap_urls if await is_valid_url(urljoin(domain_url, url))]))

    tasks = [fetch_and_parse_sitemap_content(url) for url in potential_sitemap_urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, list):
            for entry in result:
                all_entries[entry["url"]] = entry["lastmod"]

    return [{"url": url, "lastmod": lastmod} for url, lastmod in all_entries.items()]

@app.post("/get-sitemaps-csv-crawl4ai/")
async def get_sitemaps_as_csv_crawl4ai(site_urls: List[str]):
    """FastAPI endpoint to retrieve sitemap links and lastmod dates as CSV."""
    if not site_urls:
        raise HTTPException(status_code=400, detail="No URLs provided.")

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Domain", "URL", "Last Modified"])

    all_results = []
    for url in site_urls:
        original_domain = url
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"

        try:
            entries = await find_and_parse_sitemaps_crawl4ai(url)
            if entries:
                for entry in entries:
                    all_results.append({
                        "domain": original_domain,
                        "url": entry["url"],
                        "lastmod": entry["lastmod"]
                    })
            else:
                all_results.append({
                    "domain": original_domain,
                    "url": "No sitemap or links found",
                    "lastmod": ""
                })
        except Exception as e:
            print(f"Failed to process {url}: {e}")
            all_results.append({
                "domain": original_domain,
                "url": f"ERROR: {str(e)}",
                "lastmod": ""
            })

    for row_data in all_results:
        writer.writerow([row_data["domain"], row_data["url"], row_data["lastmod"]])

    output.seek(0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"sitemap_links_with_dates_{timestamp}.csv"
    file_path = os.path.join(os.getcwd(), output_filename)

    try:
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            f.write(output.getvalue())
        print(f"Sitemap results saved to {file_path}")
    except Exception as e:
        print(f"ERROR: Could not save CSV file: {e}")

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={output_filename}"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)