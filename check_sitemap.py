import httpx
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from typing import List, Optional, Dict
import xml.etree.ElementTree as ET
import gzip
import io
import csv
import re
import asyncio
from datetime import datetime
import os # <-- NEW: Import the os module

# --- Start Crawl4AI Specific Imports ---
from crawl4ai import AsyncWebCrawler
# In Crawl4AI 0.6.x, CrawlerRunConfig and PlaywrightConfig are often not directly imported classes.
# Instead, configuration parameters are passed directly or as a dict.
# We will define a config dictionary directly.
# --- End Crawl4AI Specific Imports ---

app = FastAPI()

# User-Agent to identify our scraper
HEADERS = {"User-Agent": "SitemapScraperWithCrawl4AI/1.0 (https://yourwebsite.com)"}
SITEMAP_ROBOTS_REGEX = re.compile(r"Sitemap:\s*(.*)")
MAX_SITEMAP_DEPTH = 5

# --- Crawl4AI specific configuration (as a dictionary) ---
CRAWL4AI_DEFAULT_CONFIG = {
    "headless": True, # For Playwright browser. 'headless' is typically accepted as a direct arg in arun() config dict for v0.6.3
    "page_timeout": 30000, # 30 seconds timeout for page loading
    "user_agent": HEADERS["User-Agent"], # Set user agent
    "accept_downloads": False,
    "capture_console_logs": False,
    "capture_network_traffic": False,
    "ignore_pages": True, # Essential for sitemaps! Don't process them as regular pages.
}

# --- Re-introduce the XML parsing logic from the original solution ---
async def parse_xml_content_for_sitemap_entries(xml_content: bytes, base_url: str, depth: int) -> List[Dict]:
    """
    Parses a sitemap XML content and extracts URLs along with their lastmod dates if available.
    Handles sitemap index files recursively.
    Returns a list of dictionaries: [{"url": "...", "lastmod": "..."}, ...]
    """
    entries = []
    try:
        root = ET.fromstring(xml_content)
        sitemap_ns = "{http://www.sitemaps.org/schemas/sitemap/0.9}" # XML Namespace

        if "sitemapindex" in root.tag:
            if depth >= MAX_SITEMAP_DEPTH:
                print(f"Max sitemap recursion depth reached for {base_url}")
                return entries
            sitemap_urls = []
            for sitemap in root.findall(f"{sitemap_ns}sitemap"):
                loc = sitemap.find(f"{sitemap_ns}loc")
                if loc is not None and loc.text:
                    sitemap_urls.append(loc.text)

            # Concurrently fetch and parse nested sitemaps
            tasks = [
                fetch_and_parse_sitemap_content(sitemap_url, depth + 1)
                for sitemap_url in sitemap_urls
            ]
            results = await asyncio.gather(*tasks)
            for result_entries in results:
                entries.extend(result_entries)

        elif "urlset" in root.tag:
            for url_elem in root.findall(f"{sitemap_ns}url"):
                loc_elem = url_elem.find(f"{sitemap_ns}loc")
                lastmod_elem = url_elem.find(f"{sitemap_ns}lastmod")

                url = loc_elem.text if loc_elem is not None and loc_elem.text else None
                lastmod = lastmod_elem.text if lastmod_elem is not None and lastmod_elem.text else ""

                if url:
                    entries.append({"url": url, "lastmod": lastmod})
        else:
            print(f"Unknown sitemap root tag: {root.tag} for {base_url}")

    except ET.ParseError as e:
        print(f"XML parsing error for {base_url}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during sitemap XML parsing for {base_url}: {e}")
    return entries


async def fetch_and_parse_sitemap_content(sitemap_url: str, depth: int = 0) -> List[Dict]:
    """
    Fetches raw sitemap content using Crawl4AI, handles gzip, and then parses it with ElementTree.
    """
    print(f"Crawl4AI: Attempting to fetch content for sitemap: {sitemap_url}")
    content = None
    response_content_type = None # To store the Content-Type header

    try:
        async with AsyncWebCrawler() as crawler:
            response = await crawler.arun(
                url=sitemap_url,
                **CRAWL4AI_DEFAULT_CONFIG, # Pass config dict
            )

            if response.success:
                # Get Content-Type header from response
                response_content_type = response.headers.get('Content-Type', '').lower()

                if response.html: # Crawl4AI typically puts content into .html if it detects HTML
                    content = response.html.encode('utf-8')
                elif response.markdown: # In case it was interpreted as text/markdown
                    content = response.markdown.encode('utf-8')
                elif response.raw_content: # Fallback to raw_content if available
                    content = response.raw_content
            else:
                print(f"Crawl4AI: Failed to fetch content for {sitemap_url}: {response.error_message or 'No content or error'}")
                return []

    except Exception as e:
        print(f"Crawl4AI: Error fetching {sitemap_url}: {e}")
        return []

    if not content:
        print(f"Crawl4AI: No content received for {sitemap_url}")
        return []

    # --- NEW LOGIC FOR CONTENT TYPE HANDLING ---
    # Heuristics for content type based on headers and content itself

    # Check if content type is clearly HTML
    if 'text/html' in response_content_type or content.strip().startswith(b'<!DOCTYPE html>') or content.strip().startswith(b'<html'):
        print(f"Crawl4AI: Received HTML content for {sitemap_url}, not an XML sitemap. Skipping parsing.")
        return []

    # Check for gzip signature if it's a .gz URL or header suggests it
    if sitemap_url.endswith(".gz") or sitemap_url.endswith(".xml.gz") or 'application/x-gzip' in response_content_type or 'application/gzip' in response_content_type:
        try:
            # Check for gzip magic number (first two bytes: 0x1f 0x8b)
            if content.startswith(b'\x1f\x8b'):
                decompressed_content = gzip.decompress(content)
                print(f"Successfully decompressed gzipped content for {sitemap_url}.")
                return await parse_xml_content_for_sitemap_entries(decompressed_content, sitemap_url, depth)
            else:
                print(f"Content for {sitemap_url} (expected gzip) does not start with gzip signature. Trying as plain XML.")
        except OSError as e:
            print(f"Failed to decompress gzipped content for {sitemap_url}: {e}. Trying as plain XML.")
        except Exception as e:
            print(f"An error occurred during gzip decompression for {sitemap_url}: {e}. Trying as plain XML.")

    # Fallback: Always try parsing as plain XML
    print(f"Attempting to parse {sitemap_url} as plain XML.")
    return await parse_xml_content_for_sitemap_entries(content, sitemap_url, depth)


async def get_sitemaps_from_robots_txt_crawl4ai(domain_url: str) -> List[str]:
    """
    Attempts to find sitemap URLs in a website's robots.txt using Crawl4AI.
    """
    robots_txt_url = f"{domain_url.rstrip('/')}/robots.txt"
    print(f"Crawl4AI: Checking robots.txt: {robots_txt_url}")
    try:
        async with AsyncWebCrawler() as crawler:
            response = await crawler.arun(
                url=robots_txt_url,
                **CRAWL4AI_DEFAULT_CONFIG # Pass config dictionary using **
            )
            if response.success and response.html:
                content = response.html.encode('utf-8')
                sitemap_urls = []
                for line in content.decode('utf-8').splitlines():
                    match = SITEMAP_ROBOTS_REGEX.match(line)
                    if match:
                        sitemap_urls.append(match.group(1).strip())
                return sitemap_urls
            else:
                print(f"Crawl4AI: Failed to fetch or no content in robots.txt for {domain_url}: {response.error_message}")
                return []
    except Exception as e:
        print(f"Crawl4AI Error parsing robots.txt for {domain_url}: {e}")
        return []


async def find_and_parse_sitemaps_crawl4ai(domain_url: str) -> List[dict]:
    """
    Orchestrates the process of finding and parsing sitemaps for a given domain using Crawl4AI for fetching.
    Returns a list of dictionaries: [{"url": "...", "lastmod": "..."}, ...]
    """
    all_entries = {}

    potential_sitemap_urls = [
        f"{domain_url.rstrip('/')}/sitemap.xml",
        f"{domain_url.rstrip('/')}/sitemap_index.xml",
        f"{domain_url.rstrip('/')}/sitemap.xml.gz",
        f"{domain_url.rstrip('/')}/sitemap_index.xml.gz",
    ]

    # New: Add the sitemap URL from robots.txt only if found
    robots_sitemaps = await get_sitemaps_from_robots_txt_crawl4ai(domain_url)
    potential_sitemap_urls.extend(robots_sitemaps)

    # Remove duplicates from the list of potential sitemap URLs
    potential_sitemap_urls = list(set(potential_sitemap_urls))

    tasks = [fetch_and_parse_sitemap_content(url) for url in potential_sitemap_urls]
    results = await asyncio.gather(*tasks)

    for entries_found in results:
        for entry in entries_found:
            all_entries[entry["url"]] = entry["lastmod"]

    return [{"url": url, "lastmod": lastmod} for url, lastmod in all_entries.items()]


@app.post("/get-sitemaps-csv-crawl4ai/")
async def get_sitemaps_as_csv_crawl4ai(site_urls: List[str]):
    """
    FastAPI endpoint to retrieve all unique links and their lastmod dates from sitemaps
    for provided URLs using Crawl4AI (for fetching) and ElementTree (for parsing).
    Returns them in a CSV format AND saves a copy to the server's CWD.
    """
    if not site_urls:
        raise HTTPException(status_code=400, detail="No URLs provided.")

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Domain", "URL", "Last Modified"])

    all_results = []
    for url in site_urls:
        original_domain = url
        if not url.startswith("http://") and not url.startswith("https://"):
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
            print(f"Failed to process {url} with Crawl4AI: {e}")
            all_results.append({
                "domain": original_domain,
                "url": f"ERROR (Crawl4AI): {e}",
                "lastmod": ""
            })

    for row_data in all_results:
        writer.writerow([row_data["domain"], row_data["url"], row_data["lastmod"]])

    output.seek(0) # Rewind the in-memory buffer

    # --- NEW ADDITION: Save to CWD on the server ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"sitemap_links_with_dates_{timestamp}.csv"
    cwd = os.getcwd() # Get the server's current working directory
    file_path = os.path.join(cwd, output_filename)

    try:
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            f.write(output.getvalue())
        print(f"Sitemap results successfully saved to {file_path} on the server.")
    except Exception as e:
        print(f"ERROR: Could not save CSV file to CWD: {e}")
    # --- END NEW ADDITION ---

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=sitemap_links_with_dates_crawl4ai.csv"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)