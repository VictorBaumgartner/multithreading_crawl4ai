from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
import asyncio
import os
import json
import re
from urllib.parse import urljoin, urlparse
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from typing import List, Dict, Any, Tuple
import csv
import io
from concurrent.futures import ThreadPoolExecutor
import uvicorn
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"crawl_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

EXCLUDE_KEYWORDS = []  # Relaxed to allow all URLs

class CrawlCSVRequest(BaseModel):
    """Request model for crawling URLs from a CSV."""
    output_dir: str = "./crawl_output_csv"
    max_concurrency_per_site: int = Field(default=5, ge=1, le=50, description="Maximum concurrent requests *per site being crawled* (max 50).")
    max_depth: int = Field(default=2, ge=0, description="Maximum depth to crawl from each starting URL in the CSV.")
    max_threads: int = Field(default=10, ge=1, le=100, description="Maximum number of concurrent sites to crawl (max 100).")

def clean_markdown(md_text: str) -> str:
    """Cleans Markdown content by removing or modifying specific elements."""
    md_text = re.sub(r'!\[([^\]]*)\]\((http[s]?://[^\)]+)\)', '', md_text)
    md_text = re.sub(r'\[([^\]]+)\]\((http[s]?://[^\)]+)\)', r'\1', md_text)
    md_text = re.sub(r'(?<!\]\()https?://\S+', '', md_text)
    md_text = re.sub(r'\[\^?\d+\]', '', md_text)
    md_text = re.sub(r'^\[\^?\d+\]:\s?.*$', '', md_text, flags=re.MULTILINE)
    md_text = re.sub(r'^\s{0,3}>\s?', '', md_text, flags=re.MULTILINE)
    md_text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', md_text)
    md_text = re.sub(r'(\*|_)(.*?)\1', r'\2', md_text)
    md_text = re.sub(r'^\s*#+\s*$', '', md_text, flags=re.MULTILINE)
    md_text = re.sub(r'\(\)', '', md_text)
    md_text = re.sub(r'\n\s*\n+', '\n\n', md_text)
    md_text = re.sub(r'[ \t]+', ' ', md_text)
    return md_text.strip()

def read_urls_from_csv(csv_content: str) -> List[str]:
    """Reads URLs from CSV content string. Assumes one URL per line."""
    urls = []
    csvfile = io.StringIO(csv_content)
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if not row:
            continue
        url = row[0].strip()
        if url and (url.startswith("http://") or url.startswith("https://")):
            try:
                parsed_url = urlparse(url)
                if parsed_url.netloc:
                    urls.append(url)
                else:
                    logger.warning(f"Skipping URL with no recognizable domain on line {i+1}: '{row[0]}'")
            except Exception:
                logger.warning(f"Skipping invalid URL format on line {i+1}: '{row[0]}'")
        else:
            logger.warning(f"Skipping non-HTTP/HTTPS or empty entry on line {i+1}: '{row[0]}'")
    return urls

def sanitize_filename(url: str) -> str:
    """Sanitizes a URL to create a safe and shorter filename."""
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc.replace(".", "_")
        path = parsed.path.strip("/").replace("/", "_").replace(".", "_")
        if not path:
            path = "index"

        query = parsed.query
        if query:
            query = query[:50]
            query = query.replace("=", "-").replace("&", "_")
            filename = f"{netloc}_{path}_{query}"
        else:
            filename = f"{netloc}_{path}"

        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        filename = re.sub(r'[\s\._-]+', '_', filename)
        filename = re.sub(r'^_+', '', filename)
        filename = re.sub(r'_+$', '', filename)

        if not filename:
            filename = f"url_{abs(hash(url))}"

        max_len_without_suffix = 150 - 3
        filename = filename[:max_len_without_suffix] + ".md"
        return filename
    except Exception as e:
        logger.error(f"Error sanitizing URL filename for {url}: {e}")
        return f"error_parsing_{abs(hash(url))}.md"

def sanitize_dirname(url: str) -> str:
    """Sanitizes a URL's domain to create a safe directory name."""
    try:
        parsed = urlparse(url)
        dirname = parsed.netloc.replace(".", "_")
        dirname = re.sub(r'[<>:"/\\|?*]', '_', dirname)
        dirname = re.sub(r'[\s\._-]+', '_', dirname)
        dirname = re.sub(r'^_+', '', dirname)
        dirname = re.sub(r'_+$', '', dirname)

        if not dirname:
            dirname = f"domain_{abs(hash(url))}"

        return dirname[:150]
    except Exception as e:
        logger.error(f"Error sanitizing URL directory name for {url}: {e}")
        return f"domain_error_{abs(hash(url))}"

def read_file_info_list(file_path: str) -> Dict[str, str]:
    """Reads file_info_list.csv to get the last crawled URL for each start_url."""
    file_info = {}
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    start_url = row.get("start_url", "").strip()
                    last_crawled_url = row.get("last_crawled_url", "").strip()
                    if start_url:
                        file_info[start_url] = last_crawled_url
        except Exception as e:
            logger.error(f"Error reading file_info_list.csv: {e}")
    return file_info

def write_file_info_list(file_path: str, file_info: Dict[str, str]):
    """Writes the file_info_list.csv with start_url and last_crawled_url."""
    try:
        with open(file_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["start_url", "last_crawled_url"])
            writer.writeheader()
            for start_url, last_crawled_url in file_info.items():
                writer.writerow({"start_url": start_url, "last_crawled_url": last_crawled_url})
        logger.info(f"Updated file_info_list.csv at {file_path}")
    except Exception as e:
        logger.error(f"Error writing file_info_list.csv: {e}")

CrawlQueueItem = Tuple[str, int, str, str]

async def crawl_website_single_site(
    start_url: str,
    output_dir: str,
    max_concurrency: int,
    max_depth: int,
    global_crawled_urls: set
) -> Dict[str, Any]:
    """
    Crawl a single website deeply and save each page as a cleaned Markdown file
    in a site-specific subdirectory, with parallelization. Resume from last_crawled_url.
    """
    crawled_urls = set()
    queued_urls = set()
    crawl_queue: asyncio.Queue[CrawlQueueItem] = asyncio.Queue()
    semaphore = asyncio.Semaphore(max_concurrency)  # Limit concurrency per site
    results = {"success": [], "failed": [], "skipped_by_filter": [], "initial_url": start_url}
    last_crawled_url = ""

    try:
        parsed_start_url = urlparse(start_url)
        start_domain = parsed_start_url.netloc
        if not start_domain:
            results["failed"].append({"url": start_url, "error": "Could not extract domain from start URL"})
            logger.error(f"Error: Could not extract domain from start URL: {start_url}")
            return results

        site_subdir_name = sanitize_dirname(start_url)
        site_output_path = os.path.join(output_dir, site_subdir_name)
        site_output_path = os.path.abspath(site_output_path)
        logger.info(f"Crawl limited to domain: {start_domain}")
        logger.info(f"Saving files for this site in: {site_output_path}")

        try:
            os.makedirs(site_output_path, exist_ok=True)
            if not os.path.exists(site_output_path):
                raise OSError(f"Failed to create directory: {site_output_path}")
        except Exception as e:
            results["failed"].append({"url": start_url, "error": f"Cannot create output directory: {e}"})
            logger.error(f"Error creating output directory {site_output_path}: {e}")
            return results

    except Exception as e:
        results["failed"].append({"url": start_url, "error": f"Error parsing start URL or determining output path: {e}"})
        logger.error(f"Error processing start URL {start_url} or determining output path: {e}")
        return results

    # Check file_info_list.csv for last_crawled_url
    file_info_path = os.path.join(output_dir, "file_info_list.csv")
    file_info = read_file_info_list(file_info_path)
    initial_url = file_info.get(start_url, start_url)  # Use last_crawled_url or start_url
    crawl_queue.put_nowait((initial_url, 0, start_domain, site_output_path))
    queued_urls.add(initial_url)
    logger.info(f"Starting crawl for {start_url} from {initial_url} with max_depth={max_depth}, max_concurrency={max_concurrency}")

    md_generator = DefaultMarkdownGenerator(
        options={
            "ignore_links": True,
            "escape_html": True,
            "body_width": 0
        }
    )

    config = CrawlerRunConfig(
        markdown_generator=md_generator,
        cache_mode="BYPASS",
        exclude_social_media_links=True,
        check_robots_txt=True
    )

    async def crawl_page():
        """Worker function to process URLs from the queue."""
        nonlocal last_crawled_url
        while not crawl_queue.empty():
            try:
                current_url, current_depth, crawl_start_domain, current_site_output_path = await crawl_queue.get()

                if current_url in crawled_urls or current_url in global_crawled_urls:
                    crawl_queue.task_done()
                    continue

                try:
                    current_domain = urlparse(current_url).netloc
                    if current_domain != crawl_start_domain:
                        logger.info(f"Skipping external URL: {current_url} (Domain: {current_domain}, Expected: {crawl_start_domain})")
                        crawled_urls.add(current_url)
                        crawl_queue.task_done()
                        continue
                except Exception as e:
                    logger.error(f"Error parsing domain for URL {current_url}: {e}. Skipping.")
                    crawled_urls.add(current_url)
                    crawl_queue.task_done()
                    continue

                crawled_urls.add(current_url)
                last_crawled_url = current_url  # Track the last crawled URL
                logger.info(f"Crawling ({len(crawled_urls)}): {current_url} (Depth: {current_depth})")

                filename = sanitize_filename(current_url)
                output_path = os.path.join(current_site_output_path, filename)

                async with semaphore:
                    async with AsyncWebCrawler(verbose=True) as crawler:
                        result = await crawler.arun(url=current_url, config=config)
                    await asyncio.sleep(1.0)  # Manual delay

                if result.success:
                    cleaned_markdown = clean_markdown(result.markdown.raw_markdown)
                    if not cleaned_markdown.strip():
                        logger.warning(f"No content to save for {current_url}")
                        results["failed"].append({"url": current_url, "error": "Empty Markdown content"})
                    else:
                        try:
                            if not os.access(current_site_output_path, os.W_OK):
                                raise OSError(f"No write permission for directory: {current_site_output_path}")
                            with open(output_path, "w", encoding="utf-8") as f:
                                f.write(f"# {current_url}\n\n{cleaned_markdown}\n")
                            if os.path.exists(output_path):
                                logger.info(f"Saved cleaned Markdown to: {output_path}")
                                results["success"].append(current_url)
                            else:
                                raise IOError(f"File was not created: {output_path}")
                        except Exception as e:
                            logger.error(f"Error saving file {output_path}: {e}")
                            results["failed"].append({"url": current_url, "error": f"File save error: {e}"})

                    if current_depth < max_depth:
                        internal_links = result.links.get("internal", [])
                        for link in internal_links:
                            href = link["href"]
                            try:
                                absolute_url = urljoin(current_url, href)
                                parsed_absolute_url = urlparse(absolute_url)
                                if parsed_absolute_url.netloc == crawl_start_domain:
                                    if absolute_url not in crawled_urls and absolute_url not in queued_urls and absolute_url not in global_crawled_urls:
                                        crawl_queue.put_nowait((absolute_url, current_depth + 1, crawl_start_domain, current_site_output_path))
                                        queued_urls.add(absolute_url)
                            except Exception as link_e:
                                logger.error(f"Error processing link {href} from {current_url}: {link_e}")
                else:
                    logger.error(f"Failed to crawl {current_url}: {result.error_message}")
                    results["failed"].append({"url": current_url, "error": result.error_message})

                crawl_queue.task_done()
            except Exception as e:
                logger.error(f"Error in crawl_page worker: {e}")
                crawl_queue.task_done()

        # Save progress for this site
        progress_path = os.path.join(output_dir, f"progress_{sanitize_dirname(start_url)}.json")
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump({"crawled_urls": list(crawled_urls), "results": results}, f, indent=2)
        logger.info(f"Progress saved for {start_url} to {progress_path}")

        # Update file_info_list.csv with last_crawled_url
        file_info[start_url] = last_crawled_url or start_url
        write_file_info_list(file_info_path, file_info)

    worker_tasks = []
    for _ in range(max_concurrency):
        task = asyncio.create_task(crawl_page())
        worker_tasks.append(task)

    await crawl_queue.join()

    for task in worker_tasks:
        task.cancel()

    await asyncio.gather(*worker_tasks, return_exceptions=True)

    logger.info(f"Finished crawl for: {start_url}")
    return results

def run_crawl_in_thread(start_url: str, output_dir: str, max_concurrency: int, max_depth: int, global_crawled_urls: set) -> Dict[str, Any]:
    """Run crawl_website_single_site in a separate thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(
            crawl_website_single_site(start_url, output_dir, max_concurrency, max_depth, global_crawled_urls)
        )
        return result
    except Exception as e:
        logger.error(f"Thread error for {start_url}: {e}")
        return {"status": "error", "initial_url": start_url, "error": str(e)}
    finally:
        loop.close()

@app.post("/crawl_csv_upload")
async def crawl_csv_upload_endpoint(
    csv_file: UploadFile = File(...),
    output_dir: str = Form("./crawl_output_csv"),
    max_concurrency_per_site: int = Form(default=5, ge=1),
    max_depth: int = Form(default=2, ge=0),
    max_threads: int = Form(default=10, ge=1)
):
    """
    FastAPI endpoint to crawl URLs provided in an uploaded CSV file using multithreading.
    Resumes from last_crawled_url in file_info_list.csv.
    """
    if not csv_file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    try:
        csv_content = await csv_file.read()
        csv_content = csv_content.decode("utf-8")
        urls_to_crawl = read_urls_from_csv(csv_content)

        if not urls_to_crawl:
            return {"status": "warning", "message": "No valid URLs found in the CSV file to crawl."}

        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Collect all previously crawled URLs from progress files
        global_crawled_urls = set()
        progress_files = [f for f in os.listdir(output_dir) if f.startswith("progress_") and f.endswith(".json")]
        for pf in progress_files:
            try:
                with open(os.path.join(output_dir, pf), "r", encoding="utf-8") as f:
                    progress = json.load(f)
                    crawled = progress.get("crawled_urls", [])
                    global_crawled_urls.update(crawled)
            except Exception as e:
                logger.error(f"Error reading progress file {pf}: {e}")

        # Skip sites where crawling is complete (all URLs crawled successfully)
        completed_urls = set()
        for pf in progress_files:
            with open(os.path.join(output_dir, pf), "r", encoding="utf-8") as f:
                progress = json.load(f)
                results = progress.get("results", {})
                if results.get("success") and not results.get("failed") and not results.get("skipped_by_filter"):
                    completed_urls.add(results.get("initial_url", ""))

        urls_to_crawl = [url for url in urls_to_crawl if url not in completed_urls]
        logger.info(f"Found {len(completed_urls)} completed URLs, {len(urls_to_crawl)} URLs remaining to crawl")

        overall_results: Dict[str, Any] = {
            "status": "processing",
            "total_urls_from_csv": len(urls_to_crawl),
            "site_crawl_results": {}
        }

        # Use ThreadPoolExecutor to crawl sites in parallel
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            # Submit each site to a thread
            futures = [
                executor.submit(
                    run_crawl_in_thread,
                    url,
                    output_dir,
                    max_concurrency_per_site,
                    max_depth,
                    global_crawled_urls
                ) for url in urls_to_crawl
            ]

            # Collect results
            for future, url in zip(futures, urls_to_crawl):
                try:
                    result = future.result()
                    overall_results["site_crawl_results"][url] = result
                except Exception as e:
                    logger.error(f"Error crawling {url}: {e}")
                    overall_results["site_crawl_results"][url] = {
                        "status": "error",
                        "initial_url": url,
                        "error": str(e)
                    }

        # Save overall metadata
        metadata_path = os.path.join(output_dir, "overall_metadata.json")
        try:
            serializable_results = overall_results.copy()
            for url, res in serializable_results["site_crawl_results"].items():
                if "success" in res and isinstance(res["success"], set):
                    res["success"] = list(res["success"])
                if "skipped_by_filter" in res and isinstance(res["skipped_by_filter"], set):
                    res["skipped_by_filter"] = list(res["skipped_by_filter"])

            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(serializable_results, f, indent=2)
            overall_results["metadata_path"] = metadata_path
            logger.info(f"Overall metadata saved to {metadata_path}")
        except Exception as e:
            logger.error(f"Error saving overall metadata: {e}")
            overall_results["metadata_save_error"] = str(e)

        overall_results["status"] = "completed"
        logger.info("Overall CSV processing completed")
        return overall_results
    except Exception as e:
        logger.error(f"Critical error in crawl_csv_upload_endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred during request processing: {str(e)}")

if __name__ == "__main__":
    print("Starting FastAPI application...")
    print("Navigate to http://localhost:8001/docs for interactive documentation (Swagger UI).")
    uvicorn.run(app, host="0.0.0.0", port=8001)