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

app = FastAPI()

# Define the exclusion keywords for filenames (case-insensitive check will be used)
EXCLUDE_KEYWORDS = ['pdf', 'jpeg', 'jpg', 'png', 'webp']

class CrawlCSVRequest(BaseModel):
    """Request model for crawling URLs from a CSV."""
    output_dir: str = "./crawl_output_csv"
    max_concurrency_per_site: int = Field(default=8, ge=1, description="Maximum concurrent requests *per site being crawled*.")
    max_depth: int = Field(default=2, ge=0, description="Maximum depth to crawl from each starting URL in the CSV.")

def clean_markdown(md_text: str) -> str:
    """
    Cleans Markdown content by removing or modifying specific elements.
    """
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
                    print(f"Skipping URL with no recognizable domain on line {i+1}: '{row[0]}'")
            except Exception:
                print(f"Skipping invalid URL format on line {i+1}: '{row[0]}'")
        else:
            print(f"Skipping non-HTTP/HTTPS or empty entry on line {i+1}: '{row[0]}'")
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
        print(f"Error sanitizing URL filename for {url}: {e}")
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
        print(f"Error sanitizing URL directory name for {url}: {e}")
        return f"domain_error_{abs(hash(url))}"

CrawlQueueItem = Tuple[str, int, str, str]

def process_markdown_and_save(url: str, markdown_content: str, output_path: str) -> Dict[str, Any]:
    """Process Markdown content and save it to a file (executed in a thread)."""
    try:
        cleaned_markdown = clean_markdown(markdown_content)
        if not os.access(os.path.dirname(output_path), os.W_OK):
            raise OSError(f"No write permission for directory: {os.path.dirname(output_path)}")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# {url}\n\n{cleaned_markdown}\n")
        if os.path.exists(output_path):
            print(f"Saved cleaned Markdown to: {output_path}")
            return {"status": "success", "url": url}
        else:
            raise IOError(f"File was not created: {output_path}")
    except Exception as e:
        print(f"Error processing/saving {url}: {e}")
        return {"status": "failed", "url": url, "error": str(e)}

async def crawl_website_single_site(
    start_url: str,
    output_dir: str,
    max_concurrency: int,
    max_depth: int
) -> Dict[str, Any]:
    """
    Crawl a single website deeply and save each page as a cleaned Markdown file
    in a site-specific subdirectory, with parallelization.
    """
    crawled_urls = set()
    queued_urls = set()
    crawl_queue: asyncio.Queue[CrawlQueueItem] = asyncio.Queue()
    semaphore = asyncio.Semaphore(max_concurrency)
    results = {"success": [], "failed": [], "skipped_by_filter": [], "initial_url": start_url}

    try:
        parsed_start_url = urlparse(start_url)
        start_domain = parsed_start_url.netloc
        if not start_domain:
            results["failed"].append({"url": start_url, "error": "Could not extract domain from start URL"})
            print(f"Error: Could not extract domain from start URL: {start_url}")
            return results

        site_subdir_name = sanitize_dirname(start_url)
        site_output_path = os.path.join(output_dir, site_subdir_name)
        site_output_path = os.path.abspath(site_output_path)
        print(f"Crawl limited to domain: {start_domain}")
        print(f"Saving files for this site in: {site_output_path}")

        try:
            os.makedirs(site_output_path, exist_ok=True)
            if not os.path.exists(site_output_path):
                raise OSError(f"Failed to create directory: {site_output_path}")
        except Exception as e:
            results["failed"].append({"url": start_url, "error": f"Cannot create output directory: {e}"})
            print(f"Error creating output directory {site_output_path}: {e}")
            return results

    except Exception as e:
        results["failed"].append({"url": start_url, "error": f"Error parsing start URL or determining output path: {e}"})
        print(f"Error processing start URL {start_url} or determining output path: {e}")
        return results

    crawl_queue.put_nowait((start_url, 0, start_domain, site_output_path))
    queued_urls.add(start_url)

    print(f"Starting crawl for: {start_url} with max_depth={max_depth}, max_concurrency={max_concurrency}")

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
    )

    # ThreadPoolExecutor pour le post-traitement
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        async def crawl_page():
            """Worker function to process URLs from the queue."""
            while not crawl_queue.empty():
                try:
                    current_url, current_depth, crawl_start_domain, current_site_output_path = await crawl_queue.get()

                    if current_url in crawled_urls:
                        crawl_queue.task_done()
                        continue

                    try:
                        current_domain = urlparse(current_url).netloc
                        if current_domain != crawl_start_domain:
                            print(f"Skipping external URL: {current_url} (Domain: {current_domain}, Expected: {crawl_start_domain})")
                            crawled_urls.add(current_url)
                            crawl_queue.task_done()
                            continue
                    except Exception as e:
                        print(f"Error parsing domain for URL {current_url}: {e}. Skipping.")
                        crawled_urls.add(current_url)
                        crawl_queue.task_done()
                        continue

                    crawled_urls.add(current_url)
                    print(f"Crawling ({len(crawled_urls)}): {current_url} (Depth: {current_depth})")

                    filename = sanitize_filename(current_url)
                    output_path = os.path.join(current_site_output_path, filename)

                    if any(keyword in filename.lower() for keyword in EXCLUDE_KEYWORDS):
                        print(f"Skipping save for {current_url} due to filename filter: {filename}")
                        results["skipped_by_filter"].append(current_url)
                        if current_depth < max_depth:
                            async with semaphore:
                                async with AsyncWebCrawler(verbose=False) as crawler:
                                    result = await crawler.arun(url=current_url, config=config)
                                    if result.success:
                                        internal_links = result.links.get("internal", [])
                                        for link in internal_links:
                                            href = link["href"]
                                            try:
                                                absolute_url = urljoin(current_url, href)
                                                parsed_absolute_url = urlparse(absolute_url)
                                                if parsed_absolute_url.netloc == crawl_start_domain:
                                                    if absolute_url not in crawled_urls and absolute_url not in queued_urls:
                                                        crawl_queue.put_nowait((absolute_url, current_depth + 1, crawl_start_domain, current_site_output_path))
                                                        queued_urls.add(absolute_url)
                                            except Exception as link_e:
                                                print(f"Error processing link {href} from {current_url}: {link_e}")
                                    else:
                                        print(f"Failed to get links from {current_url} (skipped save): {result.error_message}")
                        crawl_queue.task_done()
                        continue

                    async with semaphore:
                        async with AsyncWebCrawler(verbose=False) as crawler:
                            result = await crawler.arun(url=current_url, config=config)

                        if result.success:
                            # Déléguer le traitement du Markdown et l'écriture à un thread
                            future = executor.submit(
                                process_markdown_and_save,
                                current_url,
                                result.markdown.raw_markdown,
                                output_path
                            )
                            process_result = future.result()  # Bloque jusqu'à ce que le thread termine
                            if process_result["status"] == "success":
                                results["success"].append(current_url)
                            else:
                                results["failed"].append({"url": current_url, "error": process_result["error"]})

                            if current_depth < max_depth:
                                internal_links = result.links.get("internal", [])
                                for link in internal_links:
                                    href = link["href"]
                                    try:
                                        absolute_url = urljoin(current_url, href)
                                        parsed_absolute_url = urlparse(absolute_url)
                                        if parsed_absolute_url.netloc == crawl_start_domain:
                                            if absolute_url not in crawled_urls and absolute_url not in queued_urls:
                                                crawl_queue.put_nowait((absolute_url, current_depth + 1, crawl_start_domain, current_site_output_path))
                                                queued_urls.add(absolute_url)
                                    except Exception as link_e:
                                        print(f"Error processing link {href} from {current_url}: {link_e}")
                        else:
                            print(f"Failed to crawl {current_url}: {result.error_message}")
                            results["failed"].append({"url": current_url, "error": result.error_message})

                    crawl_queue.task_done()
                except Exception as e:
                    print(f"Error in crawl_page worker: {e}")
                    crawl_queue.task_done()

        worker_tasks = []
        for _ in range(max_concurrency):
            task = asyncio.create_task(crawl_page())
            worker_tasks.append(task)

        await crawl_queue.join()

        for task in worker_tasks:
            task.cancel()

        await asyncio.gather(*worker_tasks, return_exceptions=True)

    print(f"Finished crawl for: {start_url}")
    return results

@app.post("/crawl_csv_upload")
async def crawl_csv_upload_endpoint(
    csv_file: UploadFile = File(...),
    output_dir: str = Form("./crawl_output_csv"),
    max_concurrency_per_site: int = Form(default=50, ge=1),
    max_depth: int = Form(default=2, ge=0)
):
    """
    FastAPI endpoint to crawl URLs provided in an uploaded CSV file.
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

        overall_results: Dict[str, Any] = {
            "status": "processing",
            "total_urls_from_csv": len(urls_to_crawl),
            "site_crawl_results": {}
        }

        for i, url in enumerate(urls_to_crawl):
            print(f"\n--- Processing site {i+1}/{len(urls_to_crawl)}: {url} ---")
            try:
                site_results = await crawl_website_single_site(
                    start_url=url,
                    output_dir=output_dir,
                    max_concurrency=max_concurrency_per_site,
                    max_depth=max_depth
                )
                overall_results["site_crawl_results"][url] = site_results
            except Exception as e:
                print(f"An unexpected error occurred during the crawl of {url}: {e}")
                overall_results["site_crawl_results"][url] = {"status": "error", "message": f"Unexpected error during site processing: {str(e)}"}

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
            print(f"\nOverall metadata saved to {metadata_path}")
        except Exception as e:
            print(f"Error saving overall metadata: {e}")
            overall_results["metadata_save_error"] = str(e)

        overall_results["status"] = "completed"
        print("\n--- Overall CSV processing completed ---")
        return overall_results
    except Exception as e:
        print(f"Critical error in crawl_csv_upload_endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred during request processing: {str(e)}")

if __name__ == "__main__":
    print("Starting FastAPI application...")
    print("Navigate to http://0.0.0.0:8001/docs for interactive documentation (Swagger UI).")
    uvicorn.run(app, host="0.0.0.0", port=8001)