This is a fairly complex, multi-functional FastAPI application designed for web crawling and content extraction, with features for incremental crawling based on sitemaps and a daemon mode for scheduled crawls.

```markdown
# 🕷️ Advanced Web Crawler & Markdown Extractor API

This FastAPI application provides a robust solution for crawling websites, extracting content into cleaned Markdown, and managing crawl jobs. It features incremental crawling based on sitemap last modification dates, parallel processing for multiple sites, and a daemon mode for scheduled daily crawls.

## 🎯 Core Purpose

*   **🔗 Batch Crawling:** Process multiple websites from a CSV list.
*   **📝 Markdown Extraction:** Convert web page content into cleaned Markdown format using `crawl4ai`.
*   **⏱️ Incremental Crawling:** Utilizes sitemaps (`sitemap.xml`) to identify and crawl only pages that have been modified since the last crawl, or new pages.
*   **💾 Persistent State:** Keeps track of the last crawled URL and time for each site to resume efficiently.
*   **🚀 Parallel Processing:** Crawls multiple sites concurrently using a thread pool and manages concurrency per site using `asyncio.Semaphore`.
*   **📄 Organized Output:** Saves cleaned Markdown files in a structured directory, organized by site.
*   **🌐 API Interface:** Provides a FastAPI endpoint for initiating crawls via CSV upload.
*   **🤖 Daemon Mode:** Can run as a scheduled daily job to keep content updated.

## ✨ Key Features

*   **FastAPI Backend:** Modern, high-performance API framework.
*   **`crawl4ai` Integration:** Leverages the `crawl4ai` library for efficient asynchronous web crawling and Markdown generation.
*   **CSV Input:** Accepts a list of URLs to crawl from a CSV file (one URL per line or in the first column).
*   **Configurable Crawl Parameters:**
    *   `output_dir`: Directory to save crawled data.
    *   `max_concurrency_per_site`: Limits concurrent requests to a single website.
    *   `max_depth`: Controls how deep the crawler goes from each starting URL.
    *   `max_threads`: Number of websites to crawl in parallel.
*   **Markdown Cleaning:** Includes a function to sanitize and clean the extracted Markdown (remove image tags, convert links to text, normalize whitespace, etc.).
*   **Filename & Directory Sanitization:** Creates safe filenames and directory names from URLs.
*   **Sitemap Parsing:**
    *   Automatically attempts to fetch and parse `sitemap.xml` for each site.
    *   Extracts URLs and their `lastmod` (last modification) timestamps.
*   **Incremental Crawl Logic:**
    *   Maintains a `file_info_list.csv` in the output directory to store the last crawled URL and timestamp for each starting site.
    *   Compares sitemap `lastmod` dates with the last crawl time to crawl only new or updated pages.
*   **Resumability & Progress Tracking:**
    *   Saves crawl progress for each site in a `<site_name>.json` file within its output directory.
    *   Avoids re-crawling already processed URLs within a job and across jobs (using `global_crawled_urls`).
    *   Skips sites that were successfully completed in previous runs.
*   **Logging:** Comprehensive logging to both a file (timestamped) and the console.
*   **Daemon Mode (`--daemon`):**
    *   Uses the `schedule` library to run crawls automatically at a configured time (e.g., daily at midnight).
*   **Thread Pool for Parallelism:** Efficiently manages concurrent crawling of multiple sites.

## 📋 Prerequisites

*   🐍 Python 3.x
*   📦 Python libraries: `fastapi`, `uvicorn`, `pydantic`, `crawl4ai`, `aiohttp`, `schedule`, `requests` (requests might be pulled by crawl4ai).
    ```bash
    pip install fastapi uvicorn pydantic crawl4ai aiohttp schedule requests
    ```
*   (Implicit) Network access for crawling and fetching sitemaps.

## 🔧 Configuration

*   **Logging:** Log files are created with timestamps (e.g., `crawl_log_YYYYMMDD_HHMMSS.log`).
*   **Crawl Parameters:** Configurable via the API endpoint or within the `run_daemon` function for scheduled tasks.
    *   Default output directory for API: `./crawl_output_csv`
    *   Daemon mode CSV path and output directory are hardcoded in `run_daemon` (consider making these configurable).
*   `EXCLUDE_KEYWORDS`: Currently empty, allowing all URLs. Can be populated to skip URLs containing certain keywords.

## 🚀 How to Use

### 1. 🌐 As a FastAPI Service (for on-demand crawls via CSV upload)

1.  **💾 Save the Script:** Save the code as a Python file (e.g., `main_crawler_api.py`).
2.  **▶️ Run the FastAPI Application:**
    ```bash
    uvicorn main_crawler_api:app --host 0.0.0.0 --port 8001 --reload
    ```
    *   `--reload` is useful for development. Remove it for production.
3.  **📄 Access API Documentation (Swagger UI):**
    Open your browser and navigate to `http://localhost:8001/docs`.
4.  **📤 Use the `/crawl_csv_upload` Endpoint:**
    *   Expand the endpoint in Swagger UI.
    *   Click "Try it out".
    *   Upload a CSV file containing URLs (one URL per line, or in a column that the `read_urls_from_csv` function is adapted to read).
    *   Optionally, adjust `output_dir`, `max_concurrency_per_site`, `max_depth`, and `max_threads`.
    *   Click "Execute".
5.  **📊 Monitor & Output:**
    *   Check the application's console for extensive logging.
    *   Crawled Markdown files will be saved in the specified `output_dir`, organized into subdirectories named after the sanitized domain of each starting URL.
    *   A `file_info_list.csv` will be created/updated in the `output_dir` to track crawl progress for incremental crawls.
    *   An `overall_metadata_upload.json` file will summarize the results of the batch crawl.
    *   Individual site progress is saved in `<output_dir>/<site_subdir_name>/<site_subdir_name>.json`.

### 2. 🤖 As a Scheduled Daemon (for automatic daily crawls)

1.  **💾 Save the Script:** (e.g., `main_crawler_api.py`).
2.  **🔧 Configure Daemon Settings:**
    *   Modify the hardcoded `csv_path`, `output_dir`, and crawl parameters within the `run_daemon()` function to suit your needs. Ensure the CSV file at `csv_path` exists and contains the URLs to be crawled daily.
3.  **▶️ Run in Daemon Mode:**
    ```bash
    python main_crawler_api.py --daemon
    ```
4.  **⚙️ Operation:**
    *   The script will start and log that the daemon is running.
    *   It will then execute the `crawl_from_csv` function daily at midnight (or as configured in `schedule.every().day.at("00:00")`).
    *   Output and progress tracking are the same as with the API-triggered crawl, but logs will primarily go to the timestamped log file.

## 📁 Output Structure (Example)

```
./crawl_output_csv/  (or your specified output_dir)
├── example_com/
│   ├── example_com_index.md
│   ├── example_com_about_us.md
│   └── example_com.json  (progress for example.com)
├── another_site_org/
│   ├── another_site_org_index.md
│   ├── another_site_org_blog_post_1.md
│   └── another_site_org.json (progress for another-site.org)
├── file_info_list.csv
└── overall_metadata_upload.json (if triggered by API)
└── overall_metadata.json (if triggered by daemon/direct crawl_from_csv)
```

## 📝 Key Functions & Logic

*   **`read_urls_from_csv`**: Parses URLs from CSV content.
*   **`clean_markdown`**: Applies regex rules to sanitize extracted Markdown.
*   **`sanitize_filename` / `sanitize_dirname`**: Creates safe file/directory names from URLs.
*   **`fetch_sitemap`**: Asynchronously fetches and parses `sitemap.xml`.
*   **`parse_lastmod`**: Converts sitemap lastmod strings to datetime objects.
*   **`read_file_info_list` / `write_file_info_list`**: Manages the state for incremental crawling.
*   **`crawl_website_single_site`**: Core asynchronous logic for crawling one website, considering sitemap modifications and depth.
*   **`run_crawl_in_thread`**: Wraps the async crawl logic to be run in a `ThreadPoolExecutor`.
*   **`find_progress_files`**: Locates existing site progress JSON files to resume/avoid re-crawling.
*   **`crawl_from_csv`**: Orchestrates the crawling of multiple sites from a CSV, managing threads and global state.
*   **FastAPI Endpoints (`/crawl_csv_upload`):** Exposes crawling functionality via HTTP.
*   **`run_daemon`**: Implements the scheduled task execution.

## 💡 Potential Improvements & Considerations

*   **Configuration Management:** Move hardcoded paths and parameters (especially in `run_daemon`) to environment variables or a configuration file.
*   **Error Handling & Retries:** Enhance retry mechanisms for network issues during crawling or sitemap fetching.
*   **Distributed Task Queue:** For very large-scale crawling, consider using a distributed task queue like Celery instead of `ThreadPoolExecutor`.
*   **Robots.txt Respect:** The `crawl4ai` library has a `check_robots_txt` option, which is enabled. Ensure it behaves as expected for politeness.
*   **Dynamic Content:** `crawl4ai` handles JavaScript rendering to some extent. Evaluate its effectiveness on heavily dynamic sites.
*   **Advanced Filtering:** The `EXCLUDE_KEYWORDS` is basic; more sophisticated URL filtering might be needed.
*   **Resource Limits:** Monitor CPU, memory, and network usage, especially when running with high `max_threads`.
```
