🕷️ Web Crawler & Markdown Extractor API
A powerful FastAPI application for crawling websites, extracting content into clean Markdown, and managing crawl jobs. It supports incremental crawling using sitemaps, parallel processing, and a daemon mode for scheduled tasks.
🎯 Purpose
This application automates web content extraction with the following capabilities:

🔗 Batch Crawling: Processes multiple websites from a CSV list.
📝 Markdown Extraction: Converts web content to clean Markdown using crawl4ai.
⏱️ Incremental Crawling: Crawls only updated or new pages based on sitemap lastmod timestamps.
💾 Persistent State: Tracks crawl progress to resume efficiently.
🚀 Parallel Processing: Crawls multiple sites concurrently with configurable concurrency limits.
📂 Structured Output: Saves Markdown files in organized directories by site.
🌐 API Interface: Provides a FastAPI endpoint for on-demand crawling via CSV upload.
🤖 Daemon Mode: Runs scheduled daily crawls for automated updates.

✨ Features

FastAPI Backend: High-performance API for seamless integration.
Crawl4AI Integration: Asynchronous crawling and Markdown generation.
CSV Input: Reads URLs from a CSV file (one URL per line or in the first column).
Configurable Parameters:
output_dir: Where crawled data is saved.
max_concurrency_per_site: Limits concurrent requests per site.
max_depth: Controls crawl depth from starting URLs.
max_threads: Number of sites crawled in parallel.


Markdown Cleaning: Sanitizes Markdown by removing images, normalizing links, and cleaning whitespace.
Sitemap Support: Parses sitemap.xml to identify updated or new pages.
Progress Tracking: Saves crawl state in file_info_list.csv and per-site JSON files.
Logging: Detailed logs to console and timestamped files (e.g., crawl_log_YYYYMMDD_HHMMSS.log).
Daemon Mode: Schedules daily crawls using the schedule library.

📋 Prerequisites

🐍 Python: 3.8 or higher.
📦 Dependencies:pip install fastapi uvicorn pydantic crawl4ai aiohttp schedule requests


🌐 Network Access: Required for crawling and fetching sitemaps.

🔧 Setup

Install Dependencies:
pip install fastapi uvicorn pydantic crawl4ai aiohttp schedule requests


Save the Script:

Save the code as main_crawler_api.py.


Prepare CSV Input:

Create a CSV file with URLs to crawl (e.g., urls.csv).
Example format:url
https://example.com
https://another-site.org




Configure Settings:

For API usage, set output_dir via the endpoint.
For daemon mode, edit run_daemon() in the script to set csv_path, output_dir, and crawl parameters (e.g., max_concurrency_per_site=2, max_depth=3, max_threads=4).



🚀 Usage
Option 1: 🌐 FastAPI Service (On-Demand Crawling)

Start the Server:
uvicorn main_crawler_api:app --host 0.0.0.0 --port 8001 --reload


Use --reload for development; omit for production.


Access API Documentation:

Open http://localhost:8001/docs in a browser to view Swagger UI.


Use the `/ statutory endpoint:

Navigate to /crawl_csv_upload in Swagger UI.
Upload your CSV file.
Optionally adjust parameters (output_dir, max_concurrency_per_site, max_depth, max_threads).
Click "Execute" to start the crawl.


Monitor Output:

Check console logs for real-time progress.
Find Markdown files in <output_dir> (default: ./crawl_output_csv), organized by site domain.
Review file_info_list.csv for crawl state and overall_metadata_upload.json for summary.



Option 2: 🤖 Daemon Mode (Scheduled Crawling)

Configure Daemon:

In main_crawler_api.py, update run_daemon() with your CSV path and parameters:csv_path = "path/to/urls.csv"
output_dir = "path/to/crawl_output"




Run in Daemon Mode:
python main_crawler_api.py --daemon


The script runs indefinitely, crawling daily at midnight (configurable in schedule.every().day.at("00:00")).


Monitor Output:

Logs are saved to timestamped files (e.g., crawl_log_YYYYMMDD_HHMMSS.log).
Markdown files and metadata are saved as in API mode.



📁 Output Structure
./crawl_output_csv/
├── example_com/
│   ├── example_com_index.md
│   ├── example_com_about.md
│   └── example_com.json
├── another_site_org/
│   ├── another_site_org_index.md
│   ├── another_site_org_blog.md
│   └── another_site_org.json
├── file_info_list.csv
├── overall_metadata_upload.json  (API mode)
└── overall_metadata.json  (daemon/direct mode)


Markdown Files: Cleaned content per page.
JSON Files: Track progress for each site.
CSV File: Stores last crawl details for incremental crawling.
Metadata JSON: Summarizes crawl results.

🛠️ Key Components

CSV Parsing: read_urls_from_csv extracts URLs from CSV files.
Markdown Cleaning: clean_markdown sanitizes output for consistency.
Sitemap Parsing: fetch_sitemap and parse_lastmod handle incremental crawling.
Crawl Logic: crawl_website_single_site manages async crawling with crawl4ai.
Parallelism: run_crawl_in_thread uses ThreadPoolExecutor for multi-site crawling.
API Endpoint: /crawl_csv_upload handles CSV uploads and crawl initiation.
Daemon: run_daemon schedules automated crawls.

🔍 Troubleshooting

Network Issues: Ensure stable internet; check crawl_log*.log for errors.
Invalid CSV: Verify CSV format (URLs in first column or one per line).
Sitemap Errors: Some sites may lack sitemap.xml; the crawler will fall back to full crawling.
Concurrency Limits: Adjust max_concurrency_per_site or max_threads if rate-limited by sites.
Dynamic Content: Test crawl4ai on JavaScript-heavy sites, as rendering may vary.

💡 Potential Improvements

⚙️ Config File: Replace hardcoded paths with a config file or environment variables.
🔄 Retries: Add retry logic for failed requests or sitemaps.
🚀 Scalability: Use a task queue (e.g., Celery) for large-scale crawling.
🤝 Politeness: Verify crawl4ai respects robots.txt as configured.
🔍 Advanced Filtering: Enhance EXCLUDE_KEYWORDS for more granular URL filtering.

📜 License
This project is licensed under the MIT License.
