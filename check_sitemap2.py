from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import uvicorn
import csv
import io
import os # Import os for file operations

app = FastAPI()

class URLInfo(BaseModel):
    input_url: str
    url: str
    last_modified: Optional[str] = None

def get_sitemap_urls(url):
    """
    Recursively fetches URLs and their last modified dates from a sitemap XML.
    Handles sitemap indexes and individual sitemap files.
    """
    try:
        request = urllib.request.Request(url, method='GET')
        with urllib.request.urlopen(request, timeout=5) as response: # Added timeout
            content = response.read().decode('utf-8')
            root = ET.fromstring(content)
            
            # Extract namespace if present, to correctly parse XML elements
            ns = ''
            if '}' in root.tag:
                ns = root.tag.split('}', 1)[0] + '}'
            
            root_tag_local = root.tag.split('}')[-1] if '}' in root.tag else root.tag
            result = []
            
            loc_tag = ns + 'loc' if ns else 'loc'
            lastmod_tag = ns + 'lastmod' if ns else 'lastmod'

            if root_tag_local == 'sitemapindex':
                # If it's a sitemap index, recursively fetch URLs from nested sitemaps
                sitemap_tag = ns + 'sitemap' if ns else 'sitemap'
                print(f"Found sitemapindex: {url}")
                for child in root.findall(sitemap_tag):
                    loc_elem = child.find(loc_tag)
                    if loc_elem is not None and loc_elem.text:
                        print(f"  Processing nested sitemap: {loc_elem.text}")
                        result.extend(get_sitemap_urls(loc_elem.text))
            elif root_tag_local == 'urlset':
                # If it's a URL set, extract individual URLs and their last modified dates
                print(f"Found urlset: {url}")
                for url_elem in root.findall(ns + 'url' if ns else 'url'):
                    loc_elem = url_elem.find(loc_tag)
                    lastmod_elem = url_elem.find(lastmod_tag)
                    if loc_elem is not None and loc_elem.text:
                        lastmod = lastmod_elem.text if lastmod_elem is not None else None
                        result.append((loc_elem.text, lastmod))
            else:
                print(f"Warning: Unknown sitemap root tag: {root.tag} for URL: {url}")
            return result
    except urllib.error.URLError as e:
        print(f"Network error fetching sitemap {url}: {e.reason}")
        return []
    except ET.ParseError as e:
        print(f"XML parsing error for sitemap {url}: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred processing sitemap {url}: {e}")
        return []

def get_all_sitemaps(base_url):
    """
    Discovers all sitemaps for a given base URL by checking robots.txt
    and common sitemap locations.
    """
    parsed = urllib.parse.urlparse(base_url)
    scheme = parsed.scheme
    netloc = parsed.netloc
    
    robots_url = f"{scheme}://{netloc}/robots.txt"
    potential_sitemaps = []
    
    print(f"Attempting to fetch robots.txt for: {robots_url}")
    try:
        with urllib.request.urlopen(robots_url, timeout=5) as response: # Added timeout
            robots_content = response.read().decode('utf-8')
            for line in robots_content.splitlines():
                if line.strip().lower().startswith('sitemap:'):
                    sitemap_rel = line.split(':', 1)[1].strip()
                    sitemap_abs = urllib.parse.urljoin(f"{scheme}://{netloc}/", sitemap_rel)
                    potential_sitemaps.append(sitemap_abs)
            print(f"Sitemaps found in robots.txt: {potential_sitemaps}")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"robots.txt not found (404) for {base_url}. Trying common sitemap locations.")
        else:
            print(f"HTTP Error fetching robots.txt for {base_url}: {e}")
    except urllib.error.URLError as e:
        print(f"Network error fetching robots.txt for {base_url}: {e.reason}")
    except Exception as e:
        print(f"An unexpected error occurred fetching robots.txt for {base_url}: {e}")
    
    # Add common sitemap locations if none found in robots.txt or robots.txt was not found
    if not potential_sitemaps:
        print("No sitemaps found in robots.txt or robots.txt not accessible. Checking common paths.")
        common_sitemap_paths = ["/sitemap.xml", "/sitemap_index.xml", "/sitemap_news.xml", "/sitemap_pages.xml"]
        for path in common_sitemap_paths:
            full_url = f"{scheme}://{netloc}{path}"
            # Make a HEAD request to check if the sitemap exists without downloading it fully
            try:
                req = urllib.request.Request(full_url, method='HEAD')
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        potential_sitemaps.append(full_url)
                        print(f"Found common sitemap at: {full_url}")
            except (urllib.error.HTTPError, urllib.error.URLError) as e:
                # print(f"Common sitemap {full_url} not found or error: {e}") # Too verbose
                pass
            except Exception as e:
                print(f"Error checking common sitemap {full_url}: {e}")

    all_urls = []
    processed_sitemaps = set() # To avoid processing the same sitemap multiple times

    # Process potential sitemaps
    for sm_url in potential_sitemaps:
        if sm_url not in processed_sitemaps:
            print(f"Processing sitemap URL: {sm_url}")
            current_sitemap_urls = get_sitemap_urls(sm_url)
            all_urls.extend(current_sitemap_urls)
            processed_sitemaps.add(sm_url)

    # Fallback: if no sitemaps found via robots.txt or common paths, try the base URL directly
    if not all_urls and not potential_sitemaps:
        default_sitemap_url = f"{scheme}://{netloc}/sitemap.xml"
        print(f"No sitemaps found. Attempting default: {default_sitemap_url}")
        all_urls.extend(get_sitemap_urls(default_sitemap_url))

    return all_urls

@app.post("/sitemaps/csv", response_model=dict)
async def get_sitemap_data_from_csv(file: UploadFile = File(...)):
    """
    Processes an uploaded CSV file containing URLs, fetches sitemap data for each.
    It appends only URLs with a 'last_modified' date to 'all_sitemaps.csv'
    and writes initial URLs that failed to yield any sitemap data with a
    'last_modified' date to 'failed_urls.csv', both within an 'output' folder.
    """
    contents = await file.read()
    csv_text = contents.decode('utf-8')
    reader = csv.DictReader(io.StringIO(csv_text))
    
    # Define the output directory and filenames
    output_dir = "output"
    all_sitemaps_filename = os.path.join(output_dir, "all_sitemaps.csv")
    failed_urls_filename = os.path.join(output_dir, "failed_urls.csv")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # List to store URLs that failed to produce valid sitemap entries
    failed_urls_to_record = []
    processed_sitemap_entries_count = 0

    # Determine if the all_sitemaps.csv file exists and is empty to decide whether to write headers
    all_sitemaps_file_exists = os.path.exists(all_sitemaps_filename)
    
    # Open the all_sitemaps.csv file in append mode ('a')
    with open(all_sitemaps_filename, 'a', newline='', encoding='utf-8') as all_sitemaps_outfile:
        sitemap_fieldnames = ["input_url", "url", "last_modified"]
        sitemap_writer = csv.DictWriter(all_sitemaps_outfile, fieldnames=sitemap_fieldnames)
        
        # Write header for all_sitemaps.csv only if file is new or currently empty
        if not all_sitemaps_file_exists or os.stat(all_sitemaps_filename).st_size == 0:
            sitemap_writer.writeheader()

        for row in reader:
            input_url = row.get("url") # Assuming the input CSV has a 'url' column
            if not input_url:
                print(f"Skipping row due to missing 'url' key in input CSV: {row}")
                continue
            
            print(f"Processing input URL from CSV: {input_url}")
            
            # Flag to track if any valid sitemap entry was found for the current input_url
            found_valid_sitemap_entry_for_this_url = False
            
            all_urls_from_site = get_all_sitemaps(input_url)
            
            if all_urls_from_site:
                for url, lastmod in all_urls_from_site:
                    # ONLY append if last_modified date is found
                    if lastmod is not None:
                        row_to_write = URLInfo(
                            input_url=input_url,
                            url=url,
                            last_modified=lastmod
                        )
                        sitemap_writer.writerow(row_to_write.model_dump())
                        processed_sitemap_entries_count += 1
                        found_valid_sitemap_entry_for_this_url = True
                        print(f"  Appended to all_sitemaps.csv: {url} (Last Modified: {lastmod})")
                    else:
                        print(f"  Skipped (no last_modified): {url}")
            else:
                print(f"No sitemap URLs found for input: {input_url}")
            
            # If no valid sitemap entries were found for this input_url, record it as failed
            if not found_valid_sitemap_entry_for_this_url:
                failed_urls_to_record.append(input_url)
                print(f"  Recorded as failed: {input_url}")

    # Now, write the failed URLs to failed_urls.csv
    failed_urls_file_exists = os.path.exists(failed_urls_filename)
    with open(failed_urls_filename, 'a', newline='', encoding='utf-8') as failed_urls_outfile:
        failed_writer = csv.writer(failed_urls_outfile)
        
        # Write header for failed_urls.csv only if file is new or currently empty
        if not failed_urls_file_exists or os.stat(failed_urls_filename).st_size == 0:
            failed_writer.writerow(["input_url"])
        
        for failed_url in failed_urls_to_record:
            failed_writer.writerow([failed_url])
    
    return {
        "message": (
            f"Sitemap data processed. "
            f"Total entries with last_modified date appended to '{all_sitemaps_filename}': {processed_sitemap_entries_count}. "
            f"Initial URLs that yielded no valid sitemap entries written to '{failed_urls_filename}': {len(failed_urls_to_record)}."
        )
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)