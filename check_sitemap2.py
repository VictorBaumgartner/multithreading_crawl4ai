from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import uvicorn
import csv
import io

app = FastAPI()

class URLInfo(BaseModel):
    input_url: str
    url: str
    last_modified: Optional[str] = None

def get_sitemap_urls(url):
    try:
        request = urllib.request.Request(url, method='GET')
        with urllib.request.urlopen(request) as response:
            content = response.read().decode('utf-8')
            root = ET.fromstring(content)
            ns = root.tag.split('}', 1)[0] + '}' if '}' in root.tag else ''
            root_tag_local = root.tag.split('}')[-1] if '}' in root.tag else root.tag
            result = []
            loc_tag = ns + 'loc' if ns else 'loc'
            lastmod_tag = ns + 'lastmod' if ns else 'lastmod'
            if root_tag_local == 'sitemapindex':
                sitemap_tag = ns + 'sitemap' if ns else 'sitemap'
                for child in root.findall(sitemap_tag):
                    loc_elem = child.find(loc_tag)
                    if loc_elem is not None:
                        result.extend(get_sitemap_urls(loc_elem.text))
            elif root_tag_local == 'urlset':
                for url_elem in root.findall(ns + 'url' if ns else 'url'):
                    loc_elem = url_elem.find(loc_tag)
                    lastmod_elem = url_elem.find(lastmod_tag)
                    if loc_elem is not None:
                        lastmod = lastmod_elem.text if lastmod_elem is not None else None
                        result.append((loc_elem.text, lastmod))
            return result
    except Exception as e:
        print(f"Error processing sitemap {url}: {e}")
        return []

def get_all_sitemaps(base_url):
    parsed = urllib.parse.urlparse(base_url)
    scheme = parsed.scheme
    netloc = parsed.netloc
    robots_url = f"{scheme}://{netloc}/robots.txt"
    try:
        with urllib.request.urlopen(robots_url) as response:
            robots_content = response.read().decode('utf-8')
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return get_sitemap_urls(f"{scheme}://{netloc}/sitemap.xml")
        return []
    except Exception as e:
        print(f"Error fetching robots.txt for {base_url}: {e}")
        return []
    
    sitemaps = []
    for line in robots_content.splitlines():
        if line.strip().lower().startswith('sitemap:'):
            sitemap_rel = line.split(':', 1)[1].strip()
            sitemap_abs = urllib.parse.urljoin(f"{scheme}://{netloc}/", sitemap_rel)
            sitemaps.append(sitemap_abs)
    
    all_urls = []
    for sm in sitemaps:
        all_urls.extend(get_sitemap_urls(sm))
    return all_urls

@app.post("/sitemaps/csv", response_model=dict)
async def get_sitemap_data_from_csv(file: UploadFile = File(...)):
    contents = await file.read()
    csv_text = contents.decode('utf-8')
    reader = csv.DictReader(io.StringIO(csv_text))
    results = []

    for row in reader:
        input_url = row.get("url")
        if not input_url:
            continue
        all_urls = get_all_sitemaps(input_url)
        if all_urls:
            for url, lastmod in all_urls:
                results.append(URLInfo(
                    input_url=input_url,
                    url=url,
                    last_modified=lastmod
                ))
        else:
            results.append(URLInfo(
                input_url=input_url,
                url="",
                last_modified=None
            ))

    return {"results": [r.model_dump() for r in results]}  # ✅ Fixes deprecation

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
