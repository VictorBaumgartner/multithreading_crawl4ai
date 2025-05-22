import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
import csv

def get_sitemap_info(url):
    """
    Fetches a sitemap URL and recursively retrieves all sub-sitemaps if it's a sitemap index.
    Returns a list of tuples (sitemap_url, last_modified_datetime).
    """
    try:
        request = urllib.request.Request(url, method='GET')
        with urllib.request.urlopen(request) as response:
            last_modified_str = response.getheader('Last-Modified')
            last_modified = parsedate_to_datetime(last_modified_str) if last_modified_str else None
            content = response.read().decode('utf-8')
            root = ET.fromstring(content)
            root_tag_local = root.tag.split('}')[-1] if '}' in root.tag else root.tag
            result = []
            if root_tag_local in ['sitemapindex', 'urlset']:
                result.append((url, last_modified))
                if root_tag_local == 'sitemapindex':
                    ns = root.tag.split('}', 1)[0] + '}' if '}' in root.tag else ''
                    sitemap_tag = ns + 'sitemap' if ns else 'sitemap'
                    loc_tag = ns + 'loc' if ns else 'loc'
                    for child in root.findall(sitemap_tag):
                        loc_elem = child.find(loc_tag)
                        if loc_elem is not None:
                            loc = loc_elem.text
                            result.extend(get_sitemap_info(loc))
            return result
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return []

def get_all_sitemaps(base_url):
    """
    Retrieves all sitemaps for a given website by checking robots.txt or default locations.
    Returns a list of tuples (sitemap_url, last_modified_datetime).
    """
    parsed = urllib.parse.urlparse(base_url)
    scheme = parsed.scheme
    netloc = parsed.netloc
    robots_url = f"{scheme}://{netloc}/robots.txt"
    try:
        with urllib.request.urlopen(robots_url) as response:
            robots_content = response.read().decode('utf-8')
    except urllib.error.HTTPError as e:
        if e.code == 404:
            default_sitemap = f"{scheme}://{netloc}/sitemap.xml"
            return get_sitemap_info(default_sitemap)
        else:
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
    
    all_sitemaps_info = []
    for sm in sitemaps:
        all_sitemaps_info.extend(get_sitemap_info(sm))
    return all_sitemaps_info

def main(input_file='input.csv', output_file='output.csv'):
    with open(input_file, 'r') as f_in, open(output_file, 'w', newline='') as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.writer(f_out)
        writer.writerow(['input_url', 'sitemap_url', 'last_modified'])
        
        for row in reader:
            input_url = row['url']
            all_sms = get_all_sitemaps(input_url)
            if all_sms:
                for sm_url, lm in all_sms:
                    lm_str = lm.isoformat() if lm else ''
                    writer.writerow([input_url, sm_url, lm_str])
            else:
                writer.writerow([input_url, '', ''])

if __name__ == "__main__":
    main()