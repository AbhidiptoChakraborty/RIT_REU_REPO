"""
Run this script FIRST to see exactly what venue strings
Semantic Scholar returns for PNAS-related queries.
It prints every unique venue string so you can add the
right substrings to VENUE_BUCKETS.
"""

import time
import requests
from collections import Counter

API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
API_KEY = "s2k-Wxrcv54yu97WQtaD0lRfMcvwEd6fTWOvFgHnqF5Q"

FIELDS = "paperId,title,venue,publicationVenue,year"
YEAR_RANGE = "2000-2019"

QUERIES = [
    "machine learning PNAS",
    "deep learning national academy sciences",
    "neural network national academy",
    "computational biology PNAS",
    "convolutional neural network PNAS",
]

def fetch(query, limit=50):
    headers = {"x-api-key": API_KEY}
    params = {
        "query":  query,
        "limit":  limit,
        "fields": FIELDS,
        "year":   YEAR_RANGE,
    }
    for attempt in range(3):
        r = requests.get(API_URL, params=params, headers=headers)
        if r.status_code == 429:
            wait = 10 * (attempt + 1)
            print(f"  Rate limited, waiting {wait}s...")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json().get("data", [])
    return []

venue_counter = Counter()

for q in QUERIES:
    print(f"\nQuery: '{q}'")
    papers = fetch(q)
    print(f"  {len(papers)} results")
    for p in papers:
        pub_venue = (p.get("publicationVenue") or {}).get("name", "")
        venue     = p.get("venue", "")
        # prefer publicationVenue.name, fall back to venue field
        v = pub_venue or venue
        if v:
            venue_counter[v] += 1
            # print papers that look PNAS-related
            lower = v.lower()
            if any(kw in lower for kw in ["academy", "pnas", "national"]):
                print(f"  *** POSSIBLE MATCH: '{v}'  |  title: {p.get('title','')[:60]}")
        else:
            print(f"  [NO VENUE]  title: {p.get('title','')[:60]}")
    time.sleep(1)

print("\n" + "="*60)
print("ALL UNIQUE VENUE STRINGS SEEN (sorted by frequency):")
print("="*60)
for venue, count in venue_counter.most_common():
    print(f"  {count:>3}x  '{venue}'")