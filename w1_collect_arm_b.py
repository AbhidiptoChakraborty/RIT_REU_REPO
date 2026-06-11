"""
W1 Arm B Collector (v6)

Changes vs v5-fixed:
  - CATEGORIES is now a dict of {cat: [list of queries]} so when one query
    exhausts (S2 400 at offset>1000), the next query picks up.
  - S2 400 now breaks the page loop (query exhausted) instead of continuing
    (which was spinning on 400s forever burning MAX_PAGES budget).
  - q-bio queries replaced with specific computational biology terms that
    actually return arXiv papers with journal venues.
"""

import os
import time
import requests
import xml.etree.ElementTree as ET
import pandas as pd
from Levenshtein import distance as lev_distance

from w1_dataset_schema import COLUMNS, validate_row, validate_dataset

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

S2_API_KEY = "s2k-Wxrcv54yu97WQtaD0lRfMcvwEd6fTWOvFgHnqF5Q"

TARGET_TOTAL = 200
YEAR_MIN, YEAR_MAX = 2000, 2019

S2_BATCH         = 100
MAX_PAGES_PER_QUERY = 10   # S2 400s after ~offset=1000; no point going beyond

S2_DELAY    = 2.0
ARXIV_DELAY = 6.0

REVISION_THRESHOLD = 0.20  # strictly greater than this

S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = (
    "paperId,title,abstract,authors,venue,year,url,"
    "publicationVenue,externalIds,fieldsOfStudy"
)

# Multiple queries per category. When the target isn't met after one query
# exhausts, the next query runs automatically.
CATEGORY_QUERIES = {
    "cs.CL": [
        "natural language processing",
        "computational linguistics arxiv",
        "machine translation arxiv",
        "text classification arxiv",
        "named entity recognition arxiv",
        "sentiment analysis arxiv",
        "question answering arxiv",
        "information extraction arxiv",
    ],
    "cs.LG": [
        "machine learning",
        "deep learning arxiv",
        "neural network optimization arxiv",
        "reinforcement learning arxiv",
        "graph neural network arxiv",
        "generative adversarial network arxiv",
        "transfer learning arxiv",
        "federated learning arxiv",
    ],
    "q-bio": [
        "computational biology arxiv",
        "bioinformatics machine learning arxiv",
        "protein structure prediction arxiv",
        "genomics deep learning arxiv",
        "gene expression neural network arxiv",
        "systems biology arxiv",
        "drug discovery machine learning arxiv",
        "single cell RNA sequencing arxiv",
    ],
}

# ─────────────────────────────────────────────
# UTIL
# ─────────────────────────────────────────────

def norm_lev(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return lev_distance(a, b) / max(len(a), len(b))


def s2_headers():
    return {"x-api-key": S2_API_KEY}


def s2_search(query, offset):
    """
    One page of S2 results.
    Returns (papers, exhausted) where exhausted=True means this query is done
    (S2 returned 400, which happens when offset > ~1000 for most queries).
    """
    params = {
        "query":  query,
        "fields": S2_FIELDS,
        "limit":  S2_BATCH,
        "offset": offset,
    }
    wait = 15
    for attempt in range(5):
        try:
            r = requests.get(
                S2_SEARCH_URL, params=params,
                headers=s2_headers(), timeout=30
            )
            if r.status_code == 429:
                print(f"  [S2 429] waiting {wait}s (attempt {attempt+1}/5)...")
                time.sleep(wait)
                wait *= 2
                continue
            if r.status_code == 400:
                # S2 offset limit reached — this query is exhausted
                print(f"  [S2 400] query exhausted at offset={offset}")
                return [], True
            if r.status_code >= 400:
                print(f"  [S2 error {r.status_code}] skipping page")
                return [], False   # transient error — caller may retry next page
            return r.json().get("data", []), False
        except Exception as e:
            print(f"  [S2 exception] {e}")
            return [], False
    return [], False


def get_arxiv_id(paper):
    ext = paper.get("externalIds") or {}
    return ext.get("ArXiv")


def fetch_arxiv_v1(arxiv_id):
    """Fetch arXiv v1 abstract. Retries up to 4× on 429 with exponential backoff."""
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}v1&max_results=1"
    wait = 20
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 429:
                print(f"    [arXiv 429] waiting {wait}s (attempt {attempt+1}/4)...")
                time.sleep(wait)
                wait *= 2
                continue
            root = ET.fromstring(r.text)
            entry   = root.find("{http://www.w3.org/2005/Atom}entry")
            if entry is None:
                return None
            summary = entry.find("{http://www.w3.org/2005/Atom}summary")
            if summary is None or not summary.text:
                return None
            return summary.text.strip().replace("\n", " ")
        except Exception as e:
            print(f"    [arXiv fetch failed] {arxiv_id}: {e}")
            return None
    print(f"    [arXiv gave up] {arxiv_id}")
    return None


# ─────────────────────────────────────────────
# CONTAMINATION PROBE (HOOK)
# ─────────────────────────────────────────────

def ts_guessing_score(text: str):
    """Placeholder for Week 1 contamination probe. Replace with real probe."""
    return None


# ─────────────────────────────────────────────
# ROW BUILDER
# ─────────────────────────────────────────────

def make_row(i, version, arxiv_id, title, authors,
             abstract, venue, year, domain, url, notes):
    version_suffix = "v1" if version == "v1" else "fn"
    row = {
        "id":                f"B-{i:04d}-{version_suffix}",
        "arm":               "B",
        "paper_id":          f"arXiv:{arxiv_id}",
        "title":             title,
        "authors":           ", ".join(authors),
        "abstract":          abstract,
        "venue":             venue,
        "venue_type":        "preprint",
        "year":              year,
        "domain":            domain,
        "source_url":        url,
        "version":           version,
        "collection_date":   "2026-06-10",
        "ts_guessing_score": ts_guessing_score(abstract),
        "is_contaminated":   None,
        "notes":             notes,
    }
    validate_row(row)
    return row


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def collect_arm_b():
    rows     = []
    seen     = set()
    global_i = 1

    per_cat_target = TARGET_TOTAL // len(CATEGORY_QUERIES)  # 66 each
    cat_counts = {c: 0 for c in CATEGORY_QUERIES}

    for cat, queries in CATEGORY_QUERIES.items():
        print(f"\n=== CATEGORY {cat} (target {per_cat_target}, "
              f"{len(queries)} queries) ===")

        for query in queries:
            if cat_counts[cat] >= per_cat_target:
                break

            print(f"\n  Query: '{query}'")
            pages = 0

            while (
                cat_counts[cat] < per_cat_target
                and pages < MAX_PAGES_PER_QUERY
            ):
                offset = pages * S2_BATCH
                pages += 1

                time.sleep(S2_DELAY)
                papers, exhausted = s2_search(query, offset)

                if exhausted:
                    # 400 = offset limit hit; move to next query
                    break

                if not papers:
                    # transient error or empty results; try next page
                    continue

                for p in papers:
                    if cat_counts[cat] >= per_cat_target:
                        break

                    year = p.get("year")
                    if not year or year < YEAR_MIN or year > YEAR_MAX:
                        continue

                    arxiv_id = get_arxiv_id(p)
                    if not arxiv_id or arxiv_id in seen:
                        continue

                    pub_abs = (p.get("abstract") or "").strip()
                    if not pub_abs:
                        continue

                    venue = (p.get("venue") or "").lower()
                    if "arxiv" in venue or not venue:
                        continue

                    title   = p.get("title", "")
                    authors = [a["name"] for a in p.get("authors", [])]

                    time.sleep(ARXIV_DELAY)
                    v1 = fetch_arxiv_v1(arxiv_id)
                    if not v1:
                        continue

                    dist = norm_lev(v1, pub_abs)
                    if dist <= REVISION_THRESHOLD:
                        continue

                    notes = f"lev={dist:.3f}; q={query!r}"

                    try:
                        row_v1 = make_row(
                            global_i, "v1",
                            arxiv_id, title, authors,
                            v1, "arXiv", year,
                            cat, f"https://arxiv.org/abs/{arxiv_id}",
                            notes
                        )
                        row_final = make_row(
                            global_i, "final",
                            arxiv_id, title, authors,
                            pub_abs, venue, year,
                            cat, f"https://arxiv.org/abs/{arxiv_id}",
                            notes
                        )
                    except Exception as e:
                        print(f"  [schema error] {e}")
                        continue

                    rows.append(row_v1)
                    rows.append(row_final)
                    seen.add(arxiv_id)
                    cat_counts[cat] += 1
                    global_i += 1

                    print(f"  ✓ {cat}: {cat_counts[cat]:>3}/{per_cat_target}"
                          f"  dist={dist:.2f}  {title[:52]}")

        print(f"\n{cat} done: {cat_counts[cat]}/{per_cat_target}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    for cat, count in cat_counts.items():
        status = "✓" if count >= per_cat_target else f"NEED {per_cat_target-count} MORE"
        print(f"  {cat:<10} {count:>3}/{per_cat_target}  {status}")
    print(f"  TOTAL rows: {len(rows)}  ({len(rows)//2} unique papers)")

    df = pd.DataFrame(rows, columns=COLUMNS)
    validate_dataset(df)
    return df


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    df = collect_arm_b()

    os.makedirs("data/raw", exist_ok=True)
    out = "data/raw/arm_b_sample.csv"
    df.to_csv(out, index=False)

    print(f"\nSaved: {out}")
    print(df[["id", "version", "year", "venue", "notes"]].head(10).to_string(index=False))