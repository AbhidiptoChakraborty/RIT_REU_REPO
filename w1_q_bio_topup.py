"""
W1 Arm B — q-bio Top-up Collector
===================================
Collects the remaining q-bio papers needed to reach 66 total.

Run AFTER the main arm_b_collector.py has already saved
data/raw/arm_b_sample.csv with cs.CL and cs.LG complete.

This script:
  1. Loads the existing CSV to find already-seen arXiv IDs.
  2. Counts how many q-bio papers are already collected.
  3. Runs additional q-bio queries (fresh set, no overlap with v6 queries)
     until the q-bio target is met.
  4. Appends new rows to the existing CSV and re-validates.

Usage:
  python arm_b_qbio_topup.py
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

EXISTING_CSV     = "data/raw/arm_b_sample.csv"
OUT_CSV          = "data/raw/arm_b_sample.csv"   # overwrite in place
QBIO_TARGET      = 66          # papers needed for q-bio
QBIO_NEEDED      = 26          # how many more we need right now
YEAR_MIN         = 2000
YEAR_MAX         = 2019
REVISION_THRESHOLD = 0.20      # strictly greater than

S2_BATCH            = 100
MAX_PAGES_PER_QUERY = 10       # S2 400s past offset=1000
S2_DELAY            = 3.0      # slightly more generous than main script
ARXIV_DELAY         = 8.0      # extra breathing room for arXiv

S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = (
    "paperId,title,abstract,authors,venue,year,url,"
    "publicationVenue,externalIds,fieldsOfStudy"
)

# Fresh q-bio queries — none overlap with the v6 query list.
# 20 queries gives plenty of headroom to find 26 more papers.
# Queries are ordered from most-likely-to-yield to least, so earlier
# queries do most of the work and later ones act as fallback.
QBIO_QUERIES = [
    # High-yield: active ML/bio intersection with strong arXiv+journal presence
    "cancer genomics machine learning arxiv",
    "medical imaging deep learning arxiv",
    "clinical prediction model arxiv",
    "RNA structure prediction arxiv",
    "sequence alignment algorithm arxiv",
    "cell segmentation neural network arxiv",
    "variant calling machine learning arxiv",
    "microbiome machine learning arxiv",
    # Medium-yield
    "phylogenetics evolutionary arxiv",
    "population genetics statistical arxiv",
    "metabolic network modeling arxiv",
    "epigenomics deep learning arxiv",
    "neuroscience computational model arxiv",
    "genome assembly algorithm arxiv",
    "enzyme activity prediction arxiv",
    # Fallback: broader terms if specific ones run dry
    "biological network deep learning arxiv",
    "protein function prediction arxiv",
    "biomedical text mining arxiv",
    "EHR electronic health records machine learning arxiv",
    "molecular dynamics simulation machine learning arxiv",
]

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
    """Returns (papers, exhausted). exhausted=True on S2 400 (offset limit)."""
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
                print(f"  [S2 400] query exhausted at offset={offset}")
                return [], True
            if r.status_code >= 400:
                print(f"  [S2 error {r.status_code}] skipping page")
                return [], False
            return r.json().get("data", []), False
        except Exception as e:
            print(f"  [S2 exception] {e}")
            return [], False
    return [], False


def get_arxiv_id(paper):
    ext = paper.get("externalIds") or {}
    return ext.get("ArXiv")


def fetch_arxiv_v1(arxiv_id):
    """Fetch arXiv v1 abstract with exponential backoff on 429."""
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
            root    = ET.fromstring(r.text)
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


def ts_guessing_score(text: str):
    """Placeholder contamination probe."""
    return None


def make_row(i, version, arxiv_id, title, authors,
             abstract, venue, year, url, notes):
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
        "domain":            "q-bio",
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
# MAIN
# ─────────────────────────────────────────────

def topup_qbio():

    # ── Load existing data ────────────────────────────────────────────────────
    if os.path.exists(EXISTING_CSV):
        existing_df = pd.read_csv(EXISTING_CSV)
        print(f"Loaded {len(existing_df)} existing rows from {EXISTING_CSV}")
    else:
        existing_df = pd.DataFrame(columns=COLUMNS)
        print("No existing CSV found — starting fresh.")

    # Count current q-bio papers (count unique papers = v1 rows only)
    qbio_v1 = existing_df[
        (existing_df["arm"] == "B") &
        (existing_df["domain"] == "q-bio") &
        (existing_df["version"] == "v1")
    ]
    already_collected = len(qbio_v1)
    print(f"q-bio already collected: {already_collected}/{QBIO_TARGET}")

    if already_collected >= QBIO_TARGET:
        print("q-bio target already met. Nothing to do.")
        return existing_df

    # Seen arXiv IDs across ALL existing rows (not just q-bio) to avoid dupes
    seen = set(
        existing_df["paper_id"]
        .dropna()
        .str.replace("arXiv:", "", regex=False)
        .tolist()
    )
    print(f"Seen arXiv IDs loaded: {len(seen)}")

    # Next global index = max existing B-NNNN + 1
    b_ids = existing_df["id"].dropna().str.extract(r"B-(\d+)-")[0].dropna()
    global_i = int(b_ids.max()) + 1 if len(b_ids) else 1
    print(f"Next paper index: {global_i}")

    needed  = QBIO_TARGET - already_collected
    print(f"Papers still needed: {needed}")
    if needed != QBIO_NEEDED:
        print(f"  (Note: QBIO_NEEDED={QBIO_NEEDED} in config; "
              f"actual calculation gives {needed} — using actual)")
    new_rows = []
    count   = 0

    print(f"\n=== q-bio top-up: need {needed} more papers ===")

    for query in QBIO_QUERIES:
        if count >= needed:
            break

        print(f"\n  Query: '{query}'")
        pages = 0

        while count < needed and pages < MAX_PAGES_PER_QUERY:
            offset = pages * S2_BATCH
            pages += 1

            time.sleep(S2_DELAY)
            papers, exhausted = s2_search(query, offset)

            if exhausted:
                break   # move to next query

            if not papers:
                continue

            for p in papers:
                if count >= needed:
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
                src   = f"https://arxiv.org/abs/{arxiv_id}"

                try:
                    row_v1 = make_row(
                        global_i, "v1", arxiv_id, title, authors,
                        v1, "arXiv", year, src, notes
                    )
                    row_fn = make_row(
                        global_i, "final", arxiv_id, title, authors,
                        pub_abs, venue, year, src, notes
                    )
                except Exception as e:
                    print(f"  [schema error] {e}")
                    continue

                new_rows.append(row_v1)
                new_rows.append(row_fn)
                seen.add(arxiv_id)
                count     += 1
                global_i  += 1

                print(f"  ✓ q-bio: {already_collected+count:>3}/{QBIO_TARGET}"
                      f"  dist={dist:.2f}  {title[:50]}")

    # ── Merge and save ────────────────────────────────────────────────────────
    new_df     = pd.DataFrame(new_rows, columns=COLUMNS)
    combined   = pd.concat([existing_df, new_df], ignore_index=True)

    print(f"\n{'='*50}")
    print(f"New q-bio papers added: {count}")
    print(f"q-bio total: {already_collected + count}/{QBIO_TARGET}")
    print(f"Total rows in dataset: {len(combined)}")

    validate_dataset(combined)

    os.makedirs("data/raw", exist_ok=True)
    combined.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")

    # Quick sanity print
    summary = (
        combined[combined["version"] == "v1"]
        .groupby("domain")["id"]
        .count()
        .rename("papers")
    )
    print("\nPapers per domain (v1 rows):")
    print(summary.to_string())

    return combined


if __name__ == "__main__":
    topup_qbio()