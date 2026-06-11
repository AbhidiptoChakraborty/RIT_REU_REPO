import time
import requests
import pandas as pd
from collections import defaultdict

from w1_dataset_schema import (
    COLUMNS,
    validate_row,
    validate_dataset,
    infer_venue_type
)

# ---------------------------------
# Semantic Scholar API
# ---------------------------------

API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

FIELDS = ",".join([
    "paperId",
    "title",
    "abstract",
    "authors",
    "venue",
    "year",
    "url",
    "publicationVenue",
    "fieldsOfStudy",
])

API_KEY    = "s2k-Wxrcv54yu97WQtaD0lRfMcvwEd6fTWOvFgHnqF5Q"
YEAR_RANGE = "2000-2019"
LIMIT_PER_QUERY  = 100
TARGET_PER_VENUE = 33
DEBUG_UNMATCHED  = False   # flip to True to see unmatched venue strings

# ---------------------------------
# Venue buckets
# ---------------------------------

VENUE_BUCKETS = {
    # PNAS must be checked BEFORE Nature and Science.
    # The full PNAS venue string contains "sciences" which would
    # incorrectly match the Science bucket if checked first.
    "PNAS": [
        "proceedings of the national academy",
        "national academy of sciences",
        "pnas",
    ],
    "Nature":  ["nature"],
    "Science": ["science"],
    "NeurIPS": [
        "neurips",
        "neural information processing systems",
        "advances in neural information processing",
    ],
    "AAAI": [
        "aaai conference on artificial intelligence",
        "aaai",
    ],
    "ACL": [
        "annual meeting of the association for computational linguistics",
        "transactions of the association for computational linguistics",
        "association for computational linguistics",
    ],
}

# ---------------------------------
# Queries
# ---------------------------------

ARM_A_QUERIES = [

    # --- Nature ---
    ("machine learning nature",                         "Nature"),
    ("deep learning biology nature",                    "Nature"),
    ("neural network protein nature",                   "Nature"),
    ("artificial intelligence nature",                  "Nature"),
    ("quantum computing nature",                        "Nature"),
    ("genomics sequencing nature",                      "Nature"),

    # --- Science ---
    ("machine learning science journal",                "Science"),
    ("deep learning physics science",                   "Science"),
    ("neural network science journal",                  "Science"),
    ("computational biology science",                   "Science"),
    ("artificial intelligence science",                 "Science"),
    ("molecular design machine learning",               "Science"),

    # --- PNAS ---
    ("machine learning PNAS",                           "PNAS"),
    ("deep learning national academy sciences",         "PNAS"),
    ("neural network PNAS",                             "PNAS"),
    ("computational biology PNAS",                      "PNAS"),
    ("interpretable machine learning PNAS",             "PNAS"),
    ("machine learning proceedings national academy",   "PNAS"),
    ("reinforcement learning PNAS",                     "PNAS"),
    ("convolutional neural network PNAS",               "PNAS"),
    ("artificial intelligence national academy",        "PNAS"),
    ("image recognition national academy sciences",     "PNAS"),
    ("graph neural network national academy",           "PNAS"),
    ("natural language processing national academy",    "PNAS"),
    ("optimization machine learning national academy",  "PNAS"),
    ("classification algorithm national academy",       "PNAS"),
    ("protein folding machine learning PNAS",           "PNAS"),
    ("genomics deep learning national academy",         "PNAS"),

    # --- NeurIPS ---
    ("neural network optimization NeurIPS",             "NeurIPS"),
    ("generative model NeurIPS",                        "NeurIPS"),
    ("deep learning NeurIPS",                           "NeurIPS"),
    ("reinforcement learning NeurIPS",                  "NeurIPS"),
    ("attention mechanism NeurIPS",                     "NeurIPS"),
    ("variational inference NeurIPS",                   "NeurIPS"),
    ("deep learning neural information processing",     "NeurIPS"),
    ("transformer attention neural information",        "NeurIPS"),
    ("graph neural network NeurIPS 2018",               "NeurIPS"),
    ("language model neural information processing",    "NeurIPS"),
    ("image classification NeurIPS",                    "NeurIPS"),
    ("generative adversarial network NeurIPS",          "NeurIPS"),
    ("Bayesian deep learning NeurIPS",                  "NeurIPS"),
    ("recurrent neural network NeurIPS",                "NeurIPS"),
    ("meta-learning NeurIPS",                           "NeurIPS"),

    # --- AAAI ---
    ("knowledge representation AAAI",                   "AAAI"),
    ("planning artificial intelligence",                "AAAI"),
    ("natural language processing AAAI",                "AAAI"),
    ("deep learning AAAI",                              "AAAI"),
    ("knowledge graph embedding AAAI",                  "AAAI"),
    ("computer vision AAAI",                            "AAAI"),

    # --- ACL ---
    ("natural language processing ACL",                 "ACL"),
    ("machine translation ACL",                         "ACL"),
    ("sentiment analysis ACL",                          "ACL"),
    ("question answering ACL",                          "ACL"),
    ("text summarization ACL",                          "ACL"),
    ("word embeddings ACL",                             "ACL"),
    ("named entity recognition ACL",                    "ACL"),
    ("dependency parsing ACL",                          "ACL"),
    ("coreference resolution ACL",                      "ACL"),
    ("dialogue systems ACL",                            "ACL"),
    ("neural machine translation ACL 2018",             "ACL"),
    ("reading comprehension ACL",                       "ACL"),
    ("text classification ACL",                         "ACL"),
    ("information extraction ACL",                      "ACL"),
    ("semantic parsing ACL",                            "ACL"),
    ("relation extraction ACL",                         "ACL"),
]


# ---------------------------------
# Fetch
# ---------------------------------

def fetch_papers(query, limit=100, api_key=None, year_range=YEAR_RANGE):
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    params = {
        "query":  query,
        "limit":  limit,
        "fields": FIELDS,
        "year":   year_range,
    }

    for attempt in range(3):
        response = requests.get(API_URL, params=params, headers=headers)
        if response.status_code == 429:
            wait = 10 * (attempt + 1)
            print(f"  Rate limited. Waiting {wait}s (attempt {attempt+1}/3)...")
            time.sleep(wait)
            continue
        response.raise_for_status()
        return response.json().get("data", [])

    raise RuntimeError("Failed after 3 retries due to rate limiting.")


# ---------------------------------
# Venue matching
# ---------------------------------

def match_venue_bucket(paper):
    pub_venue = paper.get("publicationVenue") or {}
    venue_str = pub_venue.get("name") or paper.get("venue") or ""

    if not venue_str.strip():
        return None

    if "@" in venue_str:
        return None

    venue_lower = venue_str.lower().strip()

    for bucket, substrings in VENUE_BUCKETS.items():
        for sub in substrings:
            if sub in venue_lower:
                return bucket, venue_str

    if DEBUG_UNMATCHED:
        keywords = ["academy", "neural information", "computational linguistics",
                    "neurips", "pnas", "national academy"]
        if any(kw in venue_lower for kw in keywords):
            print(f"  [UNMATCHED VENUE] '{venue_str}'")

    return None


# ---------------------------------
# Pre-validate: check year before
# calling validate_row so we can
# log the real reason for rejection.
# ---------------------------------

def pre_filter(paper, title):
    year = paper.get("year")
    if year is None:
        print(f"  Skipped (no year): {title[:70]}")
        return False
    if year > 2019:
        print(f"  Skipped (year={year}): {title[:70]}")
        return False
    if not paper.get("abstract"):
        print(f"  Skipped (no abstract): {title[:70]}")
        return False
    return True


# ---------------------------------
# Convert
# ---------------------------------

def convert_to_schema(paper, idx, venue_str, venue_type):
    authors = ", ".join(
        author["name"] for author in paper.get("authors", [])
    )

    row = {
        "id":                f"A-{idx:04d}",
        "arm":               "A",
        "paper_id":          paper.get("paperId"),
        "title":             paper.get("title"),
        "authors":           authors,
        "abstract":          paper.get("abstract"),
        "venue":             venue_str,
        "venue_type":        venue_type,
        "year":              paper.get("year"),
        "domain":            ", ".join(paper.get("fieldsOfStudy") or []),
        "source_url":        paper.get("url"),
        "version":           "final",
        "collection_date":   "2026-06-08",
        "ts_guessing_score": None,
        "is_contaminated":   None,
        "notes":             "",
    }

    validate_row(row)
    return row


# ---------------------------------
# Main
# ---------------------------------

def main():
    all_rows      = []
    seen_ids      = set()
    global_idx    = 1
    bucket_counts = defaultdict(int)

    for query, target_bucket in ARM_A_QUERIES:

        if bucket_counts[target_bucket] >= TARGET_PER_VENUE:
            print(f"\nSkipping '{query}' — "
                  f"{target_bucket} already at {TARGET_PER_VENUE}")
            continue

        print(f"\nFetching: '{query}' → bucket [{target_bucket}] "
              f"({bucket_counts[target_bucket]}/{TARGET_PER_VENUE}) ...")

        try:
            papers = fetch_papers(
                query=query,
                limit=LIMIT_PER_QUERY,
                api_key=API_KEY,
                year_range=YEAR_RANGE,
            )
        except Exception as e:
            print(f"  Failed: {e}")
            continue

        print(f"  Got {len(papers)} results from API")

        accepted = 0
        for paper in papers:

            if bucket_counts[target_bucket] >= TARGET_PER_VENUE:
                break

            paper_id = paper.get("paperId")
            if paper_id in seen_ids:
                continue

            match = match_venue_bucket(paper)
            if match is None:
                continue

            bucket_name, venue_str = match
            if bucket_name != target_bucket:
                continue

            title = paper.get("title") or ""

            # Pre-filter: log year/abstract issues before validate_row
            if not pre_filter(paper, title):
                continue

            try:
                venue_type = infer_venue_type(venue_str)
                row = convert_to_schema(paper, global_idx,
                                        venue_str, venue_type)
                all_rows.append(row)
                seen_ids.add(paper_id)
                bucket_counts[target_bucket] += 1
                global_idx += 1
                accepted += 1

            except Exception as e:
                print(f"  Skipped (schema error) '{title[:60]}': {e!r}")

        print(f"  Accepted {accepted} | "
              f"{target_bucket} bucket: "
              f"{bucket_counts[target_bucket]}/{TARGET_PER_VENUE}")

        time.sleep(1)

    # ---------------------------
    # Summary
    # ---------------------------

    print(f"\n{'='*60}")
    print("VENUE BUCKET SUMMARY")
    print(f"{'='*60}")
    for bucket, count in sorted(bucket_counts.items()):
        status = "✓" if count >= TARGET_PER_VENUE else f"NEED {TARGET_PER_VENUE - count} MORE"
        print(f"  {bucket:<10} {count:>3}/{TARGET_PER_VENUE}  {status}")
    print(f"{'='*60}")
    print(f"  TOTAL: {len(all_rows)} papers collected")

    df = pd.DataFrame(all_rows, columns=COLUMNS)
    validate_dataset(df)

    import os
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/arm_a_sample.csv", index=False)
    print(f"\nSaved to data/raw/arm_a_sample.csv")


if __name__ == "__main__":
    main()