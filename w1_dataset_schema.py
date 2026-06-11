import pandas as pd

# -----------------------------
# W1 DATASET SCHEMA (FINAL v4)
# -----------------------------

"""
W1 DATASET SCHEMA CONTRACT

Each row represents one abstract instance (one version of one paper).

Constraints:
- id must be unique
- arm ∈ {"A", "B"}
- year ≤ 2019
- abstract must be raw text (no edits)
- version ∈ {"v1", "final"}
- venue_type ∈ {"journal", "conference", "preprint"}
- ts_guessing_score may be None initially
- is_contaminated may be None initially

Venue type mapping (substring match, case-insensitive):
- "nature", "science", "pnas"                              → "journal"
- "acl", "neurips", "neural information processing", "aaai" → "conference"
- "arxiv"                                                  → "preprint"

Do NOT modify title or abstract after ingestion.
All downstream transformations belong to later workstreams (W2+).
"""

# -----------------------------
# Column Definitions
# -----------------------------

COLUMNS = [

    # Dataset identity
    "id",                  # A-0001 ... A-0200 or B-0001 ... B-0200
    "arm",                 # A = prestige abstracts, B = arXiv v1

    # Paper metadata
    "paper_id",            # DOI / ACL Anthology ID / arXiv ID
    "title",
    "authors",
    "abstract",

    # Provenance
    "venue",
    "venue_type",          # "journal", "conference", "preprint"
    "year",
    "domain",
    "source_url",
    "version",             # v1 or final

    # Collection metadata
    "collection_date",

    # Contamination analysis
    "ts_guessing_score",
    "is_contaminated",

    # Miscellaneous
    "notes"
]

# -----------------------------
# Valid Values
# -----------------------------

VALID_ARMS         = {"A", "B"}
VALID_VERSIONS     = {"v1", "final"}
VALID_VENUE_TYPES  = {"journal", "conference", "preprint"}



# -----------------------------
# Create Empty Dataset
# -----------------------------

df = pd.DataFrame(columns=COLUMNS)

print("Schema initialized with columns:")
print(df.columns.tolist())

# Ensure column order stays fixed
df = df[COLUMNS]


# -----------------------------
# Row Validation
# -----------------------------

def validate_row(row):

    assert row["id"] is not None, \
        "Missing id"

    assert row["arm"] in VALID_ARMS, \
        f"Invalid arm: must be one of {VALID_ARMS}"

    assert row["paper_id"] is not None, \
        "Missing paper_id"

    assert row["title"] is not None, \
        "Missing title"

    assert row["abstract"] is not None, \
    "Missing abstract"

    assert isinstance(row["abstract"], str), \
        "Abstract must be a string"

    assert len(row["abstract"].strip()) > 0, \
        "Abstract cannot be empty"

    if row["year"] is None or row["year"] > 2019:
        raise ValueError("Skip: invalid year")

    assert row["version"] in VALID_VERSIONS, \
        f"Version must be one of {VALID_VERSIONS}"

    assert row["venue_type"] in VALID_VENUE_TYPES, \
        f"venue_type must be one of {VALID_VENUE_TYPES}"

    # Cross-field check: Arm B should always be preprint
    if row["arm"] == "B":
        assert row["venue_type"] == "preprint", \
            "Arm B entries must have venue_type='preprint'"

    # Cross-field check: Arm A should never be preprint
    if row["arm"] == "A":
        assert row["venue_type"] != "preprint", \
            "Arm A entries must not have venue_type='preprint'"

    return True


# -----------------------------
# Dataset Validation
# -----------------------------

def validate_dataset(df):

    print("Running dataset validation...")

    assert df["id"].is_unique, \
        "Duplicate IDs found"

    assert df["arm"].isin(VALID_ARMS).all(), \
        "Invalid arm values"

    assert df["venue_type"].isin(VALID_VENUE_TYPES).all(), \
        "Invalid venue_type values"

    if len(df) > 0:
        assert df["year"].le(2019).all(), \
            "Year constraint violated"

        # Cross-field: Arm B must all be preprint
        arm_b = df[df["arm"] == "B"]
        assert (arm_b["venue_type"] == "preprint").all(), \
            "All Arm B rows must have venue_type='preprint'"

        # Cross-field: Arm A must never be preprint
        arm_a = df[df["arm"] == "A"]
        assert (arm_a["venue_type"] != "preprint").all(), \
            "Arm A rows must not have venue_type='preprint'"

    print("Dataset validation PASSED")


# -----------------------------
# Dataset Factory
# -----------------------------

def create_empty_dataset():
    """
    Returns a new empty dataframe following the W1 schema.
    """
    return pd.DataFrame(columns=COLUMNS)


# -----------------------------
# Helper: Infer venue_type from venue string
# -----------------------------

def infer_venue_type(venue: str) -> str:
    if venue is None or venue.strip() == "":
        raise ValueError("Venue is missing or empty")
    
    venue_lower = venue.lower()
    
    # Journals
    if "nature" in venue_lower:
        return "journal"
    if "science" in venue_lower:
        return "journal"
    if "pnas" in venue_lower:
        return "journal"
    if "journal" in venue_lower:
        return "journal"
    if "transactions" in venue_lower:
        return "journal"
    if "trans." in venue_lower:
        return "journal"
    if "survey" in venue_lower:
        return "journal"
    if "review" in venue_lower:
        return "journal"

    # Conferences
    if "acl" in venue_lower:
        return "conference"
    if "computational linguistics" in venue_lower:
        return "conference"
    if "neurips" in venue_lower:
        return "conference"
    if "neural information processing systems" in venue_lower:
        return "conference"
    if "aaai" in venue_lower:
        return "conference"
    if "conference" in venue_lower:
        return "conference"
    if "workshop" in venue_lower:
        return "conference"
    if "proceedings" in venue_lower:
        return "conference"
    if "symposium" in venue_lower:
        return "conference"

    # Preprints
    if "arxiv" in venue_lower:
        return "preprint"

    raise ValueError(f"Unknown venue: '{venue}'")

# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":

    validate_dataset(df)

    # Smoke test: verify infer_venue_type handles real-world strings
    test_cases = [
        ("Nature",                                "journal"),
        ("Nature Communications",                 "journal"),
        ("Science",                               "journal"),
        ("PNAS",                                  "journal"),
        ("ACL",                                   "conference"),
        ("NeurIPS",                               "conference"),
        ("Advances in Neural Information Processing Systems", "conference"),
        ("AAAI",                                  "conference"),
        ("arXiv",                                 "preprint"),
        ("arXiv:2301.00001",                      "preprint"),
    ]

    print("\nRunning infer_venue_type smoke tests...")
    for venue_str, expected in test_cases:
        result = infer_venue_type(venue_str)
        status = "PASS" if result == expected else "FAIL"
        print(f"  [{status}] '{venue_str}' → '{result}'")

    print("\nW1 schema file is ready.")