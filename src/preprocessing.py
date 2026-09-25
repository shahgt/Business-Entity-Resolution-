"""
Preprocessing module for Business Entity Resolution Challenge.
Cleans business names, addresses, and country labels.
"""

import re
import pandas as pd

# Standard abbreviation replacements for business suffixes & terms
NAME_REPLACEMENTS = [
    (r'\bcorp(oration)?\b', 'corp'),
    (r'\bpvt\b', 'private'),
    (r'\bltd\b', 'limited'),
    (r'\binc(orporated)?\b', 'inc'),
    (r'\bco(mpany)?\b', 'company'),
    (r'\bllc\b', 'llc'),
    (r'\bllp\b', 'llp'),
    (r'\bd/?b/?a\b', 'dba'),
    (r'\b&', ' and '),
]

# Standard address abbreviations
ADDRESS_REPLACEMENTS = [
    (r'\brd\.?\b', 'road'),
    (r'\bst\.?\b', 'street'),
    (r'\bave\.?\b', 'avenue'),
    (r'\bdr\.?\b', 'drive'),
    (r'\bste\.?\b', 'suite'),
    (r'\bblvd\.?\b', 'boulevard'),
    (r'\bapt\.?\b', 'apartment'),
    (r'\b&', ' and '),
]


def clean_text(text: str) -> str:
    """Basic text normalization."""
    if not isinstance(text, str) or pd.isna(text):
        return ""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def normalize_business_name(name: str) -> str:
    """Normalize business name by cleaning punctuation and standardizing common suffixes."""
    cleaned = clean_text(name)
    for pattern, replacement in NAME_REPLACEMENTS:
        cleaned = re.sub(pattern, replacement, cleaned)
    return re.sub(r'\s+', ' ', cleaned).strip()


def normalize_address(address: str) -> str:
    """Normalize business address by standardizing street/suite abbreviations."""
    cleaned = clean_text(address)
    for pattern, replacement in ADDRESS_REPLACEMENTS:
        cleaned = re.sub(pattern, replacement, cleaned)
    return re.sub(r'\s+', ' ', cleaned).strip()


def load_and_preprocess_tsv(file_path: str) -> pd.DataFrame:
    """
    Load TSV dataset file with tab separator and add cleaned feature columns.
    """
    df = pd.read_csv(file_path, sep='\t', dtype=str)
    
    # Fill missing string values
    for col in ['business_name', 'business_address', 'country']:
        if col in df.columns:
            df[col] = df[col].fillna('')

    if 'business_name' in df.columns:
        df['clean_name'] = df['business_name'].apply(normalize_business_name)
    if 'business_address' in df.columns:
        df['clean_address'] = df['business_address'].apply(normalize_address)
    if 'country' in df.columns:
        df['clean_country'] = df['country'].apply(clean_text)

    return df
