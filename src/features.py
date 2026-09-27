"""
Feature extraction module for Business Entity Resolution Challenge.
Computes similarity metrics between Source 1 entities and candidate records.
"""

from typing import Dict
import pandas as pd
import numpy as np
from rapidfuzz import fuzz


def jaccard_similarity(s1: str, s2: str) -> float:
    """Compute token Jaccard similarity."""
    set1, set2 = set(s1.split()), set(s2.split())
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def compute_pair_features(row: pd.Series) -> Dict[str, float]:
    """
    Compute fine-grained similarity features for a pair of records.
    Expects columns: clean_name_s1, clean_name_cand, clean_address_s1, clean_address_cand, clean_country_s1, clean_country_cand.
    """
    name1, name2 = str(row.get('clean_name_s1', '')), str(row.get('clean_name_cand', ''))
    addr1, addr2 = str(row.get('clean_address_s1', '')), str(row.get('clean_address_cand', ''))
    country1, country2 = str(row.get('clean_country_s1', '')), str(row.get('clean_country_cand', ''))

    # Name similarities
    name_ratio = fuzz.ratio(name1, name2) / 100.0
    name_partial = fuzz.partial_ratio(name1, name2) / 100.0
    name_token_sort = fuzz.token_sort_ratio(name1, name2) / 100.0
    name_token_set = fuzz.token_set_ratio(name1, name2) / 100.0
    name_jaccard = jaccard_similarity(name1, name2)
    name_exact = 1.0 if (name1 == name2 and name1 != '') else 0.0

    # Address similarities
    addr_ratio = fuzz.ratio(addr1, addr2) / 100.0
    addr_partial = fuzz.partial_ratio(addr1, addr2) / 100.0
    addr_token_sort = fuzz.token_sort_ratio(addr1, addr2) / 100.0
    addr_token_set = fuzz.token_set_ratio(addr1, addr2) / 100.0
    addr_jaccard = jaccard_similarity(addr1, addr2)

    # Length features
    name_len_diff = abs(len(name1) - len(name2))
    addr_len_diff = abs(len(addr1) - len(addr2))

    # Country match
    country_match = 1.0 if (country1 == country2 and country1 != '') else 0.0

    return {
        'name_ratio': name_ratio,
        'name_partial': name_partial,
        'name_token_sort': name_token_sort,
        'name_token_set': name_token_set,
        'name_jaccard': name_jaccard,
        'name_exact': name_exact,
        'addr_ratio': addr_ratio,
        'addr_partial': addr_partial,
        'addr_token_sort': addr_token_sort,
        'addr_token_set': addr_token_set,
        'addr_jaccard': addr_jaccard,
        'name_len_diff': float(name_len_diff),
        'addr_len_diff': float(addr_len_diff),
        'country_match': country_match
    }



def compute_pair_features_fast(name1: str, name2: str, addr1: str, addr2: str, country1: str, country2: str) -> list:
    """
    Direct ultra-fast C-speed feature extraction avoiding pandas Series/dict overhead.
    Returns 14-dimensional feature vector in exact model column order.
    """
    nr = fuzz.ratio(name1, name2) / 100.0
    np_ = fuzz.partial_ratio(name1, name2) / 100.0
    nts = fuzz.token_sort_ratio(name1, name2) / 100.0
    nte = fuzz.token_set_ratio(name1, name2) / 100.0

    w1, w2 = set(name1.split()), set(name2.split())
    nj = len(w1 & w2) / len(w1 | w2) if w1 and w2 else 0.0
    ne = 1.0 if name1 == name2 and name1 != '' else 0.0

    ar = fuzz.ratio(addr1, addr2) / 100.0 if addr1 and addr2 else 0.0
    ap = fuzz.partial_ratio(addr1, addr2) / 100.0 if addr1 and addr2 else 0.0
    ats = fuzz.token_sort_ratio(addr1, addr2) / 100.0 if addr1 and addr2 else 0.0
    ate = fuzz.token_set_ratio(addr1, addr2) / 100.0 if addr1 and addr2 else 0.0

    aw1, aw2 = set(addr1.split()), set(addr2.split())
    aj = len(aw1 & aw2) / len(aw1 | aw2) if aw1 and aw2 else 0.0

    nld = float(abs(len(name1) - len(name2)))
    ald = float(abs(len(addr1) - len(addr2)))
    cm = 1.0 if country1 == country2 and country1 != '' else 0.0

    return [nr, np_, nts, nte, nj, ne, ar, ap, ats, ate, aj, nld, ald, cm]


def extract_features_df(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract feature DataFrame from paired record rows.
    """
    feature_rows = pairs_df.apply(compute_pair_features, axis=1)
    return pd.DataFrame(list(feature_rows))

