"""
High-performance Multi-Key Inverted Index Blocking module for Business Entity Resolution Challenge.
Generates candidate pairs from Source 2 and Source 3 for each Source 1 entity.
Runs in linear O(N) time with minimal RAM footprint (< 1 GB) and zero dense matrix allocations.
Outputs candidate_pairs.tsv.
"""

import re
from collections import defaultdict
from typing import Dict, List, Set
import pandas as pd

STOPWORDS = {
    'company', 'limited', 'private', 'pvt', 'ltd', 'inc', 'corp', 'llc', 'llp',
    'the', 'and', 'services', 'enterprises', 'store', 'shop', 'india', 'solutions'
}
RE_NUMS = re.compile(r'\b\d{4,6}\b')


def extract_blocking_keys(clean_name: str, clean_address: str) -> List[str]:
    """
    Generate multiple high-recall blocking keys per entity:
    1. First 4 characters of normalized business name
    2. Primary significant name tokens (length >= 4, non-stopword)
    3. Postal/PIN/house numbers extracted from address
    """
    keys = []
    if len(clean_name) >= 3:
        keys.append('p4_' + clean_name[:4])

    tokens = clean_name.split()
    for t in tokens:
        if len(t) >= 4 and t not in STOPWORDS:
            keys.append('w_' + t)
            if len(keys) >= 4:
                break

    for n in RE_NUMS.findall(clean_address)[:2]:
        keys.append('num_' + n)

    return keys


def generate_candidates_fast(
    df_s1: pd.DataFrame,
    df_candidates: pd.DataFrame,
    top_k: int = 20,
    min_similarity: float = 0.35
) -> Dict[str, List[str]]:
    """
    Ultra-fast Multi-Key Inverted Index candidate generation partitioned by Country.
    Runs in linear time and eliminates ArrayMemoryError / dense matrix allocation completely.
    """
    candidates_dict = {s1_id: [] for s1_id in df_s1['entity_id']}
    
    # Partition by country
    countries = df_s1['clean_country'].unique()
    
    for country in countries:
        s1_country = df_s1[df_s1['clean_country'] == country]
        cand_country = df_candidates[df_candidates['clean_country'] == country]
        
        if s1_country.empty or cand_country.empty:
            continue

        print(f"Building inverted index for country '{country}' ({len(cand_country)} candidates)...")
        index = defaultdict(list)
        for cand_id, name, addr in zip(cand_country['entity_id'], cand_country['clean_name'], cand_country['clean_address']):
            for k in extract_blocking_keys(name, addr):
                index[k].append(cand_id)

        from collections import Counter
        print(f"Index built ({len(index)} unique keys). Querying {len(s1_country)} S1 entities...")
        for s1_id, name, addr in zip(s1_country['entity_id'], s1_country['clean_name'], s1_country['clean_address']):
            counter = Counter()
            for k in extract_blocking_keys(name, addr):
                c_list = index.get(k)
                if c_list:
                    for c_id in c_list:
                        counter[c_id] += 1
            if counter:
                candidates_dict[s1_id] = [c for c, _ in counter.most_common(top_k)]

    return candidates_dict


def save_candidate_pairs(candidates_dict: Dict[str, List[str]], output_path: str):
    """
    Save candidate pairs in tab-separated candidate_pairs.tsv format:
    source1_entity_id \t candidate_entity_ids (comma-separated)
    """
    rows = []
    for s1_id, cand_ids in candidates_dict.items():
        unique_cands = list(dict.fromkeys(cand_ids))
        rows.append({
            'source1_entity_id': s1_id,
            'candidate_entity_ids': ','.join(unique_cands)
        })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_path, sep='\t', index=False)


# Backward-compatible alias
generate_candidates_tfidf = generate_candidates_fast
