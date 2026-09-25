"""
High-performance Blocking module for Business Entity Resolution Challenge.
Generates candidate pairs from Source 2 and Source 3 for each Source 1 entity.
Enforces memory-efficient partitioning by Country and Token Keys.
Outputs candidate_pairs.tsv.
"""

import re
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Set, Tuple


def extract_blocking_key(clean_name: str) -> str:
    """
    Extract first 3 alphanumeric characters of clean name as a primary block key.
    """
    tokens = [t for t in clean_name.split() if len(t) >= 2]
    if tokens:
        return tokens[0][:3]
    return clean_name[:3] if len(clean_name) >= 3 else "unk"


def generate_candidates_fast(
    df_s1: pd.DataFrame,
    df_candidates: pd.DataFrame,
    top_k: int = 20,
    min_similarity: float = 0.35
) -> Dict[str, List[str]]:
    """
    Fast candidate generation partitioned by country and token prefix blocking.
    Uses TF-IDF character n-gram cosine similarity within each partition block.
    """
    candidates_dict = {s1_id: [] for s1_id in df_s1['entity_id']}
    
    # Partition by country
    countries = df_s1['clean_country'].unique()
    
    for country in countries:
        s1_country = df_s1[df_s1['clean_country'] == country]
        cand_country = df_candidates[df_candidates['clean_country'] == country]
        
        if s1_country.empty or cand_country.empty:
            continue

        # Sub-partition by primary token key to maintain low memory & fast speed
        s1_country_keys = s1_country['clean_name'].apply(extract_blocking_key)
        cand_country_keys = cand_country['clean_name'].apply(extract_blocking_key)

        unique_keys = s1_country_keys.unique()

        for key in unique_keys:
            s1_sub = s1_country[s1_country_keys == key]
            cand_sub = cand_country[cand_country_keys == key]
            
            # If block has no candidates, relax block to country-wide top matches
            if cand_sub.empty:
                cand_sub = cand_country

            s1_texts = (s1_sub['clean_name'] + " " + s1_sub['clean_address']).tolist()
            cand_texts = (cand_sub['clean_name'] + " " + cand_sub['clean_address']).tolist()

            vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 3), min_df=1)
            try:
                cand_tfidf = vectorizer.fit_transform(cand_texts)
                s1_tfidf = vectorizer.transform(s1_texts)
            except ValueError:
                continue

            n_neighbors = min(top_k, cand_tfidf.shape[0])
            nn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', algorithm='brute')
            nn.fit(cand_tfidf)

            distances, indices = nn.kneighbors(s1_tfidf)
            
            cand_ids = cand_sub['entity_id'].values
            s1_ids = s1_sub['entity_id'].values

            for i, s1_id in enumerate(s1_ids):
                # Filter candidates by cosine distance threshold (1 - similarity <= 0.65)
                valid_mask = distances[i] <= (1.0 - min_similarity)
                selected_cand_ids = cand_ids[indices[i][valid_mask]].tolist()
                candidates_dict[s1_id].extend(selected_cand_ids)

    return candidates_dict


def save_candidate_pairs(candidates_dict: Dict[str, List[str]], output_path: str):
    """
    Save candidate pairs in tab-separated candidate_pairs.tsv format:
    source1_entity_id \t candidate_entity_ids (comma-separated)
    """
    rows = []
    for s1_id, cand_ids in candidates_dict.items():
        # Preserve uniqueness while maintaining order
        unique_cands = list(dict.fromkeys(cand_ids))
        rows.append({
            'source1_entity_id': s1_id,
            'candidate_entity_ids': ','.join(unique_cands)
        })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_path, sep='\t', index=False)
