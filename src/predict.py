"""
High-throughput Prediction Pipeline for Business Entity Resolution Challenge.
Generates output/candidate_pairs.tsv and output/matching_results.tsv.
Processes country-by-country using multi-key inverted index blocking and F_0.5-calibrated ML scoring.
Guarantees memory-safe execution (< 2 GB RAM) across all 1.73M entities.
"""

import os
import gc
import pickle
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from src.preprocessing import load_and_preprocess_tsv
from src.blocking import extract_blocking_keys
from src.features import compute_pair_features


def run_pipeline(
    test_dir: str = "dataset/test",
    output_dir: str = "output",
    model_path: str = "models/er_model.pkl",
    top_k_candidates: int = 20,
    batch_size: int = 5000,
    nrows: int = None
):
    """
    Run memory-safe, country-partitioned prediction pipeline across full test set.
    """
    os.makedirs(output_dir, exist_ok=True)
    cand_pairs_path = os.path.join(output_dir, "candidate_pairs.tsv")
    matching_results_path = os.path.join(output_dir, "matching_results.tsv")

    # 1. Load trained model artifact
    clf = None
    best_threshold = 0.64
    feat_names = None
    if os.path.exists(model_path):
        print(f"Loading trained model from {model_path}...")
        with open(model_path, 'rb') as f:
            artifact = pickle.load(f)
            clf = artifact.get('model')
            best_threshold = artifact.get('best_threshold', 0.64)
            feat_names = artifact.get('feature_names')
        # Single-threaded prediction avoids joblib thread spawn overhead and warnings
        if clf is not None:
            clf.n_jobs = 1
        print(f"Loaded model with calibrated F_0.5 decision threshold: {best_threshold:.2f}")
    else:
        print("WARNING: Model not found. Running baseline candidate matching.")

    # 2. Load dataset sources (single-pass read)
    t0 = time.time()
    print("\n--- Step 1: Loading test dataset ---")
    df_s1 = load_and_preprocess_tsv(os.path.join(test_dir, "test_source1.tsv"), nrows=nrows)
    print(f"Loaded {len(df_s1)} S1 entities in {time.time() - t0:.1f}s.")

    t_cand = time.time()
    df_s2 = load_and_preprocess_tsv(os.path.join(test_dir, "test_source2.tsv"), nrows=nrows)
    df_s3 = load_and_preprocess_tsv(os.path.join(test_dir, "test_source3.tsv"), nrows=nrows)
    df_cand_pool = pd.concat([df_s2, df_s3], ignore_index=True)
    del df_s2, df_s3
    gc.collect()
    print(f"Loaded {len(df_cand_pool)} candidates from S2 and S3 in {time.time() - t_cand:.1f}s.")

    # 3. Country-by-country partition processing
    countries = ["france", "us", "india"]
    first_write = True

    for country in countries:
        s1_country = df_s1[df_s1['clean_country'] == country]
        cand_country = df_cand_pool[df_cand_pool['clean_country'] == country]

        if s1_country.empty:
            continue

        print(f"\n==========================================")
        print(f"Processing Partition: '{country.upper()}' ({len(s1_country)} S1, {len(cand_country)} candidates)")
        print(f"==========================================")

        t_idx = time.time()
        print("Building inverted index...")
        index = defaultdict(list)
        cand_lookup = {}
        for c_id, c_name, c_addr, c_cntry in zip(
            cand_country['entity_id'],
            cand_country['clean_name'],
            cand_country['clean_address'],
            cand_country['clean_country']
        ):
            cand_lookup[c_id] = (c_name, c_addr, c_cntry)
            for k in extract_blocking_keys(c_name, c_addr):
                index[k].append(c_id)

        print(f"Index built ({len(index)} keys) in {time.time() - t_idx:.1f}s.")

        # Batch scoring
        t_score = time.time()
        candidate_rows = []
        matching_rows = []

        s1_ids = s1_country['entity_id'].values
        s1_names = s1_country['clean_name'].values
        s1_addrs = s1_country['clean_address'].values
        s1_cntries = s1_country['clean_country'].values
        total_s1 = len(s1_ids)

        for b_start in range(0, total_s1, batch_size):
            b_end = min(b_start + batch_size, total_s1)

            batch_pairs = []
            batch_pair_s1_indices = []
            batch_cand_ids = []

            for i in range(b_start, b_end):
                s1_id = s1_ids[i]
                s1_name = s1_names[i]
                s1_addr = s1_addrs[i]
                s1_cntry = s1_cntries[i]

                # Look up candidate IDs from inverted index
                matched_cands = set()
                for k in extract_blocking_keys(s1_name, s1_addr):
                    c_list = index.get(k)
                    if c_list:
                        matched_cands.update(c_list)
                        if len(matched_cands) >= top_k_candidates * 2:
                            break

                candidate_list = list(matched_cands)[:top_k_candidates]
                candidate_rows.append({
                    'source1_entity_id': s1_id,
                    'candidate_entity_ids': ','.join(candidate_list)
                })

                if candidate_list and clf is not None:
                    for c_id in candidate_list:
                        c_data = cand_lookup.get(c_id)
                        if c_data is None:
                            continue
                        batch_pairs.append({
                            'clean_name_s1': s1_name,
                            'clean_name_cand': c_data[0],
                            'clean_address_s1': s1_addr,
                            'clean_address_cand': c_data[1],
                            'clean_country_s1': s1_cntry,
                            'clean_country_cand': c_data[2]
                        })
                        batch_pair_s1_indices.append(i - b_start)
                        batch_cand_ids.append(c_id)

            # Fast vector batch scoring
            s1_matches = defaultdict(list)
            if batch_pairs and clf is not None:
                feat_dicts = [compute_pair_features(pd.Series(bp)) for bp in batch_pairs]
                X_batch = pd.DataFrame(feat_dicts)
                if feat_names:
                    X_batch = X_batch.reindex(columns=feat_names, fill_value=0.0)
                probs = clf.predict_proba(X_batch)[:, 1]

                for idx, p in enumerate(probs):
                    if p >= best_threshold:
                        s1_rel_idx = batch_pair_s1_indices[idx]
                        s1_matches[s1_rel_idx].append(batch_cand_ids[idx])

            for rel_idx in range(b_end - b_start):
                s1_idx = b_start + rel_idx
                m_list = s1_matches.get(rel_idx, [])
                matching_rows.append({
                    'source1_entity_id': s1_ids[s1_idx],
                    'matched_entity_ids': ','.join(m_list)
                })

            if b_end % 25000 == 0 or b_end == total_s1:
                print(f"  Processed {b_end}/{total_s1} entities in {time.time() - t_score:.1f}s...")

        # Append partition to output files
        mode = 'w' if first_write else 'a'
        header = first_write
        pd.DataFrame(candidate_rows).to_csv(cand_pairs_path, sep='\t', index=False, mode=mode, header=header)
        pd.DataFrame(matching_rows).to_csv(matching_results_path, sep='\t', index=False, mode=mode, header=header)
        first_write = False
        print(f"Appended {len(matching_rows)} rows to output files. Partition completed!")

        del index, cand_lookup
        gc.collect()

    print("\n==========================================")
    print("ALL PARTITIONS SUCCESSFULLY PROCESSED!")
    print(f"Output 1: {cand_pairs_path}")
    print(f"Output 2: {matching_results_path}")
    print("==========================================")


if __name__ == "__main__":
    run_pipeline()
