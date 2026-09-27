"""
Model training module for Business Entity Resolution Challenge.
Fast vectorized training pair generation using pandas chunk sampling.
Saves the trained model artifact to models/er_model.pkl.
"""

import os
import pickle
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.preprocessing import normalize_business_name, normalize_address, clean_text
from src.features import compute_pair_features_fast
from src.model import EntityResolutionModel, calculate_f_05
from src.blocking import extract_blocking_keys
from collections import defaultdict, Counter


def load_candidates_vectorized(train_dir: str, needed_ids: set, sample_per_country: int = 50000) -> pd.DataFrame:
    """
    Load candidate records efficiently using vectorized pandas operations.
    Returns all needed positive IDs + a capped sample per country for hard negatives.
    """
    frames = []
    for s_file in ["train_source2.tsv", "train_source3.tsv"]:
        path = os.path.join(train_dir, s_file)
        if not os.path.exists(path):
            continue
        print(f"  Reading {s_file}...")
        for chunk in pd.read_csv(path, sep='\t', chunksize=200000, dtype=str):
            chunk = chunk.fillna('')
            # Always keep positives
            pos_mask = chunk['entity_id'].isin(needed_ids)
            pos_rows = chunk[pos_mask]

            # Sample from each country group for hard negative pool
            neg_rows_list = []
            for country, grp in chunk[~pos_mask].groupby('country'):
                take = min(len(grp), max(1, sample_per_country // 20))  # per-chunk quota
                neg_rows_list.append(grp.sample(n=take, random_state=42))
            neg_rows = pd.concat(neg_rows_list) if neg_rows_list else pd.DataFrame()

            frames.append(pd.concat([pos_rows, neg_rows]))

    if not frames:
        return pd.DataFrame()

    all_cands = pd.concat(frames, ignore_index=True)
    # Deduplicate, keeping first occurrence
    all_cands = all_cands.drop_duplicates(subset='entity_id')

    # Enforce per-country cap on negatives (keep all positives)
    pos_df = all_cands[all_cands['entity_id'].isin(needed_ids)]
    neg_df = all_cands[~all_cands['entity_id'].isin(needed_ids)]
    neg_sampled = (
        neg_df.groupby('country', group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), sample_per_country), random_state=42))
    )
    result = pd.concat([pos_df, neg_sampled], ignore_index=True)
    print(f"  Candidate pool: {len(pos_df)} positives + {len(neg_sampled)} negatives = {len(result)} total")
    return result


def build_features_from_pairs(pairs: list, s1_dict: dict, cand_dict: dict) -> tuple:
    """
    Vectorized feature extraction — avoids iterrows() completely.
    Returns (X_array, y_array).
    """
    X_rows = []
    y_rows = []
    skipped = 0
    for p in pairs:
        s1 = s1_dict.get(p['s1_id'])
        cd = cand_dict.get(p['cand_id'])
        if s1 is None or cd is None:
            skipped += 1
            continue
        feats = compute_pair_features_fast(
            s1['clean_name'], cd['clean_name'],
            s1['clean_addr'], cd['clean_addr'],
            s1['country'],   cd['country']
        )
        X_rows.append(feats)
        y_rows.append(p['label'])
    if skipped:
        print(f"  Skipped {skipped} pairs with missing data.")
    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int32)


def create_training_pairs(
    train_dir: str = "dataset/train",
    sample_s1_count: int = 20000,
    random_state: int = 42
) -> tuple:
    """
    Build (X, y) training arrays efficiently.
    Returns (X np.array, y np.array, feature_names list).
    """
    np.random.seed(random_state)

    # --- Load ground truth ---
    print("Loading ground truth...")
    gt_df = pd.read_csv(os.path.join(train_dir, "train_ground_truth.tsv"), sep='\t', dtype=str)
    gt_df['matched_entity_ids'] = gt_df['matched_entity_ids'].fillna('')
    matched_gt = gt_df[gt_df['matched_entity_ids'] != '']
    matched_gt = matched_gt.sample(n=min(sample_s1_count, len(matched_gt)), random_state=random_state)
    print(f"  Using {len(matched_gt)} S1 entities with matches.")

    # Build positive pair list and collect needed candidate IDs
    positive_pairs = []
    needed_cand_ids = set()
    for _, row in matched_gt.iterrows():
        s1_id = row['source1_entity_id']
        for m_id in str(row['matched_entity_ids']).split(','):
            m_id = m_id.strip()
            if m_id:
                positive_pairs.append({'s1_id': s1_id, 'cand_id': m_id, 'label': 1})
                needed_cand_ids.add(m_id)
    print(f"  {len(positive_pairs)} positive pairs, {len(needed_cand_ids)} unique candidate IDs needed.")

    # --- Load S1 records (only selected entities) ---
    print("Loading S1 records...")
    s1_df = pd.read_csv(os.path.join(train_dir, "train_source1.tsv"), sep='\t', dtype=str).fillna('')
    s1_ids_needed = set(matched_gt['source1_entity_id'])
    s1_df = s1_df[s1_df['entity_id'].isin(s1_ids_needed)]
    s1_df['clean_name'] = s1_df['business_name'].apply(normalize_business_name)
    s1_df['clean_addr'] = s1_df['business_address'].apply(normalize_address)
    s1_df['country'] = s1_df['country'].apply(clean_text)
    s1_dict = s1_df.set_index('entity_id')[['clean_name', 'clean_addr', 'country']].to_dict('index')
    print(f"  Loaded {len(s1_dict)} S1 records.")

    # --- Load candidate records (vectorized, sampled) ---
    print("Loading candidate records (vectorized sampling)...")
    cand_df = load_candidates_vectorized(train_dir, needed_cand_ids, sample_per_country=50000)
    cand_df['clean_name'] = cand_df['business_name'].apply(normalize_business_name)
    cand_df['clean_addr'] = cand_df['business_address'].apply(normalize_address)
    cand_df['country'] = cand_df['country'].apply(clean_text)
    cand_dict = cand_df.set_index('entity_id')[['clean_name', 'clean_addr', 'country']].to_dict('index')
    print(f"  Total candidate records in pool: {len(cand_dict)}")

    # --- Hard negative mining via blocking index ---
    print("Building blocking index for hard negative mining...")
    country_index = defaultdict(lambda: defaultdict(list))
    for c_id, c_data in cand_dict.items():
        country = c_data['country']
        for k in extract_blocking_keys(c_data['clean_name'], c_data['clean_addr']):
            country_index[country][k].append(c_id)

    positive_set = {(p['s1_id'], p['cand_id']) for p in positive_pairs}
    hard_neg_pairs = []
    target_hard = len(positive_pairs) * 2

    s1_list_gt = list(matched_gt.itertuples())
    np.random.shuffle(s1_list_gt)
    for row in s1_list_gt:
        if len(hard_neg_pairs) >= target_hard:
            break
        s1_id = row.source1_entity_id
        s1 = s1_dict.get(s1_id)
        if not s1:
            continue
        idx = country_index.get(s1['country'], {})
        counter = Counter()
        for k in extract_blocking_keys(s1['clean_name'], s1['clean_addr']):
            for c_id in idx.get(k, []):
                counter[c_id] += 1
        for c_id, _ in counter.most_common(10):
            if (s1_id, c_id) not in positive_set:
                hard_neg_pairs.append({'s1_id': s1_id, 'cand_id': c_id, 'label': 0})
                if len(hard_neg_pairs) >= target_hard:
                    break

    print(f"  {len(hard_neg_pairs)} hard negative pairs.")

    # --- Easy random negatives ---
    cand_keys = np.array(list(cand_dict.keys()))
    s1_keys = np.array(list(s1_dict.keys()))
    rand_s1 = np.random.choice(s1_keys, size=len(positive_pairs) * 2)
    rand_cd = np.random.choice(cand_keys, size=len(positive_pairs) * 2)
    easy_neg_pairs = [
        {'s1_id': s, 'cand_id': c, 'label': 0}
        for s, c in zip(rand_s1, rand_cd)
        if (s, c) not in positive_set
    ][:len(positive_pairs)]
    print(f"  {len(easy_neg_pairs)} easy random negative pairs.")

    # --- Build feature matrix ---
    all_pairs = positive_pairs + hard_neg_pairs + easy_neg_pairs
    np.random.shuffle(all_pairs)
    print(f"Building feature matrix for {len(all_pairs)} pairs (no iterrows)...")
    X, y = build_features_from_pairs(all_pairs, s1_dict, cand_dict)

    # Feature names matching compute_pair_features_fast return order
    feature_names = [
        'name_ratio', 'name_partial', 'name_token_sort', 'name_token_set',
        'name_jaccard', 'name_exact',
        'addr_ratio', 'addr_partial', 'addr_token_sort', 'addr_token_set',
        'addr_jaccard', 'name_len_diff', 'addr_len_diff', 'country_match',
        'name_token_overlap', 'addr_digit_match', 'name_len_ratio', 'name_wratio'
    ]
    print(f"Feature matrix shape: {X.shape}, positives: {y.sum()}, negatives: {(y==0).sum()}")
    return X, y, feature_names


def train_matching_model(
    train_dir: str = "dataset/train",
    model_output_path: str = "models/er_model.pkl",
    sample_s1_count: int = 20000
):
    """Train and save Entity Resolution model with GradientBoosting + F_0.5 threshold tuning."""
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    t0 = time.time()

    print("=== Step 1: Building training pairs ===")
    X, y, feature_names = create_training_pairs(train_dir=train_dir, sample_s1_count=sample_s1_count)

    print(f"\n=== Step 2: Train/Val split ===")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}")

    print("\n=== Step 3: Training GradientBoosting (300 trees) ===")
    model = EntityResolutionModel(n_estimators=300)
    model.fit(X_train, y_train)

    print("\n=== Step 4: Threshold optimization for F_0.5 ===")
    best_thresh = model.optimize_threshold(X_val, y_val)

    artifact = {
        'model': model.clf,
        'best_threshold': best_thresh,
        'feature_names': feature_names
    }
    with open(model_output_path, 'wb') as f:
        pickle.dump(artifact, f)

    print(f"\nSUCCESS! Model saved to '{model_output_path}' in {time.time()-t0:.1f}s.")
    return artifact


if __name__ == "__main__":
    train_matching_model(sample_s1_count=20000)
