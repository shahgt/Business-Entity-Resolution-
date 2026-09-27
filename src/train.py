"""
Model training module for Business Entity Resolution Challenge.
Generates balanced positive and hard negative pairs from ground truth and training sources,
extracts similarity features, trains the classifier, and optimizes the decision threshold for F_0.5.
Saves the trained model artifact to models/er_model.pkl.
"""

import os
import pickle
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.preprocessing import normalize_business_name, normalize_address, clean_text
from src.features import extract_features_df
from src.model import EntityResolutionModel, calculate_f_05


def create_training_pairs(
    train_dir: str = "dataset/train",
    sample_s1_count: int = 15000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Construct labeled (1/0) pair examples using ground truth and source files.
    """
    print(f"Loading ground truth sample ({sample_s1_count} records)...")
    gt_df = pd.read_csv(os.path.join(train_dir, "train_ground_truth.tsv"), sep='\t', nrows=sample_s1_count * 2)
    gt_df['matched_entity_ids'] = gt_df['matched_entity_ids'].fillna('')
    
    # Filter for entities with matches and singletons
    matched_gt = gt_df[gt_df['matched_entity_ids'] != ''].head(sample_s1_count)
    selected_s1_ids = set(matched_gt['source1_entity_id'])

    # Build positive pairs
    positive_pairs = []
    needed_cand_ids = set()
    for _, row in matched_gt.iterrows():
        s1_id = row['source1_entity_id']
        matches = [m.strip() for m in row['matched_entity_ids'].split(',') if m.strip()]
        for m_id in matches:
            positive_pairs.append({'s1_id': s1_id, 'cand_id': m_id, 'label': 1})
            needed_cand_ids.add(m_id)

    print(f"Generated {len(positive_pairs)} positive pairs across {len(selected_s1_ids)} S1 entities.")

    # Load S1 records
    print("Loading S1 records...")
    s1_df = pd.read_csv(os.path.join(train_dir, "train_source1.tsv"), sep='\t')
    s1_dict = s1_df.set_index('entity_id').to_dict('index')

    # Load S2 and S3 needed candidate records
    print("Loading candidate records from S2 and S3...")
    cand_records = {}
    for s_file in ["train_source2.tsv", "train_source3.tsv"]:
        path = os.path.join(train_dir, s_file)
        for chunk in pd.read_csv(path, sep='\t', chunksize=250000):
            found = chunk[chunk['entity_id'].isin(needed_cand_ids)]
            for _, r in found.iterrows():
                cand_records[r['entity_id']] = dict(r)
            if len(cand_records) >= len(needed_cand_ids):
                break

    print(f"Retrieved {len(cand_records)} true candidate records.")

    # Generate hard negative pairs by pairing S1 with candidates of different S1 in the same country
    print("Generating balanced hard negative pairs...")
    negative_pairs = []
    cand_keys = list(cand_records.keys())
    np.random.seed(random_state)
    
    # Generate 1.5x negative pairs for class balance & robustness against false merges
    num_negatives = int(len(positive_pairs) * 1.5)
    s1_list = list(selected_s1_ids)
    
    pos_set = {(p['s1_id'], p['cand_id']) for p in positive_pairs}
    while len(negative_pairs) < num_negatives and len(cand_keys) > 0:
        rand_s1 = np.random.choice(s1_list)
        rand_cand = np.random.choice(cand_keys)
        if (rand_s1, rand_cand) not in pos_set:
            negative_pairs.append({'s1_id': rand_s1, 'cand_id': rand_cand, 'label': 0})

    print(f"Generated {len(negative_pairs)} negative pairs.")

    all_pairs = positive_pairs + negative_pairs
    np.random.shuffle(all_pairs)

    # Build feature dataframe rows
    records = []
    for p in all_pairs:
        s1_data = s1_dict.get(p['s1_id'])
        cand_data = cand_records.get(p['cand_id'])
        if s1_data is None or cand_data is None:
            continue

        records.append({
            'clean_name_s1': normalize_business_name(str(s1_data.get('business_name', ''))),
            'clean_name_cand': normalize_business_name(str(cand_data.get('business_name', ''))),
            'clean_address_s1': normalize_address(str(s1_data.get('business_address', ''))),
            'clean_address_cand': normalize_address(str(cand_data.get('business_address', ''))),
            'clean_country_s1': clean_text(str(s1_data.get('country', ''))),
            'clean_country_cand': clean_text(str(cand_data.get('country', ''))),
            'label': p['label']
        })

    return pd.DataFrame(records)


def train_matching_model(
    train_dir: str = "dataset/train",
    model_output_path: str = "models/er_model.pkl",
    sample_s1_count: int = 8000
):
    """
    Train and save Entity Resolution model with tuned decision threshold.
    """
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    t0 = time.time()
    print("=== Step 1: Generating training and negative pairs ===")
    pairs_df = create_training_pairs(train_dir=train_dir, sample_s1_count=sample_s1_count)
    
    print(f"\n=== Step 2: Extracting similarity features for {len(pairs_df)} pairs ===")
    y = pairs_df['label'].values
    X = extract_features_df(pairs_df)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train)} pairs, Val size: {len(X_val)} pairs.")

    print("\n=== Step 3: Fitting Entity Resolution Classifier ===")
    model = EntityResolutionModel(n_estimators=100)
    model.fit(X_train, y_train)

    print("\n=== Step 4: Optimizing Threshold for F_0.5 Metric ===")
    best_thresh = model.optimize_threshold(X_val, y_val)

    # Save model artifact
    artifact = {
        'model': model.clf,
        'best_threshold': best_thresh,
        'feature_names': list(X.columns)
    }
    with open(model_output_path, 'wb') as f:
        pickle.dump(artifact, f)

    elapsed = time.time() - t0
    print(f"\nSUCCESS! Model saved to {model_output_path} in {elapsed:.1f}s.")
    return artifact


if __name__ == "__main__":
    train_matching_model(sample_s1_count=6000)
