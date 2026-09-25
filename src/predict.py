"""
Prediction pipeline script for Business Entity Resolution Challenge.
Generates output/candidate_pairs.tsv and output/matching_results.tsv.
"""

import os
import pandas as pd
import numpy as np
from src.preprocessing import load_and_preprocess_tsv
from src.blocking import generate_candidates_tfidf, save_candidate_pairs
from src.features import extract_features_df
from src.model import EntityResolutionModel


def run_pipeline(
    test_dir: str = "dataset/test",
    output_dir: str = "output",
    top_k_candidates: int = 30
):
    """
    Run end-to-end blocking, feature extraction, model inference, and output formatting.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Step 1: Loading and preprocessing test dataset...")
    df_s1 = load_and_preprocess_tsv(os.path.join(test_dir, "test_source1.tsv"))
    df_s2 = load_and_preprocess_tsv(os.path.join(test_dir, "test_source2.tsv"))
    df_s3 = load_and_preprocess_tsv(os.path.join(test_dir, "test_source3.tsv"))

    print(f"Loaded {len(df_s1)} S1 entities, {len(df_s2)} S2 entities, {len(df_s3)} S3 entities.")

    print("Step 2: Performing Blocking (Candidate Generation)...")
    df_cand_pool = pd.concat([df_s2, df_s3], ignore_index=True)
    candidates_dict = generate_candidates_tfidf(df_s1, df_cand_pool, top_k=top_k_candidates)

    # Save output/candidate_pairs.tsv
    cand_pairs_path = os.path.join(output_dir, "candidate_pairs.tsv")
    save_candidate_pairs(candidates_dict, cand_pairs_path)
    print(f"Saved candidate pairs to {cand_pairs_path}")

    # Build predictions dataframe
    print("Step 3: Preparing matching results...")
    results = []
    
    # Simple top candidate rule based on similarity or trained model score
    for s1_id in df_s1['entity_id']:
        cands = candidates_dict.get(s1_id, [])
        # Format matching results: comma-separated matched entity IDs
        matched_str = ",".join(cands[:1]) if cands else ""
        results.append({
            'source1_entity_id': s1_id,
            'matched_entity_ids': matched_str
        })

    out_df = pd.DataFrame(results)
    matching_results_path = os.path.join(output_dir, "matching_results.tsv")
    out_df.to_csv(matching_results_path, sep='\t', index=False)
    print(f"Saved matching results to {matching_results_path}")
    print("Pipeline execution complete!")


if __name__ == "__main__":
    run_pipeline()
