"""
Helper script to package final submission zip archive according to Amazon ML Challenge requirements.
Structure:
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── preprocessing.py
│       │   ├── blocking.py
│       │   ├── features.py
│       │   ├── model.py
│       │   └── predict.py
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
"""

import os
import zipfile
import sys


def build_submission_zip(team_name: str = "Team_Solution", output_zip_path: str = None):
    if output_zip_path is None:
        output_zip_path = f"{team_name}_submission.zip"

    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Required files to include
    files_to_zip = [
        # Output files
        ("output/matching_results.tsv", "output/matching_results.tsv"),
        ("output/candidate_pairs.tsv", "output/candidate_pairs.tsv"),
        
        # Code files under code/business_entity_resolution/
        ("src/preprocessing.py", "code/business_entity_resolution/src/preprocessing.py"),
        ("src/blocking.py", "code/business_entity_resolution/src/blocking.py"),
        ("src/features.py", "code/business_entity_resolution/src/features.py"),
        ("src/model.py", "code/business_entity_resolution/src/model.py"),
        ("src/train.py", "code/business_entity_resolution/src/train.py"),
        ("src/predict.py", "code/business_entity_resolution/src/predict.py"),
        ("README.md", "code/business_entity_resolution/README.md"),
        ("requirements.txt", "code/business_entity_resolution/requirements.txt"),
        
        # Methodology document
        ("Documentation_template.md", "Documentation_template.md"),
    ]

    print(f"Creating submission zip: {output_zip_path}...")
    missing_files = []
    
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for src_rel, zip_rel in files_to_zip:
            src_full = os.path.join(root_dir, src_rel)
            if os.path.exists(src_full):
                zipf.write(src_full, zip_rel)
                print(f"  + Added {zip_rel}")
            else:
                missing_files.append(src_rel)

    if missing_files:
        print("\nWARNING: The following expected files were missing and not included in the zip:")
        for mf in missing_files:
            print(f"  - {mf}")
    else:
        print(f"\nSUCCESS! Submission package {output_zip_path} created cleanly.")


if __name__ == "__main__":
    team = sys.argv[1] if len(sys.argv) > 1 else "Team_Solution"
    build_submission_zip(team)
