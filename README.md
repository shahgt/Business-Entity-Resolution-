# Business Entity Resolution Challenge (Amazon ML Challenge)

## Overview
Entity Resolution (ER) pipeline designed to identify matching business records across 3 independent data sources (Source 1, Source 2, Source 3) under noisy and inconsistent names and addresses.

## Project Structure
```text
amazon-ml-challenge/
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── src/
│   ├── preprocessing.py
│   ├── blocking.py
│   ├── features.py
│   ├── model.py
│   └── predict.py
├── notebooks/
├── output/
│   ├── candidate_pairs.tsv
│   └── matching_results.tsv
├── utils/
│   └── validate_submission.py
├── Documentation_template.md
├── package_submission.py
├── requirements.txt
└── README.md
```

## Dataset Summary Statistics

### Training Set
- `train_source1.tsv`: 2,206,821 records (Deduplicated reference source)
- `train_source2.tsv`: 5,034,616 records (168,967 missing addresses)
- `train_source3.tsv`: 5,285,603 records (175,916 missing addresses)
- `train_ground_truth.tsv`: 2,206,821 rows (123,247 singletons with 0 matches)
- **Countries**: US (60%), India (40%)

### Test Set
- `test_source1.tsv`: 1,732,544 records
- `test_source2.tsv`: 4,887,273 records
- `test_source3.tsv`: 5,082,316 records
- **Countries**: India (47%), US (38%), France (15% - open-set country)

## Installation & Setup

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Run prediction pipeline:
```bash
python -m src.predict
```

3. Validate output files:
```bash
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

4. Build final submission package:
```bash
python package_submission.py <your_team_name>
```
Produces `<your_team_name>_submission.zip` matching the required contest directory structure.
