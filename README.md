# Protein Structure Classification from PDB Data

## Purpose

Final project for CS549 Machine Learning course.

Multi-label classification of protein structures using features from the [Protein Data Bank (PDB)](https://www.rcsb.org/). Given a protein's biophysical properties (molecular weight, residue count, pH, solvent density, etc.), this pipeline predicts functional class membership across 23 categories, including hydrolase, transferase, DNA/RNA binding, and immune-related proteins.

## Motivation

PDB classification labels are stored as free-text strings (e.g., "TRANSFERASE/HYDROLASE/DNA-BINDING PROTEIN"), which makes them difficult to use directly as training labels for machine learning. A single protein can belong to multiple functional categories, so a standard single-label encoding doesn't capture the biology.

This project addresses both problems:

1. **Text parsing into structured labels:** A keyword-matching engine converts free-text classification strings into a 23-class binary target matrix, handling ambiguous cases (e.g., distinguishing "DNA binding" from generic "binding") with explicit priority logic.

2. **Multi-class classification:** Three models (Random Forest, Decision Tree, Neural Network) are trained and evaluated on the resulting labeled dataset.

## Pipeline Overview

```
PDB CSV data
    │
    ▼
[ multiclass_matrix.py ]
    Parse free-text classification labels
    Map keywords → 23 functional subclasses
    Handle special cases (binding disambiguation, DNA/RNA logic)
    Output: binary target matrix + biophysical features
    │
    ▼
[ dataprocessing.py ]
    Clean missing values
    Encode categorical features (LabelEncoder)
    Train/test split
    │
    ├──→ Random Forest (scikit-learn)
    ├──→ Decision Tree (scikit-learn)
    └──→ Neural Network (TensorFlow/Keras)
            │
            ▼
      [ Evaluation ]
      Accuracy, Precision, Recall, F1 (weighted)
```

## Functional Subclasses

The target matrix encodes membership across these 23 categories, derived from keyword frequency analysis of PDB classification text (threshold: >1,000 occurrences in the dataset):

| Category | Category | Category |
|---|---|---|
| hydrolase | transferase | oxidoreductase |
| DNA_RNA_binding | protein_binding | other_binding |
| inhibitor | transport | DNA |
| RNA | transcription | immune |
| structural | isomerase | signal |
| ligase | viral | genomics |
| metal | membrane | chaperone |
| adhesion | regulation | |

Binding disambiguation logic ensures that, for example, a protein labeled "DNA-BINDING" is assigned to `DNA_RNA_binding` rather than `other_binding`, and that the `DNA` flag is not redundantly set when `DNA_RNA_binding` already captures the relationship.

### Requirements

- Python 3.10+
- pandas
- numpy
- scikit-learn
- tensorflow

Install dependencies:

```bash
pip install pandas numpy scikit-learn tensorflow
```

### Data

This pipeline expects a CSV export from the Protein Data Bank containing at minimum these columns:

`structureId`, `classification`, `macromoleculeType`, `residueCount`, `resolution`, `structureMolecularWeight`, `densityMatthews`, `densityPercentSol`, `phValue`

PDB data can be downloaded from the [RCSB PDB search interface](https://www.rcsb.org/search).

### Usage

**Step 1: Generate the multi-class target matrix**

```python
import pandas as pd
from multiclass_matrix import create_multiclass_matrix

data = pd.read_csv('pdb_data_no_dups.csv')
labeled_data = create_multiclass_matrix(data, matrix_creation_log='False')
```

This outputs a CSV (`matrix_data_set.csv`) with binary subclass columns joined to the original biophysical features. Enable logging with `matrix_creation_log='True'` to inspect keyword matching decisions row by row.

**Step 2: Run classification**

```bash
python main.py
```

Select a CSV file when prompted, then choose a classifier (Random Forest, Decision Tree, or Neural Network) from the menu.

## Project Structure

```
protein-structure-classification/
├── main.py                  # Entry point: file selection and model menu
├── multiclass_matrix.py     # Free-text → binary target matrix conversion
├── dataprocessing.py        # Feature preparation and model training/evaluation
├── filemanager.py           # CSV file loading via file dialog
└── README.md
```

## Tools and Libraries

Python, pandas, NumPy, scikit-learn (RandomForestClassifier, DecisionTreeClassifier, LabelEncoder, train_test_split, classification metrics), TensorFlow/Keras (Sequential, Dense, Adam)

## Author

Reina Hastings - [GitHub](https://github.com/ReitheHeroine)
