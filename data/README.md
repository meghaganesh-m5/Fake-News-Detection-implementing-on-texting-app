# Dataset & Augmentation Pipeline

This folder contains the training, validation, and test datasets used to fine-tune the misinformation 
detection model, along with the methodology used to build them.

## Source Dataset

Built on **LIAR2** (`chengxuphd/liar2`), an updated version of the LIAR dataset containing ~23,000 
real political statements fact-checked by professional journalists at PolitiFact, each rated on a 
6-point truth scale (pants-fire, false, barely-true, half-true, mostly-true, true).

## Label Simplification

The 6-point scale was collapsed into binary labels for classification:
- **Fake** = pants-fire, false, barely-true
- **Real** = mostly-true, true
- **half-true** was dropped as an ambiguous middle category — a deliberate scoping decision to avoid 
  injecting label noise into training.

## The Core Problem: Domain Mismatch

Existing fake-news datasets like LIAR2 consist entirely of formal, journalist-style statements. Real-world 
misinformation, however, spreads through informal, emoji-heavy, ALL-CAPS WhatsApp-style forwards — a 
completely different writing style. A model trained only on formal text risks failing to generalize to 
how misinformation actually appears in the wild.

## Augmentation Strategy

Two complementary techniques were used to bridge this gap:

1. **Rule-based augmentation** — A custom transformation function applied to 40% of training data, adding 
   urgency phrases ("SHARE BEFORE DELETED"), "Fwd:" prefixes, emoji clusters, and randomized caps emphasis 
   to simulate real forward-style formatting.

2. **LLM-assisted augmentation** — A smaller, higher-quality subset (60 examples) manually rewritten via 
   LLM prompting into natural, informal forward-style language (typos, abbreviations, rhetorical hooks), 
   adding variety the rule-based method alone couldn't achieve.

## Final Dataset Composition

| Split | Rows | Composition |
|---|---|---|
| Train | 21,623 | Original (15,402) + Rule-augmented (6,161) + LLM-augmented (60) |
| Validation (formal) | 1,926 | Formal-style only |
| Test (formal) | 1,925 | Formal-style baseline test set |
| Test (forward-style) | 300 | Forward-ified validation subset — tests domain adaptation |

**Label balance:** ~2.2:1 Fake:Real ratio across all splits (documented limitation — see model README).

## Files

- `train_final.csv` — Final augmented training set
- `val_formal.csv` — Formal-style validation set
- `test_formal.csv` — Formal-style test set (baseline evaluation)
- `test_forward_style.csv` — Forward-style test set (domain adaptation evaluation)

Each file contains `statement_clean` (text) and `binary_label` (Fake/Real) columns, plus original 
metadata from the source dataset.

*(CSV files to be added)*****
