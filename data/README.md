# Fine-Tuned Misinformation Detection Model

A DistilBERT transformer fine-tuned for binary misinformation classification (Fake/Real), trained on 
the augmented dataset described in [`/data`](../data).

## Model Details

- **Base model:** `distilbert-base-uncased`
- **Task:** Binary sequence classification
- **Label mapping:** `Fake = 0`, `Real = 1`
- **Tokenizer settings:** `max_length=256`, `padding="max_length"`, `truncation=True`
- **Framework:** HuggingFace Transformers, PyTorch

## Evaluation Results

Evaluated on two separate test sets to specifically measure generalization from formal to informal, 
forward-style text:

| Test Set | Accuracy | F1 Score |
|---|---|---|
| Formal (baseline) | 70.7% | 0.619 |
| Forward-style | 74.3% | 0.655 |

**Per-class performance (formal test set):**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Fake | 0.86 | 0.68 | 0.76 |
| Real | 0.52 | 0.76 | 0.62 |

## Known Limitations

- **Class imbalance effect:** Training data has a ~2.2:1 Fake:Real ratio, resulting in lower precision 
  on "Real" predictions. This directly motivated the addition of a human-in-the-loop community 
  verification layer in the full system rather than relying on the model in isolation.
- **Domain scope:** The model is trained and validated specifically on political, social, and health-related 
  misinformation-style claims — the dominant categories of real-world WhatsApp forward content. It is 
  **not** designed as a general-purpose fact-checker for arbitrary trivia or scientific claims outside 
  this domain, and confident misclassification should be expected on out-of-domain inputs.

## Files

- `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json` — Model and tokenizer weights
- Loadable via `AutoModelForSequenceClassification.from_pretrained()` and `AutoTokenizer.from_pretrained()`

*(Model files to be added — note: model.safetensors is ~241MB, may require Git LFS)*
