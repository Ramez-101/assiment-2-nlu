# hw_sentiment.ipynb — Full Documentation

**Course:** CS224u, Stanford, Spring 2022  
**Author:** Christopher Potts  
**Topic:** Supervised Sentiment Analysis (Ternary Classification) on SST-3 and Restaurant Reviews

---

## What This Notebook Does

This notebook builds and evaluates **ternary sentiment classifiers** (positive / negative / neutral) on two datasets:

1. **SST-3** — Stanford Sentiment Treebank (movie reviews)
2. **Bakeoff dataset** — A new dataset of restaurant reviews (same 3 labels)

The primary evaluation metric is the **mean of macro-F1** across both datasets, which weights all three classes and both datasets equally regardless of size.

The notebook progresses through a series of increasingly powerful models, from a simple unigram logistic regression up to a BERT-encoded bidirectional RNN with hyperparameter search.

---

## Setup & Dependencies

### Key Libraries
- `sklearn` — `LogisticRegression` for the softmax baseline
- `torch` / `torch.nn` — Neural network components
- `transformers` — Hugging Face BERT model and tokenizer
- `numpy`, `pandas` — Data handling

### Course-specific Modules
- `sst` — Dataset readers, experiment runner, vocabulary utilities
- `utils` — GloVe loader, hyperparameter search wrapper
- `vsm` — Vector space model utilities: `hf_encode`, `hf_represent`
- `torch_rnn_classifier` — `TorchRNNClassifier`
- `torch_shallow_neural_classifier` — `TorchShallowNeuralClassifier`
- `torch_tree_nn` — `TorchTreeNN` (imported but not used in homework)

### Data Directory
```python
SST_HOME = os.path.join('data', 'sentiment')
```

---

## Datasets

### Training Data

**`sst_train`** — SST-3 training split (8,544 examples) loaded via `sst.train_reader(SST_HOME)`.  
Each example is a dict with: `example_id`, `sentence`, `label`, `is_subtree`.  
Labels: `'positive'`, `'negative'`, `'neutral'`.

Optional: passing `include_subtrees=True` to the reader expands this to ~159,274 examples (phrase-level trees), but significantly increases training time.

### Dev / Evaluation Data

**`sst_dev`** — SST-3 validation split (movie reviews), loaded via `sst.dev_reader(SST_HOME)`.

**`bakeoff_dev`** — New restaurant review dev set (2,361 examples), loaded via `sst.bakeoff_dev_reader(SST_HOME)`.  
Label distribution: neutral (1,019), positive (777), negative (565).

---

## Experimental Framework

All models are evaluated using `sst.experiment(...)`, which:
1. Trains a model on the training data
2. Evaluates it on each dataset in `assess_dataframes`
3. Prints per-class precision/recall/F1 for each dataset
4. Returns a results dict

### The `sst.experiment` Return Dict Structure
```
'model'           : trained model object
'phi'             : the feature function used
'train_dataset'   : {'X', 'y', 'vectorizer', 'raw_examples'}
'assess_datasets' : list of dicts (same structure as train_dataset)
'predictions'     : list of prediction lists (one per assess dataset)
'metric'          : score function name
'score'           : score per assessment dataset
```

---

## Baseline Models

### 1. Softmax Baseline (Unigram Logistic Regression)

**`unigrams_phi(text)`** — Feature function. Splits text on whitespace and returns a `Counter` of token frequencies.

**`fit_softmax_classifier(X, y)`** — Trains a `LogisticRegression` model with `liblinear` solver and one-vs-rest multi-class strategy.

**Result:**
- SST-3 dev macro-F1: 0.518
- Bakeoff dev macro-F1: 0.315
- Mean: **0.416**

The softmax model does poorly on the bakeoff set because it was trained only on movie review vocabulary — the restaurant domain has a different token distribution.

---

### 2. RNN Baseline

**`rnn_phi(text)`** — Feature function. Splits text on whitespace and returns a list of tokens (word order preserved for the RNN).

**`fit_rnn_classifier(X, y)`** — Trains a `TorchRNNClassifier` with:
- Vocabulary built from training data (min count = 2)
- `early_stopping=True`

**Result:**
- SST-3 dev macro-F1: 0.501
- Bakeoff dev macro-F1: 0.362
- Mean: **0.432**

The RNN improves slightly on the bakeoff set because it learns sequence patterns rather than just word counts.

---

## Error Analysis

**`find_errors(experiment)`**

Takes a results dict from `sst.experiment` and returns a `pd.DataFrame` with columns:
- `raw_examples` — the original text
- `predicted` — model's prediction
- `gold` — true label
- `correct` — boolean
- `dataset` — index of which assessment dataset (0 = SST-3 dev, 1 = bakeoff dev)

The notebook merges the softmax and RNN analysis DataFrames to allow comparison, then demonstrates filtering for specific error types (e.g., examples where softmax is correct but RNN is wrong, with gold label `'positive'`).

---

## Homework Questions

### HW1: Token-level Differences (1 pt)

**`get_token_counts(df)`**

Given a DataFrame with a `'sentence'` column, tokenizes all sentences by whitespace and returns a `pd.Series` of token frequencies, sorted descending.

**Implementation:** Uses `Counter` summed across all sentences, then wrapped in a `pd.Series` and sorted.

**Test:** `test_get_token_counts` checks exact counts for tokens `'a'` (7), `'b'` (2), and `'b.'` (1) on a toy DataFrame.

**Purpose:** Comparing token distributions between SST-3 and the bakeoff set reveals vocabulary differences — the restaurant domain uses different words and has different punctuation/encoding details that affect model performance.

---

### HW2: Training on Some Bakeoff Data (1 pt)

**`run_mixed_training_experiment(wrapper_func, bakeoff_train_size)`**

Splits `bakeoff_dev` into a train portion (first `bakeoff_train_size` rows) and an eval portion (the rest). Runs `sst.experiment` with:
- **Train set:** `sst_train` + bakeoff train portion (concatenated as a list of DataFrames)
- **Phi:** `unigrams_phi`
- **Model:** the supplied `wrapper_func`
- **Assess:** `[sst_dev, bakeoff eval portion]`

Returns the full experiment dict.

**Result (bakeoff_train_size=1000, softmax):**
- SST-3 dev macro-F1: 0.518
- Bakeoff eval macro-F1: 0.518
- Mean: **0.518** (major improvement on bakeoff from 0.315 → 0.518)

**Test:** `test_run_mixed_training_experiment` checks that exactly 2 assessment datasets are used and that the total evaluation examples match expectations.

**Key insight:** Adding just 1,000 restaurant review examples to training dramatically improves bakeoff performance without hurting SST-3 performance.

---

### HW3: More Powerful Vector-Averaging Baseline (2 pts)

**`fit_shallow_neural_classifier_with_hyperparameter_search(X, y)`**

Wraps `TorchShallowNeuralClassifier` with 3-fold cross-validation hyperparameter search over:
- `hidden_dim`: [50, 100, 200]
- `hidden_activation`: [`nn.Tanh()`, `nn.ReLU()`]

Always uses `early_stopping=True`. Best parameters are selected by CV score and the model is returned.

Used with **GloVe 300d** averaged word vectors (`glove_phi`) as input features.

**`vsm_phi(text, lookup, np_func=np.mean)`** — Maps a sentence to a fixed-size vector by looking up each word in a pretrained embedding dictionary and applying `np_func` (default: mean) column-wise.

**`glove_phi(text)`** — Thin wrapper around `vsm_phi` using the loaded GloVe 300d lookup.

**Result on SST-3 dev (GloVe mean + shallow neural, best: ReLU, hidden_dim=200):**
- macro-F1: **0.538**

---

### HW4: BERT Encoding (2 pts)

**`hf_cls_phi(text)`**

Encodes a text string using `bert-base-uncased` and returns the [CLS] token representation as a numpy array of shape `(768,)`.

**Steps:**
1. `vsm.hf_encode(text, tokenizer, add_special_tokens=True)` — Tokenizes and returns token IDs
2. `vsm.hf_represent(batch_ids, model, layer=-1)` — Runs BERT forward pass, returns last hidden states with shape `(1, n_tokens, 768)`
3. Index `reps[:, 0, :].squeeze()` to get the [CLS] vector
4. `.cpu().numpy()` to return as a numpy array

**Test:** `test_hf_cls_phi` checks that the output shape is `(768,)` and that the first value rounds to `0.1709`.

**Result (BERT CLS + shallow neural with hyperparameter search, best: Tanh, hidden_dim=100):**
- SST-3 dev macro-F1: **0.580**

Significant improvement over GloVe mean (0.538 → 0.580) due to BERT's contextual, deep representations.

**Note:** Encoding the full SST-3 train set takes ~11 minutes on CPU.

---

### HW5: Original System (3 pts)

The original system combines several improvements:

#### System Design

**Core idea:** Use BERT token embeddings (not just the CLS vector) as input to a bidirectional multi-layer RNN, trained on mixed data (SST-3 + part of bakeoff), with hyperparameter search.

**`bert_phi(text)`** — Returns the full BERT last-layer hidden states for all tokens as a numpy array of shape `(n_tokens, 768)`. Unlike `hf_cls_phi`, this preserves the sequence structure for the RNN.

**`system_RNN_bert(X, y)`** — Trains a `TorchRNNClassifier` with `use_embedding=False` (since inputs are already dense vectors) and searches over:
- `hidden_dim`: [100, 200]
- `classifier_activation`: [`nn.Tanh()`, `nn.ReLU()`]
- `bidirectional`: [True, False]
- `num_layers`: [1, 2]

Uses 3-fold cross-validation. Always uses `early_stopping=True`.

#### Training Strategy

1. Split `bakeoff_dev` 50/50 into train and eval portions
2. Run `sst.experiment` on `[sst_train, bakeoff_dev_train]` with `bert_phi` and `system_RNN_bert`
3. After finding the best model, **retrain on all data** (`sst_train + sst_dev + bakeoff_dev`) using `sst.build_dataset` and `model.fit(X_train, y_train)`

#### Results

**Best hyperparameters:** `{'bidirectional': True, 'classifier_activation': ReLU(), 'hidden_dim': 100, 'num_layers': 2}`

| Dataset | macro-F1 |
|---|---|
| SST-3 dev | 0.591 |
| Bakeoff dev (eval portion) | 0.648 |
| **Mean** | **0.619** |

This is the best result in the notebook, beating all baselines significantly.

---

## Bakeoff

### `predict_one_softmax(text)`

Applies the softmax experiment pipeline to a single text string:
1. Featurize with `unigrams_phi`
2. Vectorize using the stored `DictVectorizer`
3. Call `model.predict()` and return the single label

### `predict_one_rnn(text)`

Applies the original BERT-RNN system to a single text string:
1. Encode with `bert_phi`
2. Call `bestmod_bert['model'].predict([X])` and return the single label

### `create_bakeoff_submission(predict_one_func, output_filename)`

Loads both the SST-3 test set and the bakeoff test set, runs `predict_one_func` on each sentence, and writes a CSV file with columns: `example_id`, `sentence`, `label`, `is_subtree`, `dataset`, `prediction`.

Output file: `cs224u-sentiment-bakeoff-entry.csv`

---

## Results Summary

| System | SST-3 dev F1 | Bakeoff dev F1 | Mean |
|---|---|---|---|
| Unigram Softmax | 0.518 | 0.315 | 0.416 |
| RNN (random embeddings) | 0.501 | 0.362 | 0.432 |
| Mixed training softmax (1000 bakeoff examples) | 0.518 | 0.518 | 0.518 |
| GloVe mean + shallow neural (CV) | 0.538 | — | — |
| BERT CLS + shallow neural (CV) | 0.580 | — | — |
| **BERT tokens + bidir RNN (CV) + mixed training** | **0.591** | **0.648** | **0.619** |

---

## Notes & Code Quality

- No bugs were found in this notebook. All implemented functions are complete and correct.
- The `if 'IS_GRADESCOPE_ENV' not in os.environ` guard is used throughout to prevent autograder failures from code that requires external resources (BERT model, GloVe files, bakeoff data).
- The notebook clearly separates the development pipeline (train → eval on dev) from the final submission pipeline (retrain on all data → predict on test).
- The `predict_one_rnn` function references `bestmod_bert`, which is defined inside the `IS_GRADESCOPE_ENV` guard — make sure to run the original system cell before calling this function locally.
