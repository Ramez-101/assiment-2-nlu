# `hw_sentiment.ipynb` Change Documentation

## Overview

This document summarizes the fixes and cleanup applied to `hw_sentiment.ipynb`.
The goal was to:

- review the homework-answer cells for missing code or incorrect logic
- fix fragile or incomplete implementations
- remove leftover template placeholders
- verify the edited notebook as far as the local environment allows

## Files Changed

- `hw_sentiment.ipynb`
- `hw_sentiment_changes.md`

## What Was Updated

### 1. Cleaned the token counting function

Function updated:

- `get_token_counts`

Changes made:

- removed the leftover `YOUR CODE HERE` placeholder
- rewrote the implementation to use `Counter.update(...)`
- used `sentence.split()` instead of `sentence.split(" ")`

Why this matters:

- `split()` is the safer whitespace tokenizer for this assignment
- it handles repeated spaces more robustly than `split(" ")`
- the implementation is now clearer and still matches the notebook’s expected output

Behavior:

- returns a `pd.Series` of token counts sorted from highest to lowest frequency

### 2. Cleaned the mixed-training experiment helper

Function updated:

- `run_mixed_training_experiment`

Changes made:

- removed the leftover placeholder marker
- kept the exact task-required behavior:
  - first `bakeoff_train_size` examples from `bakeoff_dev` go into training
  - remaining examples go into evaluation
  - training uses `[sst_train, bakeoff_dev_train]`
  - evaluation uses `[sst_dev, bakeoff_dev_eval]`
  - features are based on `unigrams_phi`
  - the supplied `wrapper_func` is passed through to `sst.experiment`
- simplified the function to return `sst.experiment(...)` directly

Why this matters:

- the function now reads as a direct implementation of the assignment prompt
- unnecessary temporary variables and template comments were removed

### 3. Cleaned the shallow neural baseline wrapper

Function updated:

- `fit_shallow_neural_classifier_with_hyperparameter_search`

Changes made:

- removed the leftover placeholder marker
- kept the intended hyperparameter search over:
  - `hidden_dim` in `[50, 100, 200]`
  - `hidden_activation` in `[nn.Tanh(), nn.ReLU()]`
- simplified the return path

Why this matters:

- the code is now cleaner and reads as a finished answer cell rather than a partially completed template

### 4. Hardened the BERT CLS feature function

Section updated:

- BERT setup and `hf_cls_phi`

Changes made:

- removed leftover placeholder markers
- set `bert_model.eval()` after loading the model
- formatted the encoding call more clearly
- changed the CLS extraction from:

```python
reps[:, 0, :].squeeze()
```

- to:

```python
reps[0, 0]
```

- returned the representation via:

```python
cls_rep.detach().cpu().numpy()
```

Why this matters:

- `bert_model.eval()` is safer for representation extraction
- `reps[0, 0]` is the most direct way to select the `[CLS]` representation
- `detach().cpu().numpy()` is more robust than assuming a plain CPU tensor

### 5. Cleaned the original-system cell

Section updated:

- custom/original system code cell

Changes made:

- removed the leftover placeholder marker from the custom system section
- set the custom BERT model to eval mode with `bert_model.eval()`
- updated the custom `bert_phi` function to return:

```python
reps.squeeze(0).detach().cpu().numpy()
```

Why this matters:

- this makes the custom system’s embedding extraction more consistent with the earlier BERT feature function
- it avoids fragile assumptions about tensor device and gradient state

## What Was Checked

### Notebook structure validation

I verified that:

- `hw_sentiment.ipynb` is still valid JSON

### Logic checks completed locally

Using lightweight local stand-ins/mocks, I verified:

- `get_token_counts`
- `run_mixed_training_experiment`

These checks confirmed that the edited implementations match the notebook’s own expected behavior.

### Placeholder cleanup

I also checked that the edited homework/custom-system cells no longer contain `YOUR CODE HERE` markers.

## What Could Not Be Fully Run Here

The following could not be fully verified end-to-end in this sandbox:

- the BERT feature extraction cell using real Hugging Face classes
- full notebook execution
- full model training/evaluation for the sentiment systems

Reason:

- the local environment is missing some runtime dependencies needed for those cells, including `transformers` and `pandas`

## Summary

The notebook is now cleaner and more complete:

- homework answer cells have been finished and cleaned up
- BERT feature extraction code is safer and more explicit
- the custom/original system cell was cleaned for consistency
- placeholder markers were removed
- notebook structure and key non-BERT helper logic were validated locally
