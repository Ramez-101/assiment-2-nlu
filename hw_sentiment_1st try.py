"""
hw_sentiment.ipynb — Assignment Solutions
CS224u, Stanford, Spring 2022
Sentiment Analysis: SST-3 + Restaurant Reviews (Ternary Classification)

This file contains the code solutions for all homework tasks.
Each section corresponds to a question in the notebook.
"""

# ============================================================
# IMPORTS & SETUP
# ============================================================

from collections import Counter
import numpy as np
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
import torch.nn as nn

from torch_rnn_classifier import TorchRNNClassifier
from torch_shallow_neural_classifier import TorchShallowNeuralClassifier
import sst
import utils

SST_HOME = os.path.join('data', 'sentiment')

# Load datasets
sst_train    = sst.train_reader(SST_HOME)      # 8,544 movie review training examples
sst_dev      = sst.dev_reader(SST_HOME)        # movie review dev set
bakeoff_dev  = sst.bakeoff_dev_reader(SST_HOME)  # 2,361 restaurant review dev set


# ============================================================
# BASELINES (provided by the notebook — not homework tasks)
# ============================================================

def unigrams_phi(text):
    """Feature function: whitespace-tokenize and count tokens."""
    return Counter(text.split())


def fit_softmax_classifier(X, y):
    """Train a logistic regression (softmax) classifier."""
    mod = LogisticRegression(
        fit_intercept=True,
        solver='liblinear',
        multi_class='ovr')
    mod.fit(X, y)
    return mod


def rnn_phi(text):
    """Feature function for RNN: return list of tokens in order."""
    return text.split()


def fit_rnn_classifier(X, y):
    """Train a basic RNN classifier with learned embeddings."""
    vocab = utils.get_vocab(X, mincount=2)
    mod = TorchRNNClassifier(vocab, early_stopping=True)
    mod.fit(X, y)
    return mod


# ============================================================
# HW QUESTION 1: Token-level differences [1 point]
# ============================================================
# TASK: Write get_token_counts(df) that:
#   - Takes a DataFrame with a 'sentence' column
#   - Tokenizes each sentence by whitespace
#   - Returns a pd.Series of token counts sorted descending

def get_token_counts(df):
    """
    Count token frequencies across all sentences in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a 'sentence' column of strings.

    Returns
    -------
    pd.Series
        Token counts sorted from most to least frequent.
    """
    counts = sum(
        [Counter(sentence.split(" ")) for sentence in df["sentence"].values],
        Counter()
    )
    return pd.Series(counts).sort_values(ascending=False)


def test_get_token_counts(func):
    """Unit test for get_token_counts."""
    df = pd.DataFrame([
        {'sentence': 'a a b'},
        {'sentence': 'a b a'},
        {'sentence': 'a a a b.'}])
    result = func(df)
    for token, expected in (('a', 7), ('b', 2), ('b.', 1)):
        actual = result.loc[token]
        assert actual == expected, \
            "For token {}, expected {}; got {}".format(token, expected, actual)
    print("test_get_token_counts passed!")


# Run test
test_get_token_counts(get_token_counts)

# Example usage — compare the two dev sets:
print("\n--- Top 10 tokens in SST-3 dev ---")
print(get_token_counts(sst_dev).head(10))

print("\n--- Top 10 tokens in Bakeoff dev ---")
print(get_token_counts(bakeoff_dev).head(10))


# ============================================================
# HW QUESTION 2: Training on some of the bakeoff data [1 point]
# ============================================================
# TASK: Write run_mixed_training_experiment(wrapper_func, bakeoff_train_size) that:
#   1. Splits bakeoff_dev into train (first N rows) and eval (rest)
#   2. Runs sst.experiment with sst_train + bakeoff train portion
#   3. Evaluates on sst_dev + bakeoff eval portion
#   4. Returns the experiment dict

def run_mixed_training_experiment(wrapper_func, bakeoff_train_size):
    """
    Train on SST-3 + part of bakeoff dev, evaluate on SST-3 dev + rest of bakeoff dev.

    Parameters
    ----------
    wrapper_func : callable
        A model training function like fit_softmax_classifier.
    bakeoff_train_size : int
        Number of bakeoff_dev examples to include in the train set.

    Returns
    -------
    dict
        Result dict from sst.experiment.
    """
    # Split bakeoff_dev into train and eval portions
    bakeoff_dev_train = bakeoff_dev[:bakeoff_train_size]
    bakeoff_dev_eval  = bakeoff_dev[bakeoff_train_size:]

    # Run experiment: train on SST-3 + bakeoff train portion
    experiment = sst.experiment(
        [sst_train, bakeoff_dev_train],   # combined training data
        unigrams_phi,                      # feature function
        wrapper_func,                      # model wrapper
        vectorize=True,
        assess_dataframes=[sst_dev, bakeoff_dev_eval]  # two eval sets
    )

    return experiment


def test_run_mixed_training_experiment(func):
    """Unit test for run_mixed_training_experiment."""
    bakeoff_train_size = 1000
    experiment = func(fit_softmax_classifier, bakeoff_train_size)

    assess_size = len(experiment['assess_datasets'])
    assert assess_size == 2, \
        ("Expected 2 assessment datasets; got {}".format(assess_size))

    bakeoff_test_size = bakeoff_dev.shape[0] - bakeoff_train_size
    expected_eval_examples = bakeoff_test_size + sst_dev.shape[0]
    eval_examples = sum(len(d['raw_examples']) for d in experiment['assess_datasets'])
    assert expected_eval_examples == eval_examples, \
        "Expected {} evaluation examples; got {}".format(
            expected_eval_examples, eval_examples)
    print("test_run_mixed_training_experiment passed!")


# Run test
test_run_mixed_training_experiment(run_mixed_training_experiment)


# ============================================================
# HW QUESTION 3: More powerful vector-averaging baseline [2 points]
# ============================================================
# TASK: Write fit_shallow_neural_classifier_with_hyperparameter_search(X, y) that:
#   - Uses TorchShallowNeuralClassifier with early_stopping=True
#   - Runs 3-fold CV over:
#       hidden_dim: [50, 100, 200]
#       hidden_activation: [nn.Tanh(), nn.ReLU()]
#   - Returns the best model

def fit_shallow_neural_classifier_with_hyperparameter_search(X, y):
    """
    Train a shallow neural classifier with cross-validated hyperparameter search.

    Parameters
    ----------
    X : array-like
        Feature matrix (e.g. GloVe averaged vectors).
    y : list of str
        Labels.

    Returns
    -------
    Best fitted TorchShallowNeuralClassifier model.
    """
    base_model = TorchShallowNeuralClassifier(early_stopping=True)

    param_grid = {
        'hidden_dim': [50, 100, 200],
        'hidden_activation': [nn.Tanh(), nn.ReLU()]
    }

    best_model = utils.fit_classifier_with_hyperparameter_search(
        X, y,
        base_model,
        cv=3,
        param_grid=param_grid
    )

    return best_model


# ---- GloVe phi (needed to actually run HW3 experiment) ----

DATA_HOME  = 'data'
GLOVE_HOME = os.path.join(DATA_HOME, 'glove.6B')

glove_lookup = utils.glove2dict(
    os.path.join(GLOVE_HOME, 'glove.6B.300d.txt'))


def vsm_phi(text, lookup, np_func=np.mean):
    """
    Represent a text as the mean (or other combination) of its word vectors.

    Parameters
    ----------
    text : str
    lookup : dict
        Word -> vector mapping (e.g. GloVe).
    np_func : function
        Combination function, default np.mean.

    Returns
    -------
    np.array of shape (embedding_dim,)
    """
    vecs = np.array([lookup[w] for w in text.split() if w in lookup])
    if len(vecs) == 0:
        dim = len(next(iter(lookup.values())))
        return np.zeros(dim)
    return np_func(vecs, axis=0)


def glove_phi(text):
    """GloVe 300d mean-vector feature function."""
    return vsm_phi(text, glove_lookup, np_func=np.mean)


# Run HW3 experiment:
# glove_experiment = sst.experiment(
#     sst_train,
#     glove_phi,
#     fit_shallow_neural_classifier_with_hyperparameter_search,
#     assess_dataframes=[sst_dev],
#     vectorize=False
# )


# ============================================================
# HW QUESTION 4: BERT encoding [2 points]
# ============================================================
# TASK: Write hf_cls_phi(text) that:
#   - Encodes text with bert-base-uncased
#   - Returns the [CLS] token representation as np.array of shape (768,)

from transformers import BertModel, BertTokenizer
import vsm  # course utility: hf_encode, hf_represent

bert_weights_name = 'bert-base-uncased'
bert_tokenizer    = BertTokenizer.from_pretrained(bert_weights_name)
bert_model        = BertModel.from_pretrained(bert_weights_name)


def hf_cls_phi(text):
    """
    Encode a text string using BERT and return the [CLS] token representation.

    Parameters
    ----------
    text : str

    Returns
    -------
    np.array of shape (768,)
        The last hidden state of the [CLS] token.
    """
    # Step 1: tokenize and get token IDs (with special tokens [CLS], [SEP])
    token_ids = vsm.hf_encode(
        text=text,
        tokenizer=bert_tokenizer,
        add_special_tokens=True
    )

    # Step 2: run BERT forward pass, get last hidden states
    # reps shape: (1, n_tokens, 768)
    reps = vsm.hf_represent(
        batch_ids=token_ids,
        model=bert_model,
        layer=-1
    )

    # Step 3: extract [CLS] token (position 0 in the sequence dimension)
    cls_rep = reps[:, 0, :].squeeze()

    # Step 4: return as numpy array
    return cls_rep.cpu().numpy()


def test_hf_cls_phi(func):
    """Unit test for hf_cls_phi."""
    rep = func("Just testing!")
    assert rep.shape == (768,), \
        "Expected shape (768,); got {}".format(rep.shape)
    expected_first = str(0.1709)
    result_first   = "{0:.04f}".format(rep[0])
    assert expected_first == result_first, \
        "Expected first value {}; got {}".format(expected_first, result_first)
    print("test_hf_cls_phi passed!")


# Run test
test_hf_cls_phi(hf_cls_phi)

# Run HW4 experiment (takes ~11 min on CPU):
# bert_cls_experiment = sst.experiment(
#     sst_train,
#     hf_cls_phi,
#     fit_shallow_neural_classifier_with_hyperparameter_search,
#     assess_dataframes=[sst_dev],
#     vectorize=False
# )


# ============================================================
# HW QUESTION 5: Original System [3 points]
# ============================================================
# SYSTEM DESCRIPTION:
#   Uses BERT token-level embeddings (not just CLS) as input to a
#   bidirectional multi-layer RNN classifier. Key design choices:
#
#   1. bert_phi returns full token sequence embeddings (n_tokens, 768)
#      instead of a single CLS vector, giving the RNN rich per-token context.
#   2. Mixed training: half of bakeoff_dev is included in training so
#      the model learns restaurant-domain patterns.
#   3. Hyperparameter search over: hidden_dim, bidirectional, num_layers,
#      classifier_activation using 3-fold CV.
#   4. Final model is retrained on ALL data (sst_train + sst_dev + bakeoff_dev).
#
# Peak score during development: 0.619 (mean macro-F1 across both dev sets)


def bert_phi(text):
    """
    Encode text with BERT and return ALL token embeddings (not just CLS).

    Parameters
    ----------
    text : str

    Returns
    -------
    np.array of shape (n_tokens, 768)
        Last hidden states for every token — suitable as RNN input.
    """
    token_ids = vsm.hf_encode(
        text=text,
        tokenizer=bert_tokenizer,
        add_special_tokens=True
    )
    # reps shape: (1, n_tokens, 768)
    reps = vsm.hf_represent(
        batch_ids=token_ids,
        model=bert_model,
        layer=-1
    )
    # Return (n_tokens, 768) — squeeze out the batch dimension
    return reps.squeeze(0).numpy()


def fit_bert_rnn_classifier(X, y):
    """
    Train a bidirectional RNN on BERT token embeddings with hyperparameter search.

    Parameters
    ----------
    X : list of np.array, each of shape (n_tokens, 768)
    y : list of str

    Returns
    -------
    Best fitted TorchRNNClassifier.
    """
    base_model = TorchRNNClassifier(
        vocab=[],            # no vocab needed — inputs are dense vectors
        early_stopping=True,
        use_embedding=False  # skip the embedding layer; inputs already dense
    )

    param_grid = {
        'hidden_dim':            [100, 200],
        'bidirectional':         [True, False],
        'num_layers':            [1, 2],
        'classifier_activation': [nn.Tanh(), nn.ReLU()]
    }

    best_model = utils.fit_classifier_with_hyperparameter_search(
        X, y,
        base_model,
        cv=3,
        param_grid=param_grid
    )

    return best_model


def run_original_system():
    """
    Full pipeline for the original system:
      1. Split bakeoff_dev 50/50 into train and eval portions.
      2. Run experiment with BERT token embeddings + bidir RNN + CV.
      3. Retrain the best model on ALL available data.
      4. Return the experiment result dict (with best model updated).
    """
    # Step 1: mixed training split
    bakeoff_train_size  = bakeoff_dev.shape[0] // 2
    bakeoff_dev_train   = bakeoff_dev[:bakeoff_train_size]
    bakeoff_dev_eval    = bakeoff_dev[bakeoff_train_size:]

    # Step 2: find best model via CV
    experiment = sst.experiment(
        [sst_train, bakeoff_dev_train],
        bert_phi,
        fit_bert_rnn_classifier,
        assess_dataframes=[sst_dev, bakeoff_dev_eval],
        vectorize=False
    )

    # Step 3: retrain best model on ALL data for maximum final performance
    full_train = sst.build_dataset(
        [sst_train, sst_dev, bakeoff_dev],
        bert_phi,
        vectorizer=None,
        vectorize=False
    )
    experiment['model'] = experiment['model'].fit(
        full_train['X'], full_train['y']
    )

    return experiment


# Run:
# bestmod = run_original_system()


# ============================================================
# BAKEOFF SUBMISSION
# ============================================================

def predict_one_softmax(text):
    """Predict sentiment for a single text using the softmax baseline."""
    feats = [unigrams_phi(text)]
    X = softmax_experiment['train_dataset']['vectorizer'].transform(feats)
    return softmax_experiment['model'].predict(X)[0]


def predict_one_bert_rnn(text, experiment):
    """
    Predict sentiment for a single text using the BERT+RNN original system.

    Parameters
    ----------
    text : str
    experiment : dict
        Result dict from run_original_system().

    Returns
    -------
    str : 'positive', 'negative', or 'neutral'
    """
    X = [experiment['phi'](text)]    # encode with bert_phi
    return experiment['model'].predict(X)[0]


def create_bakeoff_submission(
        predict_one_func,
        output_filename='cs224u-sentiment-bakeoff-entry.csv'):
    """
    Generate the bakeoff submission CSV.

    Parameters
    ----------
    predict_one_func : callable
        Maps a text string to a label prediction.
    output_filename : str
    """
    bakeoff_test = sst.bakeoff_test_reader(SST_HOME)
    sst_test     = sst.test_reader(SST_HOME)

    bakeoff_test['dataset'] = 'bakeoff'
    sst_test['dataset']     = 'sst3'

    df = pd.concat((bakeoff_test, sst_test))
    df['prediction'] = df['sentence'].apply(predict_one_func)
    df.to_csv(output_filename, index=None)
    print("Submission saved to:", output_filename)


# Example final submission using original system:
# bestmod = run_original_system()
# create_bakeoff_submission(lambda text: predict_one_bert_rnn(text, bestmod))
