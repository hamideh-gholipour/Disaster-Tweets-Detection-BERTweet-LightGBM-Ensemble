# Disaster-Tweets-Detection-BERTweet-LightGBM-Ensemble

**Concise, production-minded pipeline for classifying disaster-related tweets.**

Kaggle notebook : [TweetNLP Pipeline — EDA, BERTweet & LightGBM ensemble](https://www.kaggle.com/code/hamidehgholipour/tweetnlp-pipeline-eda-bertweet-lightgbm-ensemble)

---

## Project Goal and About the Competition

Detect whether a tweet describes a real-world disaster event (binary classification). The model is developed for the Kaggle challenge **"nlp-getting-started"**, aiming for robust generalization from cross-validated training to the competition test split.

## Data

**Files**

* `train.csv` — training set
* `test.csv` — test set
* `sample_submission.csv` — sample file in submission format

**Columns**

* `id` — unique identifier for each tweet
* `text` — tweet content
* `location` — reported location (may be blank)
* `keyword` — extracted keyword (may be blank)
* `target` — (train only) `1` if tweet describes a disaster, else `0`

---

## Pipeline

Minimal tweet-aware cleaning → BERTweet fine-tuning with Stratified K-Fold (OOF probs) → TF–IDF + engineered features → LightGBM (stacking with OOF) → Weighted ensemble (default 0.7 BERTweet / 0.3 LGBM).

## Key outcomes

* **Kaggle private LB**: `0.84462` — **Rank: 16**


---

## Architecture

* **Base model:** `vinai/bertweet-base` (fine-tuned for sequence classification)
* **Tokenizer:** BERTweet tokenizer (normalization enabled)
* **Stacking:** OOF probabilities from BERTweet appended to TF–IDF + engineered feature matrix
* **Meta model:** LightGBM (gradient-boosted trees)
* **Ensemble:** Weighted average of BERTweet and LightGBM probabilities

## My approach

Aiming for a reproducible, competitive pipeline that balances contextual modeling and interpretable features:

* Preserve Twitter artifacts (hashtags, mentions, emojis) during cleaning
* Use Stratified K-Fold to produce leakage-free stacking features (OOF)
* Combine deep contextual signals (BERTweet) with lexical/statistical features (TF–IDF, counts)
* Optimize ensemble weighting to control the precision/recall trade-off

---

## Tags

`NLP` | `transformers` | `BERTweet` | `LightGBM` | `stacking` | `feature-engineering` | `Kaggle` | `text-classification` | `data-science` | `machine-learning` | `ensemble-learning` | `nlp-preprocessing` | `text-mining` | `pytorch` | `huggingface` | `model-stacking` | `eda`

---

