# Hate Speech Recognition using Machine Learning

Detects toxic and hateful language in online comments. Given a piece of text, the models predict which of **six categories** it belongs to. A single comment can belong to several at once (multi-label classification):

`toxic` · `severe_toxic` · `obscene` · `threat` · `insult` · `identity_hate`

Three approaches are implemented and compared, from a classic baseline to deep learning:

| Notebook | Model |
|----------|-------|
| `HS_NB.ipynb` | Multinomial **Naive Bayes** (one classifier per label) |
| `HS_LSTM.ipynb` | **LSTM** recurrent neural network |
| `HS_DNN.ipynb` | **Bidirectional LSTM** with a deep stack of dense layers |

---

## Dataset

The models are trained on the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) dataset from Kaggle. It contains about **159,571 Wikipedia talk-page comments**, each labelled 0/1 for the six categories above.

The dataset is **not included** in this repo. Download `train.csv` from Kaggle and either place it at the path the notebooks expect:

```
CommentToxicity-main/jigsaw-toxic-comment-classification-challenge/train.csv/train.csv
```

or edit the `pd.read_csv(...)` line in each notebook.

---

## How it works

### 1. Naive Bayes baseline (`HS_NB.ipynb`)
1. The data is split into 70% train, 21% validation and 9% test.
2. Comments are turned into bag-of-words count vectors with `CountVectorizer`.
3. A separate `MultinomialNB` classifier is trained for each of the six labels.
4. To classify new text, it is vectorised and passed through all six classifiers, giving a 0/1 prediction per label.

### 2. LSTM network (`HS_LSTM.ipynb`)
1. The data uses the same 70 / 21 / 9 split.
2. The Keras `Tokenizer` turns text into integer sequences, which are padded or truncated to 100 tokens.
3. Architecture: `Embedding(100)` → `SpatialDropout1D(0.2)` → `LSTM(100)` → `Dense(6, sigmoid)`.
4. Training uses binary cross-entropy with the Adam optimiser, for 5 epochs with batch size 64 and early stopping.
5. Each of the six sigmoid outputs is a probability. A value above 0.5 means that label applies.

### 3. Bidirectional LSTM + deep dense layers (`HS_DNN.ipynb`)
1. `TextVectorization` builds a vocabulary of up to 200,000 words and turns each comment into a 1,800-token sequence.
2. A `tf.data` pipeline does cache → shuffle → batch(16) → prefetch, then splits the data 70 / 20 / 10.
3. Architecture: `Embedding(32)` → `Bidirectional(LSTM(32))` → 7 fully connected ReLU layers (alternating 128/256 units) → `Dense(6, sigmoid)`.
4. The model is trained with binary cross-entropy and Adam, then evaluated with precision and recall.

### Example prediction
When a sample insulting comment is run through the LSTM model, it returns roughly:

| toxic | severe_toxic | obscene | threat | insult | identity_hate |
|------|------|------|------|------|------|
| 0.99 | 0.28 | 0.97 | 0.04 | 0.82 | 0.13 |

So the comment is flagged as **toxic, obscene and an insult**.

---

## Results

| Model | Precision | Recall | Notes |
|-------|-----------|--------|-------|
| Naive Bayes | 0.975 | 0.975 | Micro-averaged over flattened labels (validation set). Inflated because most labels are 0. |
| LSTM | **0.796** | 0.681 | Test set, micro-averaged. Exact-match accuracy across all six labels: 0.918. |
| BiLSTM + DNN | **0.806** | 0.673 | Test set, trained for only 1 epoch. |

The deep learning models give a much more realistic picture of performance on the rare toxic classes. They catch about two-thirds of toxic labels while keeping precision around 80%.

---

## Getting started

```bash
pip install pandas numpy scikit-learn tensorflow jupyter
jupyter notebook
```

Open any of the notebooks, point the CSV path to your downloaded `train.csv`, and run all cells. The last cells let you change `input_text` to test your own sentences.

> Training the LSTM models on the full dataset takes roughly 8–9 minutes per epoch on a CPU. A GPU is recommended.

## Tech stack
Python · pandas · NumPy · scikit-learn · TensorFlow / Keras · Jupyter
