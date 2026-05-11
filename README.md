# Building NLP Apps with Hugging Face

An educational tutorial showcasing multiple NLP pipelines using the Hugging Face `transformers` library. It covers **text generation**, **conversational chatbots**, **neural machine translation**, **question answering**, and fine-tuning a transformer for sentiment classification.

---

## 📚 Notebooks

| # | Notebook | Task | Model |
|---|----------|------|-------|
| 1 | [`01_text_generation.ipynb`](notebooks/01_text_generation.ipynb) | Autoregressive text generation | GPT-2 |
| 2 | [`02_conversational_chatbot.ipynb`](notebooks/02_conversational_chatbot.ipynb) | Multi-turn dialogue | BlenderBot-400M |
| 3 | [`03_translation.ipynb`](notebooks/03_translation.ipynb) | Neural machine translation + BLEU | T5-small, Helsinki-NLP/opus-mt |
| 4 | [`04_question_answering.ipynb`](notebooks/04_question_answering.ipynb) | Extractive QA + EM/F1 metrics | DistilBERT (SQuAD) |
| 5 | [`05_sentiment_classification.ipynb`](notebooks/05_sentiment_classification.ipynb) | Fine-tuning for sentiment | DistilBERT (IMDb) |

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/NLP_project.git
cd NLP_project
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch JupyterLab

```bash
jupyter lab
```

Open any notebook under `notebooks/` and run the cells from top to bottom.

---

## 📦 Requirements

- Python 3.9+
- See [`requirements.txt`](requirements.txt) for the full list of packages

Key libraries:

| Library | Purpose |
|---------|---------|
| `transformers` | Pretrained models and pipelines |
| `torch` | PyTorch backend |
| `tensorflow` / `keras` | TensorFlow backend (optional) |
| `datasets` | Dataset loading and preprocessing |
| `evaluate` | NLP evaluation metrics |
| `sacrebleu` | BLEU score for translation |
| `scikit-learn` | Classification metrics |

> **GPU vs CPU**: All notebooks run on CPU. A GPU significantly speeds up the fine-tuning notebook (Notebook 5). If you have a CUDA-enabled GPU, `torch` will use it automatically.

---

## 🗂️ Repository Structure

```
NLP_project/
├── notebooks/
│   ├── 01_text_generation.ipynb        # GPT-2 text generation
│   ├── 02_conversational_chatbot.ipynb # BlenderBot chatbot
│   ├── 03_translation.ipynb            # T5 / MarianMT translation + BLEU
│   ├── 04_question_answering.ipynb     # Extractive QA + EM/F1
│   └── 05_sentiment_classification.ipynb # Fine-tuning DistilBERT
├── requirements.txt
└── README.md
```

---

## 📖 Notebook Summaries

### 1 · Text Generation with GPT-2
- Load GPT-2 via the `pipeline` API and with `GPT2LMHeadModel`
- Compare greedy, sampling, and beam-search decoding
- Control creativity with `temperature`, `top_k`, and `top_p`
- Visualise next-token probability distributions

### 2 · Conversational Chatbot (BlenderBot)
- Build a stateful `Chatbot` class that maintains conversation history
- Handle context-window constraints and history truncation
- Run an interactive multi-turn chat session
- Discuss domain adaptation and Gradio UI extension

### 3 · Neural Machine Translation (T5 + MarianMT)
- Translate English → German / French / Romanian with T5-small
- Translate English → Spanish with Helsinki-NLP MarianMT
- Compute corpus-level and sentence-level **BLEU** scores
- Analyse per-sentence translation quality

### 4 · Question Answering
- Perform extractive QA with DistilBERT (SQuAD)
- Implement **Exact Match** and **F1** evaluation from scratch
- Build a simple open-domain QA system over multiple passages
- Visualise answer-confidence scores

### 5 · Sentiment Classification (Fine-Tuning)
- Fine-tune DistilBERT on 2 000 IMDb reviews using `Trainer`
- Evaluate with accuracy, precision, recall, F1, and a confusion matrix
- Run inference on custom movie reviews
- Compare fine-tuned model to zero-shot classification (BART-MNLI)
- Plot training/evaluation loss curves

---

## 💡 Extension Ideas

1. **Domain-adapted chatbot** – fine-tune BlenderBot on a customer-support corpus (e.g., Ubuntu Dialogue Corpus).
2. **Multilingual models** – add more languages using mBERT or mT5.
3. **Web UI** – wrap any pipeline in a [Gradio](https://gradio.app) or [Streamlit](https://streamlit.io) interface.
4. **Summarisation** – add a Pegasus/BART summarisation notebook.
5. **Evaluation metrics** – extend BLEU with ROUGE, BERTScore, and METEOR.
6. **Retrieval-augmented QA** – integrate FAISS + DPR for open-domain question answering at scale.

---

## 📄 Licence

This project is released under the MIT Licence. See [LICENSE](LICENSE) for details.
