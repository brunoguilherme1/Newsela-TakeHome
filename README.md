# 📚 Newsela-TakeHome: Topic Recommendation System

This project implements a **retrieval and re-ranking model** for recommending relevant **K–12 topics** to educational content. Given the 3-hour constraint outlined in the [Newsela Take-Home Instructions](https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations), we built a fast, modular, and accurate baseline system combining semantic embeddings with traditional ML-based ranking.

---

## 🚀 Project Overview

We approached the problem in two stages:

### 1. **Retrieval (Stage 1)**

We used the multilingual model `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` to encode both content and topic texts into dense embeddings. FAISS was then used to efficiently retrieve the **top-50 nearest topics** based on L2 distance.

### 2. **Re-Ranking (Stage 2)**

Each retrieved topic–content pair was enriched with additional features:

* Word2Vec + TF-IDF weighted vectors
* Cosine similarity, Euclidean distance, dot product
* Word overlap and token counts
  These features were passed to a **LightGBM binary classifier** to predict final relevance.

Despite the time constraint, this 2-stage strategy balances **speed**, **scalability**, and **accuracy**, making it suitable for deployment or further refinement.

---

## 📒 Notebooks

### 🔍 `1.0_EDA.ipynb`

* Data loading, inspection, and cleaning
* Strategy justification for multilingual support
* Stratified split by language for validation

### 🧠 `2.0_Model.ipynb`

* Embedding generation using SentenceTransformers & Word2Vec
* Retrieval + FAISS indexing
* Feature engineering
* LightGBM training with SMOTE resampling
* Evaluation (F2, AUC, Precision, Recall)

⚡ **Run both notebooks in Google Colab with GPU — total runtime < 1 hour**

> ⚠️ **Note:** This Colab notebook may not render correctly on GitHub due to widget metadata issues.
> 👉 To view and run the notebook properly, open it directly in Colab:
> [Open in Colab](https://colab.research.google.com/drive/1nyyirHwWUp6TU1dqTNmPscB8usb9w0QN?usp=sharing)
---

## 🧪 Results

### 🔎 Retrieval-only metrics (top-50 topics using FAISS)

| Metric       | @1     | @3     | @5     | @10    | @50    |
| ------------ | ------ | ------ | ------ | ------ | ------ |
| Precision\@k | 0.1426 | 0.0793 | 0.0563 | 0.0342 | 0.0098 |
| Recall\@k    | 0.1001 | 0.1469 | 0.1666 | 0.1937 | 0.2633 |
| F1\@k        | 0.1107 | 0.0947 | 0.0774 | 0.0542 | 0.0184 |
| Coverage\@k  | 0.1001 | 0.1469 | 0.1752 | 0.2386 | 0.3148 |
| Hits\@k      | 0.1001 | 0.1469 | 0.1752 | 0.2386 | 0.3148 |

### ✅ Re-ranking with LightGBM (content\_20 evaluation)

| Metric    | Score  |
| --------- | ------ |
| Precision | 0.4386 |
| Recall    | 0.3051 |
| F1 Score  | 0.4099 |
| F2 Score  | 0.3049 |
| ROC AUC   | 0.9616 |

### ✅ Cross-Validation (LightGBM)

| Metric    | Score  |
| --------- | ------ |
| Precision | 0.4586 |
| Recall    | 0.3151 |
| F1 Score  | 0.4199 |
| F2 Score  | 0.4149 |
| ROC AUC   | 0.9716 |

---

## 🧩 Model Architecture

* **Retriever**: Sentence-BERT + FAISS
* **Feature Engineering**:

  * Word2Vec + TF-IDF weighted averages
  * Similarity metrics
  * Word overlaps
* **Classifier**: LightGBM (binary classifier)
* **Post-Processing**: Select topic\_ids where `prob > 0.5`

---

## 🖥 Local Usage

To reproduce results or run inference:

```bash
make setup        # Create virtual env and install requirements
make train        # Train and save LightGBM model
make predict      # Evaluate model on content_20.csv
```
---

## 🧠 Limitations & Future Work

Due to the strict **3-hour time constraint**, this solution focuses on building a fast and scalable pipeline with strong baseline performance. However, several enhancements could significantly improve results:

* **Larger Embedding Models**: Using more powerful multilingual models (e.g., `all-mpnet-base-v2`, `mdeberta-v3`, or `xlm-roberta-large`) would increase semantic understanding, especially for nuanced topic-content relationships.

* **LLM-based Ranking**: Incorporating **LLMs like ChatGPT or Claude** for final re-ranking or scoring relevance would enable deeper reasoning and richer contextual alignment.

* **RAG (Retrieval-Augmented Generation)**: A hybrid approach that feeds retrieved candidates into a generative model could better handle edge cases or generate explanations alongside predictions.

* **Contrastive Learning**: Training a custom **retriever** using contrastive objectives (e.g., SimCSE or Sentence-BERT with hard negatives) would enhance retrieval quality and candidate diversity.

* **Better Token-Level Alignment**: Currently, we rely on average word embeddings + TF-IDF. Using full token-level models (e.g., attention-based matching) would allow better fine-grained alignment.

This current architecture is intentionally lightweight and fast. It serves as a **strong baseline** that can be extended into modern neural architectures with more compute or time.
