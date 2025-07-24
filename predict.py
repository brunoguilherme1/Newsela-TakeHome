# predict.py

import os
import numpy as np
import pandas as pd
import joblib
import faiss
from typing import List, Optional
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
from utils import encode_text_with_tfidf_word2vec, count_overlap_and_lengths


class PredictionPipeline:
    def __init__(
        self,
        sentence_model_path: str = "sentence_model",
        w2v_path: str = "models/word2vec_topics.model",
        tfidf_path: str = "models/tfidf.pkl",
        lgbm_path: str = "models/lgbm_model.pkl",
        topics_path: str = "data/topics.csv",
        topic_embeddings_path: str = "data/topic_embeddings.npy",
        faiss_index_path: str = "faiss_index.index"
    ):
        self._load_models(
            sentence_model_path,
            w2v_path,
            tfidf_path,
            lgbm_path,
            topics_path,
            topic_embeddings_path,
            faiss_index_path
        )

    def _load_models(
        self,
        sentence_model_path,
        w2v_path,
        tfidf_path,
        lgbm_path,
        topics_path,
        topic_embeddings_path,
        faiss_index_path
    ):
        print("🔄 Loading models and data...")

        self.encoder = SentenceTransformer(sentence_model_path)
        self.w2v = Word2Vec.load(w2v_path)
        self.tfidf = joblib.load(tfidf_path)
        self.lgbm_pipeline = joblib.load(lgbm_path)

        self.topics = pd.read_csv(topics_path)
        self.topic_embeddings = np.load(topic_embeddings_path)
        self.index = faiss.read_index(faiss_index_path)

        self.topic_ids = self.topics["id"].tolist()
        self.topic_texts = dict(zip(self.topics["id"], self.topics["text"]))

        print("✅ Models and data loaded successfully.")

    def _encode_query(self, query: str) -> np.ndarray:
        return self.encoder.encode([query])[0].astype("float32")

    def _retrieve_candidates(self, query_embedding: np.ndarray, top_k: int) -> List[str]:
        _, indices = self.index.search(query_embedding.reshape(1, -1), top_k)
        return [self.topic_ids[i] for i in indices[0]]

    def _build_feature_vector(self, query_text: str, query_vec: np.ndarray, topic_id: str) -> np.ndarray:
        topic_text = self.topic_texts.get(topic_id, "")
        topic_vec = encode_text_with_tfidf_word2vec(topic_text, self.w2v, self.tfidf.idf_)

        dot = np.dot(query_vec, topic_vec)
        cosine_sim = dot / (np.linalg.norm(query_vec) * np.linalg.norm(topic_vec) + 1e-9)
        euclidean_dist = np.linalg.norm(query_vec - topic_vec)
        abs_diff_mean = np.mean(np.abs(query_vec - topic_vec))

        intersection, len_q, len_t = count_overlap_and_lengths(query_text, topic_text)

        return np.concatenate([
            query_vec,
            topic_vec,
            [cosine_sim, euclidean_dist, abs_diff_mean, dot, intersection, len_q, len_t]
        ])

    def predict(self, query: str, top_k: int = 50) -> pd.DataFrame:
        print(f"🔎 Processing query: {query}")

        query_embedding = self._encode_query(query)
        query_vec = encode_text_with_tfidf_word2vec(query, self.w2v, self.tfidf.idf_)

        candidate_ids = self._retrieve_candidates(query_embedding, top_k)
        feature_matrix = np.vstack([
            self._build_feature_vector(query, query_vec, tid) for tid in candidate_ids
        ])

        probabilities = self.lgbm_pipeline.predict_proba(feature_matrix)[:, 1]

        results = pd.DataFrame({
            "topic_id": candidate_ids,
            "topic_text": [self.topic_texts[tid] for tid in candidate_ids],
            "probability": probabilities
        }).sort_values("probability", ascending=False).reset_index(drop=True)

        return results


# -------------------------
# ✅ Example usage
# -------------------------
# predict.py (bottom section)

from metrics import print_classification_metrics

if __name__ == "__main__":
    # Load content_20
    df_content = pd.read_csv("data/content.csv")
    df_corr = pd.read_csv("data/correlations.csv")
    df_corr["content_ids"] = df_corr["content_ids"].str.split()
    exploded_corr = df_corr.explode("content_ids")

    content_20, _ = train_test_split(df_content, test_size=0.8, stratify=df_content["language"], random_state=42)

    pipeline = PredictionPipeline()

    y_true_all = []
    y_pred_all = []
    y_proba_all = []

    for _, row in content_20.iterrows():
        cid = row["id"]
        query = row["title"] + " " + row["description"] + " " + row["text"]

        # Ground truth topics for this content
        true_topics = exploded_corr[exploded_corr["content_ids"] == cid]["topic_id"].tolist()

        if not true_topics:
            continue

        preds_df = pipeline.predict(query, top_k=50)

        for _, pred_row in preds_df.iterrows():
            predicted_topic = pred_row["topic_id"]
            prob = pred_row["probability"]
            label = 1 if predicted_topic in true_topics else 0

            y_true_all.append(label)
            y_pred_all.append(int(prob >= 0.5))
            y_proba_all.append(prob)

    # Show metrics
    print_classification_metrics(y_true_all, y_pred_all, y_proba_all)
