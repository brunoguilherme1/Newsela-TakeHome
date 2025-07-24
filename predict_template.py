from typing import Optional, List
from pydantic import BaseModel

import numpy as np
import pandas as pd
import joblib
import faiss
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer

from utils import encode_text_with_tfidf_word2vec, count_overlap_and_lengths


# ✅ Request schema for input content and optional topic metadata
class TopicPredictionRequest(BaseModel):
    content_title: Optional[str] = None
    content_description: Optional[str] = None
    content_kind: Optional[str] = None
    content_text: Optional[str] = None
    topic_title: Optional[str] = None
    topic_description: Optional[str] = None
    topic_category: Optional[str] = None


# ✅ Prediction logic: loads models, retrieves top-K topics, ranks them with LGBM
class TopicPredictor:
    def __init__(
        self,
        sentence_model_path: str = "sentence_model",
        w2v_path: str = "models/word2vec_topics.model",
        tfidf_path: str = "models/tfidf.pkl",
        idf_dict_path: str = "models/idf_dict.pkl",
        lgbm_path: str = "models/lgbm_model.pkl",
        topics_path: str = "data/topics.csv",
        topic_embeddings_path: str = "data/topic_embeddings.npy",
        faiss_index_path: str = "models/faiss_index.index"
    ):
        print("🔄 Loading model artifacts...")

        self.encoder = SentenceTransformer(sentence_model_path)
        self.w2v = Word2Vec.load(w2v_path)
        self.tfidf = joblib.load(tfidf_path)
        self.idf_dict = joblib.load(idf_dict_path)
        self.lgbm = joblib.load(lgbm_path)

        self.topics = pd.read_csv(topics_path)
        self.topic_embeddings = np.load(topic_embeddings_path)
        self.index = faiss.read_index(faiss_index_path)

        self.topic_ids = self.topics["id"].tolist()
        self.topic_texts = dict(zip(self.topics["id"], self.topics["text"]))

        print("✅ TopicPredictor initialized.")

    # ✅ Predict all relevant topic_ids (prob > 0.5) given a content request
    def predict(self, request: TopicPredictionRequest, top_k: int = 50) -> List[str]:
        # Construct the query text from title, description, and main content
        query_parts = [
            request.content_title or "",
            request.content_description or "",
            request.content_text or "",
        ]
        query = " ".join(query_parts).strip()

        if not query:
            print("⚠️ Empty query provided.")
            return []

        # Encode content using both SentenceTransformer and Word2Vec+TF-IDF
        query_embedding = self.encoder.encode([query])[0].astype("float32")
        query_vec = encode_text_with_tfidf_word2vec(query, self.w2v, self.idf_dict)

        # Retrieve top-K most similar topics using FAISS over transformer embeddings
        _, indices = self.index.search(query_embedding.reshape(1, -1), top_k)
        candidate_ids = [self.topic_ids[i] for i in indices[0]]

        # Build features for each (content, topic) pair
        feature_matrix = []
        for tid in candidate_ids:
            topic_text = self.topic_texts.get(tid, "")
            topic_vec = encode_text_with_tfidf_word2vec(topic_text, self.w2v, self.idf_dict)

            dot = np.dot(query_vec, topic_vec)
            cosine = dot / (np.linalg.norm(query_vec) * np.linalg.norm(topic_vec) + 1e-9)
            euclidean = np.linalg.norm(query_vec - topic_vec)
            abs_diff_mean = np.mean(np.abs(query_vec - topic_vec))
            intersection, len_q, len_t = count_overlap_and_lengths(query, topic_text)

            features = np.concatenate([
                query_vec, topic_vec,
                [cosine, euclidean, abs_diff_mean, dot, intersection, len_q, len_t]
            ])
            feature_matrix.append(features)

        # Predict relevance probabilities for each pair using LightGBM
        X = np.vstack(feature_matrix)
        probs = self.lgbm.predict_proba(X)[:, 1]

        # Return topic_ids where probability > 0.5 (binary classification)
        relevant_topic_ids = [
            tid for tid, prob in zip(candidate_ids, probs) if prob > 0.5
        ]
        return relevant_topic_ids


# ------------------------
# ✅ Example test (manual)
# ------------------------
if __name__ == "__main__":
    predictor = TopicPredictor()

    request = TopicPredictionRequest(
        content_title="Newton's law of gravitation,Why are you sticking to your chair (ignoring the spilled glue)?  Why does the earth orbit the sun (or does it)?",
        content_description="Gravitation defines our everyday life and the structure of the universe.  This tutorial will introduce it to you in the Newtonian sense.",
        content_text="math"
    )

    predicted_topic_ids = predictor.predict(request)
    print("Predicted topic IDs:", predicted_topic_ids)
