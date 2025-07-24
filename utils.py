# utils.py

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from tqdm import tqdm
from gensim.utils import simple_preprocess
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import stopwordsiso as stopwords
from gensim.models import Word2Vec
from metrics import TopicRecommendationEvaluator

# ----------------------------------------
# ✅ Utility to build full text from columns
# ----------------------------------------
def build_text(row: pd.Series) -> str:
    return " ".join([str(row.get("title", "")), str(row.get("description", "")), str(row.get("text", ""))])


# ----------------------------------------
# ✅ Load multilingual stopwords
# ----------------------------------------
def load_combined_stopwords(languages: List[str]) -> set:
    combined_stops = set()
    for lang in languages:
        try:
            combined_stops.update(stopwords.stopwords(lang))
        except:
            print(f"⚠️ Could not load stopwords for language: {lang}")
    return combined_stops


# ----------------------------------------
# ✅ Fit TF-IDF and return vocab + idf dict
# ----------------------------------------
def fit_tfidf(texts: List[str], stop_words: List[str], max_features: int = 20000):
    tfidf = TfidfVectorizer(max_features=max_features, stop_words=stop_words)
    tfidf_matrix = tfidf.fit_transform(texts)
    idf_values = tfidf.idf_
    vocab = tfidf.vocabulary_
    idf_dict = {word: idf_values[idx] for word, idx in vocab.items()}
    return tfidf, idf_dict


# ----------------------------------------
# ✅ Encode text using TF-IDF weighted Word2Vec
# ----------------------------------------
def encode_text_with_tfidf_word2vec(text: str, model: Word2Vec, idf_dict: Dict[str, float]) -> np.ndarray:
    tokens = simple_preprocess(text, deacc=True)
    vecs = []

    for word in tokens:
        if word in model.wv:
            tfidf_weight = idf_dict.get(word, 1.0)
            vecs.append(tfidf_weight * model.wv[word])

    if vecs:
        return np.mean(vecs, axis=0)
    else:
        return np.zeros(model.vector_size)


# ----------------------------------------
# ✅ Compute overlap features between two texts
# ----------------------------------------
def count_overlap_and_lengths(text_a: str, text_b: str) -> Tuple[int, int, int]:
    words_a = set(str(text_a).lower().split())
    words_b = set(str(text_b).lower().split())
    return (
        len(words_a & words_b),
        len(words_a),
        len(words_b)
    )


# ----------------------------------------
# ✅ Evaluation wrapper
# ----------------------------------------
def evaluate_predictions(y_true: List[List[str]],
                         y_pred: List[List[str]],
                         evaluator_cls=None,
                         k_values: List[int] = [1, 3, 5, 10, 50]) -> pd.DataFrame:
    if evaluator_cls is None:
        from utils import TopicRecommendationEvaluator  # Local import to avoid circular
        evaluator_cls = TopicRecommendationEvaluator

    evaluator = evaluator_cls()
    results = evaluator.evaluate(y_true, y_pred, k_values)
    return evaluator.format_results(results)
