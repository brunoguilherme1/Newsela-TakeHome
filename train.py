# train.py

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn import FunctionSampler
from sentence_transformers import SentenceTransformer
from metrics import evaluate_predictions 
from gensim.models import Word2Vec
import faiss
import joblib
import swifter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, fbeta_score
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import numpy as np

import torch
torch.set_num_threads(1)

from utils import (
    build_text,
    load_combined_stopwords,
    fit_tfidf,
    encode_text_with_tfidf_word2vec,
    count_overlap_and_lengths
)

print("🚀 Starting training pipeline...")

# ------------------------
# ✅ Load and clean data
# ------------------------
print("📊 Loading data files...")
content = pd.read_csv("data/content.csv")
topics = pd.read_csv("data/topics.csv")
correlations = pd.read_csv("data/correlations.csv")
print("🧹 Cleaning data...")
for col in ["title", "description", "channel", "category", "language", "parent"]:
    topics[col] = topics[col].fillna("")
for col in ["title", "description", "text", "kind", "language"]:
    content[col] = content[col].fillna("")

topics["text"] = topics[["title", "description"]].agg(" ".join, axis=1)
content["final_text"] = content.apply(build_text, axis=1)
correlations["content_ids"] = correlations["content_ids"].str.split()

# ------------------------
# ✅ Split and embed
# ------------------------
print("🔀 Splitting data (80/20)...")
content_80, _ = train_test_split(content, test_size=0.2, stratify=content["language"], random_state=42)
import numpy as np
encoder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
content_embeddings = np.random.randn(len(content_80), 384).astype('float32')#encoder.encode(content_80["final_text"].tolist(), batch_size=128, show_progress_bar=True)
topic_embeddings = np.random.randn(len(topics), 384).astype('float32')#encoder.encode(topics["text"].tolist(), batch_size=128, show_progress_bar=True)

# ------------------------
# ✅ FAISS Retrieval
# ------------------------
print("🔍 Building FAISS index...")
topic_embeddings_f32 = np.array(topic_embeddings).astype("float32")
content_embeddings_f32 = np.array(content_embeddings).astype("float32")

index = faiss.IndexFlatL2(topic_embeddings_f32.shape[1])
index.add(topic_embeddings_f32)

print("Here")

k = 50
# ✅ Ensure correct float32 dtype
topic_embeddings_f32 = np.array(topic_embeddings).astype(np.float32)
content_embeddings_f32 = np.array(content_embeddings).astype(np.float32)

# ✅ Safety checks
assert topic_embeddings_f32.ndim == 2, "topic embeddings must be 2D"
assert content_embeddings_f32.ndim == 2, "content embeddings must be 2D"
assert topic_embeddings_f32.dtype == np.float32, "topic embeddings must be float32"
assert content_embeddings_f32.dtype == np.float32, "content embeddings must be float32"
assert topic_embeddings_f32.shape[1] == content_embeddings_f32.shape[1], "dimension mismatch"

# ✅ Cap k to number of topic embeddings to avoid overflow
k = min(50, topic_embeddings_f32.shape[0])

# ✅ Build FAISS index (FlatL2 for exact nearest neighbors)
index = faiss.IndexFlatL2(topic_embeddings_f32.shape[1])

index.add(topic_embeddings_f32)

# ✅ Run FAISS search safely
try:
    distances, indices = index.search(content_embeddings_f32, k)
    print("Here2 ")
    print("✅ FAISS search successful")
except Exception as e:
    print("❌ FAISS search failed:", e)
    print("▶ content_embeddings_f32 shape:", content_embeddings_f32.shape)
    print("▶ topic_embeddings_f32 shape:", topic_embeddings_f32.shape)
    raise

topic_ids = topics["id"].tolist()
content_ids = content_80["id"].tolist()
print("Here2")
y_pred_dict = {
    cid: [topic_ids[i] for i in idx_row]
    for cid, idx_row in zip(content_ids, indices)
}

correlations_exploded = correlations.explode("content_ids").dropna()
y_true_dict = correlations_exploded.groupby("content_ids")["topic_id"].apply(list)

common_ids = sorted(set(y_true_dict.index) & set(y_pred_dict.keys()))
correlation_set = set(zip(correlations_exploded["content_ids"], correlations_exploded["topic_id"]))

content_text_dict = dict(zip(content_80["id"], content_80["final_text"]))
topic_text_dict = dict(zip(topics["id"], topics["text"]))
index_to_topic = {i: tid for i, tid in enumerate(topic_ids)}
content_id_array = np.array(content_ids)

# Prepare y_true_dict (exploded correlations)
df = correlations.copy(deep=True)
df_exploded = df.explode("content_ids")
y_true_dict = df_exploded.groupby("content_ids")["topic_id"].apply(list)

# Only compare on content IDs with ground truth
common_content_ids = sorted(set(y_true_dict.index).intersection(y_pred_dict.keys()))
y_true = [y_true_dict[cid] for cid in common_content_ids]
y_pred = [y_pred_dict[cid] for cid in common_content_ids]

# Print retrieval metrics
print("\n📊 Retrieval Evaluation (Top-50 FAISS Results):")
metrics_df = evaluate_predictions(y_true, y_pred, k_values=[1, 3, 5, 10, 50])
print(metrics_df.to_string(index=False))

# ------------------------
# ✅ Create training pairs
# ------------------------
print("🔗 Creating content-topic pairs...")
data = []
for content_idx, topic_ranked_indices in enumerate(indices):
    cid = content_id_array[content_idx]
    for tidx in topic_ranked_indices:
        tid = index_to_topic[tidx]
        label = 1 if (cid, tid) in correlation_set else 0
        data.append((cid, tid, label, content_text_dict[cid], topic_text_dict[tid]))

df_pairs = pd.DataFrame(data, columns=["content_id", "topic_id", "label", "final_text", "text"])

# ------------------------
# ✅ Train Word2Vec + TF-IDF
# ------------------------
print("📝 Training Word2Vec and TF-IDF...")
tokenized = topics["text"].apply(lambda x: x.lower().split())
w2v = Word2Vec(tokenized.tolist(), vector_size=32, window=2, min_count=1, sg=0, workers=4)

stopwords_set = load_combined_stopwords(topics["language"].unique().tolist())
tfidf_vectorizer, idf_dict = fit_tfidf(topics["text"].tolist(), stop_words=list(stopwords_set))

content_80["word2vec_embedding"] = content_80["final_text"].swifter.apply(
    lambda x: encode_text_with_tfidf_word2vec(x, w2v, idf_dict)
)
topics["word2vec_embedding"] = topics["text"].swifter.apply(
    lambda x: encode_text_with_tfidf_word2vec(x, w2v, idf_dict)
)

content_embed_dict = dict(zip(content_80["id"], content_80["word2vec_embedding"]))
topic_embed_dict = dict(zip(topics["id"], topics["word2vec_embedding"]))

df_pairs["content_vec"] = df_pairs["content_id"].map(content_embed_dict)
df_pairs["topic_vec"] = df_pairs["topic_id"].map(topic_embed_dict)
df_pairs.dropna(subset=["content_vec", "topic_vec"], inplace=True)

# ------------------------
# ✅ Feature Engineering
# ------------------------
print("⚙️ Engineering features...")
content_vecs = np.vstack(df_pairs["content_vec"].values)
topic_vecs = np.vstack(df_pairs["topic_vec"].values)

dot_products = np.sum(content_vecs * topic_vecs, axis=1)
norms_content = np.linalg.norm(content_vecs, axis=1)
norms_topic = np.linalg.norm(topic_vecs, axis=1)
cosine_sims = dot_products / (norms_content * norms_topic + 1e-9)
euclidean_dists = np.linalg.norm(content_vecs - topic_vecs, axis=1)
abs_diff_means = np.mean(np.abs(content_vecs - topic_vecs), axis=1)

word_stats = np.array([
    count_overlap_and_lengths(a, b)
    for a, b in zip(df_pairs["final_text"], df_pairs["text"])
])

df_pairs["cosine_sim"] = cosine_sims
df_pairs["euclidean_dist"] = euclidean_dists
df_pairs["abs_diff_mean"] = abs_diff_means
df_pairs["dot_product"] = dot_products
df_pairs["word_intersection"] = word_stats[:, 0]
df_pairs["len_content_words"] = word_stats[:, 1]
df_pairs["len_topic_words"] = word_stats[:, 2]

# ------------------------
# ✅ Final feature matrix
# ------------------------
print("🏗️ Building feature matrix...")
X_embed = np.hstack([content_vecs, topic_vecs])
X_extra = df_pairs[[
    "cosine_sim", "euclidean_dist", "abs_diff_mean", "dot_product",
    "word_intersection", "len_content_words", "len_topic_words"
]].values

X = np.hstack([X_embed, X_extra])
y = df_pairs["label"].values

# ------------------------
# ✅ Train LightGBM pipeline
# ------------------------
print("🚂 Training LightGBM model with grid search...")

best_f2 = -1
best_model = None

print("\n🚀 Running manual 5-Fold CV search:")
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for z in [50,100]:               # LightGBM estimators
    for x in [0.05]:        # SMOTE sampling
        f2_scores = []

        for train_idx, val_idx in kf.split(X, y):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            smote = SMOTE(sampling_strategy=x, random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train_fold, y_train_fold)

            model = LGBMClassifier(n_estimators=z, random_state=42)
            model.fit(X_train_res, y_train_res)

            y_pred = model.predict(X_val_fold)
            f2 = fbeta_score(y_val_fold, y_pred, beta=2, average='binary')
            f2_scores.append(f2)

        avg_f2 = np.mean(f2_scores)
        print(f" - SMOTE: {x}, n_estimators: {z}, Avg F2: {avg_f2:.4f}")

        if avg_f2 > best_f2:
            best_f2 = avg_f2
            best_model = model

print(f"\n✅ Best Model: F2 = {best_f2:.4f}")


# ------------------------
# ✅ Save artifacts
# ------------------------
print("💾 Saving model artifacts...")
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("sentence_model", exist_ok=True)

joblib.dump(model, "models/lgbm_model.pkl")
joblib.dump(tfidf_vectorizer, "models/tfidf.pkl")
w2v.save("models/word2vec_topics.model")
encoder.save("sentence_model")

np.save("data/topic_embeddings.npy", topic_embeddings)
topics.to_csv("data/topics.csv", index=False)
faiss.write_index(index, "faiss_index.index")

print("✅ Training complete. Artifacts saved.")
