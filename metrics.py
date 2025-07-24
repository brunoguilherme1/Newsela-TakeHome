# metrics.py

import numpy as np
import pandas as pd
from typing import List, Dict, Union
from collections import defaultdict


class TopicRecommendationEvaluator:
    """
    Evaluation class for topic recommendations.
    Accepts y_true and y_pred as lists of lists or dictionaries.
    """

    def __init__(self, csv_path: str = None, df: pd.DataFrame = None):
        self.df = None
        if df is not None:
            self.df = self._prepare_data(df)
        elif csv_path:
            self.df = self._prepare_data(pd.read_csv(csv_path))

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        return df  # Placeholder for optional pre-processing

    def precision_at_k(self, y_true: List[str], y_pred: List[str], k: int) -> float:
        if k == 0 or not y_pred:
            return 0.0
        top_k = y_pred[:k]
        relevant = [t for t in top_k if t in y_true]
        return len(relevant) / min(k, len(top_k))

    def recall_at_k(self, y_true: List[str], y_pred: List[str], k: int) -> float:
        if not y_true:
            return 0.0
        top_k = y_pred[:k]
        relevant = [t for t in top_k if t in y_true]
        return len(relevant) / len(y_true)

    def f1_at_k(self, y_true: List[str], y_pred: List[str], k: int) -> float:
        precision = self.precision_at_k(y_true, y_pred, k)
        recall = self.recall_at_k(y_true, y_pred, k)
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def mrr_at_k(self, y_true: List[str], y_pred: List[str], k: int) -> float:
        top_k = y_pred[:k]
        for i, topic in enumerate(top_k, 1):
            if topic in y_true:
                return 1.0 / i
        return 0.0

    def ndcg_at_k(self, y_true: List[str], y_pred: List[str], k: int) -> float:
        def dcg(relevances):
            return sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(relevances))

        top_k = y_pred[:k]
        relevances = [1 if topic in y_true else 0 for topic in top_k]

        if sum(relevances) == 0:
            return 0.0

        dcg_score = dcg(relevances)
        idcg_score = dcg([1] * min(len(y_true), k))

        return dcg_score / idcg_score if idcg_score > 0 else 0.0

    def map_score(self, y_true: List[str], y_pred: List[str]) -> float:
        if not y_true:
            return 0.0
        precisions = []
        relevant_count = 0
        for i, topic in enumerate(y_pred, 1):
            if topic in y_true:
                relevant_count += 1
                precisions.append(relevant_count / i)
        return sum(precisions) / len(y_true) if precisions else 0.0

    def coverage_at_k(self, y_true: List[str], y_pred: List[str], k: int) -> float:
        top_k = y_pred[:k]
        return 1.0 if any(topic in y_true for topic in top_k) else 0.0

    def hits_at_k(self, y_true: List[str], y_pred: List[str], k: int) -> float:
        return self.coverage_at_k(y_true, y_pred, k)

    def evaluate(self,
                 y_true: Union[List[List[str]], Dict[str, List[str]]],
                 y_pred: Union[List[List[str]], Dict[str, List[str]]],
                 k_values: List[int] = [1, 3, 5, 10, 50]) -> Dict[str, Dict[int, float]]:

        if isinstance(y_true, dict) and isinstance(y_pred, dict):
            common_ids = list(set(y_true.keys()) & set(y_pred.keys()))
            true_lists = [y_true[id_] for id_ in common_ids]
            pred_lists = [y_pred[id_] for id_ in common_ids]
        elif isinstance(y_true, list) and isinstance(y_pred, list):
            if len(y_true) != len(y_pred):
                raise ValueError("y_true and y_pred must have the same length")
            true_lists = y_true
            pred_lists = y_pred
        else:
            raise ValueError("Both y_true and y_pred must be either lists or dicts")

        results = defaultdict(lambda: defaultdict(list))

        for true_topics, pred_topics in zip(true_lists, pred_lists):
            true_topics = list(true_topics)
            pred_topics = list(pred_topics)

            results['MAP']['all'].append(self.map_score(true_topics, pred_topics))

            for k in k_values:
                results['Precision@k'][k].append(self.precision_at_k(true_topics, pred_topics, k))
                results['Recall@k'][k].append(self.recall_at_k(true_topics, pred_topics, k))
                results['F1@k'][k].append(self.f1_at_k(true_topics, pred_topics, k))
                results['MRR@k'][k].append(self.mrr_at_k(true_topics, pred_topics, k))
                results['NDCG@k'][k].append(self.ndcg_at_k(true_topics, pred_topics, k))
                results['Coverage@k'][k].append(self.coverage_at_k(true_topics, pred_topics, k))
                results['Hits@k'][k].append(self.hits_at_k(true_topics, pred_topics, k))

        # Average each metric
        final_results = {}
        for metric, k_dict in results.items():
            final_results[metric] = {}
            for k, values in k_dict.items():
                final_results[metric][k] = np.mean(values)

        return final_results

    def format_results(self, results: Dict[str, Dict[int, float]]) -> pd.DataFrame:
        formatted_data = []
        k_values = [1, 3, 5, 10, 50]

        for metric in ['Precision@k', 'Recall@k', 'F1@k', 'MRR@k', 'NDCG@k', 'Coverage@k', 'Hits@k']:
            if metric in results:
                row = {'Metric': metric}
                for k in k_values:
                    row[f'@{k}'] = f"{results[metric].get(k, 0):.4f}"
                formatted_data.append(row)

        if 'MAP' in results:
            row = {'Metric': 'MAP', '@1': "N/A", '@3': "N/A", '@5': "N/A", '@10': "N/A", '@50': "N/A"}
            row['Value'] = f"{results['MAP'].get('all', 0):.4f}"
            formatted_data.append(row)

        return pd.DataFrame(formatted_data)


from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score
)

def print_classification_metrics(y_true, y_pred, y_proba=None):
    print("\n📊 Classification Metrics (binary average):")
    print(f"Precision:  {precision_score(y_true, y_pred, average='binary'):.4f}")
    print(f"Recall:     {recall_score(y_true, y_pred, average='binary'):.4f}")
    print(f"F1 Score:   {f1_score(y_true, y_pred, average='binary'):.4f}")
    print(f"F2 Score:   {fbeta_score(y_true, y_pred, average='binary', beta=2):.4f}")
    
    if y_proba is not None:
        print(f"ROC AUC:    {roc_auc_score(y_true, y_proba):.4f}")

def evaluate_predictions(y_true, y_pred, k_values=[1, 3, 5, 10, 50]):
    evaluator = TopicRecommendationEvaluator()
    results = evaluator.evaluate(y_true, y_pred, k_values)
    return evaluator.format_results(results)