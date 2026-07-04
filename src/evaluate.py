"""Évaluation des recommandations contre la vérité terrain ACPE.

Métriques du guide : Precision@K, Recall@K, NDCG@K (K = 5 et 10).
La vérité terrain (data/appariement.csv) associe 3 offres pertinentes
à chaque demandeur (pertinence binaire).

Usage :
    python src/evaluate.py data/recommendations_top10.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"


def load_ground_truth() -> dict:
    gt = pd.read_csv(DATA / "appariement.csv")
    return {
        r["id_demandeur"]: {r["id_offre1"], r["id_offre2"], r["id_offre3"]}
        for _, r in gt.iterrows()
    }


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    dcg = sum(1.0 / np.log2(i + 2) for i, job in enumerate(recommended[:k]) if job in relevant)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def evaluate(recs: pd.DataFrame, truth: dict, ks=(5, 10)) -> pd.DataFrame:
    recs = recs.sort_values(["candidate_id", "rank"])
    grouped = recs.groupby("candidate_id")["job_id"].apply(list)
    common = [c for c in grouped.index if c in truth]
    print(f"{len(common)} demandeurs évalués (présents dans recommandations ET vérité terrain)")

    results = {}
    for k in ks:
        precisions, recalls, ndcgs = [], [], []
        for cid in common:
            rec_k = grouped[cid][:k]
            rel = truth[cid]
            hits = len(set(rec_k) & rel)
            precisions.append(hits / k)
            recalls.append(hits / len(rel))
            ndcgs.append(ndcg_at_k(grouped[cid], rel, k))
        results[f"@{k}"] = {
            "Precision": np.mean(precisions),
            "Recall": np.mean(recalls),
            "NDCG": np.mean(ndcgs),
        }
    return pd.DataFrame(results).round(4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recommendations", help="CSV au format candidate_id,rank,job_id,score")
    args = ap.parse_args()
    recs = pd.read_csv(args.recommendations)
    truth = load_ground_truth()
    print(evaluate(recs, truth).to_string())


if __name__ == "__main__":
    main()
