"""Génération des recommandations Top-K au format de soumission du hackathon.

Format de sortie (cf. guide, section V.5) :
    candidate_id, rank, job_id, score

Calcul vectorisé (matrices creuses + numpy) pour traiter des dizaines de
milliers de candidats contre des milliers d'offres en quelques secondes.

Usage :
    python src/recommend.py --top-k 10 --out data/recommendations_top10.csv
    python src/recommend.py --top-k 5  --out data/recommendations_top5.csv

Les profils demandeurs sont lus dans data/demandeurs_acpe.csv s'il existe
(données réelles ACPE converties par prepare_data.py), sinon
data/demandeurs.csv (profils synthétiques de démonstration).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from matching_engine import MatchingEngine, EMBEDDINGS_AVAILABLE, RANKING_WEIGHTS

DATA = Path(__file__).resolve().parent.parent / "data"


def batch_topk(engine: MatchingEngine, top_k: int = 10, chunk: int = 2000) -> pd.DataFrame:
    """Score vectorisé de tous les demandeurs contre toutes les offres, Top-K par demandeur."""
    n_seek, n_off = len(engine.seekers), len(engine.offers)
    w = engine.weights

    # --- composante compétences (couverture des compétences requises)
    mlb = MultiLabelBinarizer()
    mlb.fit(engine._seeker_skills + engine._offer_skills)
    S = mlb.transform(engine._seeker_skills).astype(np.float32)          # (n_seek, k)
    O = mlb.transform(engine._offer_skills).astype(np.float32)          # (n_off, k)
    req_counts = np.maximum(O.sum(axis=1), 1.0)                          # évite /0

    # --- composantes structurées (vectorisées)
    seek_city = engine.seekers["ville"].astype(str).to_numpy()
    off_city = engine.offers["ville"].astype(str).to_numpy()
    seek_exp = engine.seekers["annees_experience"].fillna(0).to_numpy(dtype=float)
    off_exp = pd.to_numeric(engine.offers.get("experience_min"), errors="coerce").to_numpy(dtype=float)
    from skills_taxonomy import EDUCATION_LEVELS
    rank = {lvl: i for i, lvl in enumerate(EDUCATION_LEVELS)}
    seek_edu = engine.seekers["niveau_etudes"].map(rank).fillna(0).to_numpy(dtype=float)
    off_edu = engine.offers.get("niveau_etudes_min", pd.Series([None] * n_off)).map(rank).to_numpy(dtype=float)

    seek_ids = engine.seekers["id_demandeur"].to_numpy()
    off_ids = engine.offers["id_offre"].to_numpy()
    rows = []
    for start in range(0, n_seek, chunk):
        end = min(start + chunk, n_seek)
        # 1. compétences
        skill_scores = (S[start:end] @ O.T) / req_counts[None, :]
        # 2. TF-IDF
        tfidf_scores = (engine._seeker_tfidf[start:end] @ engine._offer_tfidf.T).toarray()
        # 3. embeddings (déjà normalisés -> produit scalaire = cosinus)
        emb_scores = (engine._seeker_emb[start:end] @ engine._offer_emb.T
                      if engine.use_embeddings else 0.0)
        # 4. structuré : ville + expérience + niveau d'études
        city = np.where(seek_city[start:end, None] == off_city[None, :], 1.0, 0.3) * 0.4
        exp_ok = np.where(np.isnan(off_exp)[None, :], 0.35,
                          np.where(seek_exp[start:end, None] >= off_exp[None, :], 0.35,
                                   np.where(seek_exp[start:end, None] >= off_exp[None, :] - 1, 0.20, 0.0)))
        edu_ok = np.where(np.isnan(off_edu)[None, :], 0.25,
                          np.where(seek_edu[start:end, None] >= off_edu[None, :], 0.25, 0.0))
        struct = city + exp_ok + edu_ok

        total = (w["skills"] * skill_scores + w["tfidf"] * tfidf_scores
                 + w["embeddings"] * emb_scores + w["structured"] * struct)

        # Top-K par ligne
        k = min(top_k, n_off)
        idx = np.argpartition(-total, k - 1, axis=1)[:, :k]
        for i, cand_row in enumerate(idx):
            order = cand_row[np.argsort(-total[i, cand_row])]
            cid = seek_ids[start + i]
            for r, j in enumerate(order, start=1):
                rows.append((cid, r, off_ids[j], round(float(total[i, j]), 4)))
        print(f"  {end}/{n_seek} demandeurs traités", flush=True)

    return pd.DataFrame(rows, columns=["candidate_id", "rank", "job_id", "score"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    offers = pd.read_csv(DATA / ("offres_acpe.csv" if (DATA / "offres_acpe.csv").exists() else "offres.csv"))
    seekers_path = DATA / ("demandeurs_acpe.csv" if (DATA / "demandeurs_acpe.csv").exists() else "demandeurs.csv")
    seekers = pd.read_csv(seekers_path)
    print(f"{len(seekers)} demandeurs ({seekers_path.name}), {len(offers)} offres, "
          f"embeddings={'oui' if EMBEDDINGS_AVAILABLE else 'non'}")

    engine = MatchingEngine(offers, seekers, weights=RANKING_WEIGHTS)
    recs = batch_topk(engine, top_k=args.top_k)
    out = Path(args.out) if args.out else DATA / f"recommendations_top{args.top_k}.csv"
    recs.to_csv(out, index=False)
    print(f"{len(recs)} lignes -> {out}")


if __name__ == "__main__":
    main()
