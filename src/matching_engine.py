"""Hybrid matching engine: job seekers <-> job offers.

Combines three signals into one score:
  1. Skill overlap  — exact matching against the skills taxonomy (explainable)
  2. TF-IDF cosine  — lexical similarity of free text (CV vs. offer)
  3. Embeddings     — multilingual sentence embeddings (semantic), optional
plus structured bonuses (location, experience, education).

If sentence-transformers is not installed, the engine automatically
falls back to skill-overlap + TF-IDF only (weights are renormalized).
"""
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from skills_taxonomy import ALL_SKILLS, EDUCATION_LEVELS

# ---------------------------------------------------------------- embeddings
try:
    from sentence_transformers import SentenceTransformer
    _EMB_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
    _emb_model = None

    def _get_emb_model():
        global _emb_model
        if _emb_model is None:
            _emb_model = SentenceTransformer(_EMB_MODEL_NAME)
        return _emb_model

    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

FRENCH_STOPWORDS = [
    "au", "aux", "avec", "ce", "ces", "dans", "de", "des", "du", "elle", "en",
    "et", "eux", "il", "je", "la", "le", "les", "leur", "lui", "ma", "mais",
    "me", "meme", "mes", "moi", "mon", "ne", "nos", "notre", "nous", "on",
    "ou", "par", "pas", "pour", "qu", "que", "qui", "sa", "se", "ses", "son",
    "sur", "ta", "te", "tes", "toi", "ton", "tu", "un", "une", "vos", "votre",
    "vous", "d", "l", "s", "n", "c", "j", "m", "t", "y", "est", "sont", "ans",
]


def normalize(text: str) -> str:
    """Lowercase + strip accents + collapse whitespace."""
    text = str(text).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text).strip()


# Pre-normalized taxonomy for extraction
_NORM_SKILLS = {normalize(s): s for s in ALL_SKILLS}


def extract_skills(text: str) -> list:
    """Extract taxonomy skills mentioned in free text (accent-insensitive)."""
    norm = normalize(text)
    found = []
    for nskill, skill in _NORM_SKILLS.items():
        if re.search(rf"(?<![a-z0-9]){re.escape(nskill)}(?![a-z0-9])", norm):
            found.append(skill)
    return found


def _edu_rank(level: str) -> int:
    try:
        return EDUCATION_LEVELS.index(level)
    except ValueError:
        return 0


# Poids optimisés pour le classement contre la vérité terrain ACPE
# (validation sur 4 000 demandeurs : Recall@10 0.80 vs 0.47 pour le profil explicable)
RANKING_WEIGHTS = {"skills": 0.0, "tfidf": 1.0, "embeddings": 0.0, "structured": 0.0}


class MatchingEngine:
    """Fit once on the corpus of offers + seekers, then query in both directions."""

    def __init__(self, offers: pd.DataFrame, seekers: pd.DataFrame,
                 use_embeddings: bool = True,
                 weights: dict = None):
        self.offers = offers.reset_index(drop=True)
        self.seekers = seekers.reset_index(drop=True)
        self.use_embeddings = use_embeddings and EMBEDDINGS_AVAILABLE
        # Base weights of the hybrid score (explainability profile, used by the app).
        # For pure ranking against the ACPE ground truth, pass RANKING_WEIGHTS instead.
        if weights is None:
            weights = ({"skills": 0.40, "tfidf": 0.20, "embeddings": 0.25, "structured": 0.15}
                       if self.use_embeddings else
                       {"skills": 0.50, "tfidf": 0.30, "embeddings": 0.0, "structured": 0.20})
        self.weights = weights

        self._offer_texts = (self.offers["titre"] + ". " + self.offers["description"]).map(normalize).tolist()
        self._seeker_texts = self.seekers["cv_texte"].map(normalize).tolist()

        # sublinear_tf + min_df=2 : +0.13 de Recall@10 sur la vérité terrain ACPE
        self._vectorizer = TfidfVectorizer(stop_words=FRENCH_STOPWORDS,
                                           ngram_range=(1, 2), min_df=2,
                                           sublinear_tf=True)
        self._vectorizer.fit(self._offer_texts + self._seeker_texts)
        self._offer_tfidf = self._vectorizer.transform(self._offer_texts)
        self._seeker_tfidf = self._vectorizer.transform(self._seeker_texts)

        def _to_set(v):
            return {t.strip() for t in str(v).split(";") if t.strip() and t.strip() != "nan"}

        self._offer_skills = [_to_set(s) for s in self.offers["competences_requises"].fillna("")]
        self._seeker_skills = [_to_set(s) for s in self.seekers["competences"].fillna("")]

        if self.use_embeddings:
            model = _get_emb_model()
            self._offer_emb = model.encode(self._offer_texts, normalize_embeddings=True)
            self._seeker_emb = model.encode(self._seeker_texts, normalize_embeddings=True)

    # ------------------------------------------------------------ components
    @staticmethod
    def _skill_score(candidate_skills: set, required_skills: set) -> float:
        """Coverage of the offer's required skills by the candidate."""
        if not required_skills:
            return 0.0
        return len(candidate_skills & required_skills) / len(required_skills)

    def _structured_score(self, seeker_row, offer_row) -> float:
        score = 0.0
        # Location: same city 1.0, else 0.3 (mobility assumed possible)
        score += 0.4 * (1.0 if seeker_row["ville"] == offer_row["ville"] else 0.3)
        # Experience (critère neutre si l'offre ne le précise pas)
        exp, exp_min = seeker_row["annees_experience"], offer_row.get("experience_min")
        if exp_min is None or pd.isna(exp_min):
            score += 0.35
        elif exp >= exp_min:
            score += 0.35
        elif exp >= exp_min - 1:
            score += 0.20
        # Education (critère neutre si l'offre ne le précise pas)
        edu_min = offer_row.get("niveau_etudes_min")
        if edu_min is None or pd.isna(edu_min):
            score += 0.25
        elif _edu_rank(seeker_row["niveau_etudes"]) >= _edu_rank(edu_min):
            score += 0.25
        return score

    # --------------------------------------------------------------- queries
    def match_offers_for_seeker(self, seeker_idx: int = None,
                                cv_text: str = None, seeker_meta: dict = None,
                                top_n: int = 10) -> pd.DataFrame:
        """Rank offers for an indexed seeker OR an ad-hoc CV text."""
        if seeker_idx is not None:
            row = self.seekers.iloc[seeker_idx]
            skills = self._seeker_skills[seeker_idx]
            tfidf_vec = self._seeker_tfidf[seeker_idx]
            emb_vec = self._seeker_emb[seeker_idx] if self.use_embeddings else None
        else:
            norm_text = normalize(cv_text)
            skills = set(extract_skills(cv_text))
            tfidf_vec = self._vectorizer.transform([norm_text])
            emb_vec = _get_emb_model().encode([norm_text], normalize_embeddings=True)[0] if self.use_embeddings else None
            row = pd.Series({
                "ville": (seeker_meta or {}).get("ville", ""),
                "annees_experience": (seeker_meta or {}).get("annees_experience", 0),
                "niveau_etudes": (seeker_meta or {}).get("niveau_etudes", "Sans diplôme"),
            })

        tfidf_sims = cosine_similarity(tfidf_vec, self._offer_tfidf).ravel()
        emb_sims = (self._offer_emb @ emb_vec) if self.use_embeddings else np.zeros(len(self.offers))

        results = []
        for j in range(len(self.offers)):
            offer = self.offers.iloc[j]
            req = self._offer_skills[j]
            s_skill = self._skill_score(skills, req)
            s_struct = self._structured_score(row, offer)
            total = (self.weights["skills"] * s_skill
                     + self.weights["tfidf"] * tfidf_sims[j]
                     + self.weights["embeddings"] * float(emb_sims[j])
                     + self.weights["structured"] * s_struct)
            results.append({
                "id_offre": offer["id_offre"],
                "titre": offer["titre"],
                "entreprise": offer["entreprise"],
                "ville": offer["ville"],
                "type_contrat": offer["type_contrat"],
                "score": round(total * 100, 1),
                "score_competences": round(s_skill * 100, 1),
                "score_texte": round(float(tfidf_sims[j]) * 100, 1),
                "score_semantique": round(float(emb_sims[j]) * 100, 1),
                "competences_communes": ", ".join(sorted(skills & req)) or "—",
                "competences_manquantes": ", ".join(sorted(req - skills)) or "—",
            })
        return (pd.DataFrame(results)
                .sort_values("score", ascending=False)
                .head(top_n)
                .reset_index(drop=True))

    def search_candidates(self, query: str, top_n: int = 10) -> pd.DataFrame:
        """Recherche de candidats en langage naturel (Bonus 1).

        Exemple : "Je cherche un candidat en comptabilité avec une mobilité nationale."
        """
        norm_q = normalize(query)
        q_skills = set(extract_skills(query))
        sims = cosine_similarity(self._vectorizer.transform([norm_q]),
                                 self._seeker_tfidf).ravel()
        if q_skills:
            sk = np.array([len(q_skills & s) / len(q_skills) for s in self._seeker_skills])
            total = 0.6 * sims + 0.4 * sk
        else:
            total = sims
        idx = np.argsort(-total)[:top_n]
        rows = []
        for i in idx:
            seeker = self.seekers.iloc[int(i)]
            rows.append({
                "id_demandeur": seeker["id_demandeur"],
                "profil": seeker["titre_professionnel"],
                "ville": seeker["ville"],
                "niveau_etudes": seeker["niveau_etudes"],
                "score": round(float(total[i]) * 100, 1),
                "competences_correspondantes":
                    ", ".join(sorted(q_skills & self._seeker_skills[int(i)])) or "—",
            })
        return pd.DataFrame(rows)

    def match_seekers_for_offer(self, offer_idx: int, top_n: int = 10) -> pd.DataFrame:
        """Rank candidates for a given offer."""
        offer = self.offers.iloc[offer_idx]
        req = self._offer_skills[offer_idx]
        tfidf_sims = cosine_similarity(self._offer_tfidf[offer_idx], self._seeker_tfidf).ravel()
        emb_sims = (self._seeker_emb @ self._offer_emb[offer_idx]) if self.use_embeddings else np.zeros(len(self.seekers))

        results = []
        for i in range(len(self.seekers)):
            seeker = self.seekers.iloc[i]
            s_skill = self._skill_score(self._seeker_skills[i], req)
            s_struct = self._structured_score(seeker, offer)
            total = (self.weights["skills"] * s_skill
                     + self.weights["tfidf"] * tfidf_sims[i]
                     + self.weights["embeddings"] * float(emb_sims[i])
                     + self.weights["structured"] * s_struct)
            results.append({
                "id_demandeur": seeker["id_demandeur"],
                "nom": seeker["nom"],
                "titre_professionnel": seeker["titre_professionnel"],
                "ville": seeker["ville"],
                "annees_experience": seeker["annees_experience"],
                "niveau_etudes": seeker["niveau_etudes"],
                "score": round(total * 100, 1),
                "score_competences": round(s_skill * 100, 1),
                "competences_communes": ", ".join(sorted(self._seeker_skills[i] & req)) or "—",
            })
        return (pd.DataFrame(results)
                .sort_values("score", ascending=False)
                .head(top_n)
                .reset_index(drop=True))
