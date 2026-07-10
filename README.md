# 🇨🇬 EmploiMatch Congo

**Système intelligent d'appariement entre demandeurs d'emploi et offres d'emploi**
Prototype développé pour le **Hackathon IA — IndabaX Congo 2026 × ACPE**.

🔗 **Démo en ligne : [emploi-match-congo.streamlit.app](https://emploi-match-congo.streamlit.app)**
📄 Rapport technique : [`Rapport_EmploiMatch_Congo.pdf`](Rapport_EmploiMatch_Congo.pdf) · Licence : MIT

## 🎯 Objectif

Mettre automatiquement en relation les demandeurs d'emploi et les offres d'emploi
au Congo à l'aide de techniques de Data Science, de Machine Learning et de NLP,
dans les deux sens :

- **Demandeur → Offres** : un candidat colle son CV (texte libre) et obtient les offres les plus pertinentes, classées par score — couvre aussi le **Bonus 1 (recherche intelligente en langage naturel)**.
- **Employeur → Candidats** : un recruteur sélectionne une offre et obtient les candidats les mieux classés, ou recherche des candidats en langage naturel (« Je cherche un candidat en comptabilité avec une mobilité nationale »).

Les compétences manquantes sont affichées pour chaque recommandation
(**Bonus 2 — analyse des écarts de compétences / skill gap**).

## 🧠 Approche : score hybride

Le score final (0–100) combine quatre signaux complémentaires :

| Signal | Poids* | Technique | Rôle |
|---|---|---|---|
| Compétences | 40 % | Extraction par référentiel (137 compétences, 12 secteurs) | Explicable : compétences communes et manquantes affichées |
| Similarité lexicale | 20 % | TF-IDF (unigrammes + bigrammes) + cosinus | Capte le vocabulaire partagé CV/offre |
| Similarité sémantique | 25 % | Embeddings multilingues `paraphrase-multilingual-MiniLM-L12-v2` | Capte le sens même sans mots identiques |
| Critères structurés | 15 % | Règles (ville, expérience, niveau d'études) | Faisabilité pratique du recrutement |

\* Si `sentence-transformers` n'est pas installé, le moteur bascule automatiquement
en mode dégradé (compétences 50 % / TF-IDF 30 % / structuré 20 %) — aucun GPU requis.

Le texte est normalisé (minuscules, suppression des accents) et l'extraction de
compétences est insensible aux accents, ce qui la rend robuste aux CV mal saisis.

## 📁 Structure du projet

```
emploi-match/
├── app.py                    # Application web Streamlit (interface en français)
├── requirements.txt
├── data/
│   ├── raw/                  # Dataset officiel ACPE (xlsx)
│   ├── offres_acpe.csv       # 2 531 offres réelles ACPE, nettoyées et enrichies
│   ├── appariement.csv       # Vérité terrain : 41 285 demandeurs -> 3 offres
│   ├── demandeurs_acpe.csv   # 41 285 profils réels ACPE convertis
│   ├── recommendations_top5.csv / top10.csv   # Fichiers de soumission
│   ├── demandeurs.csv        # 400 profils synthétiques (démonstration)
│   └── offres.csv            # 200 offres synthétiques (secours)
└── src/
    ├── skills_taxonomy.py    # Référentiel compétences / villes / diplômes
    ├── prepare_data.py       # Nettoyage + fusion du dataset officiel ACPE
    ├── generate_data.py      # Générateur de données synthétiques
    ├── matching_engine.py    # Moteur d'appariement hybride
    ├── recommend.py          # Recommandations Top-K (format de soumission)
    └── evaluate.py           # Métriques : Precision@K, Recall@K, NDCG@K
```

## 🚀 Installation et lancement

```bash
pip install -r requirements.txt

# Préparer le dataset officiel ACPE (xlsx -> csv nettoyés)
python src/prepare_data.py

# (Optionnel) régénérer les profils demandeurs synthétiques
python src/generate_data.py --seekers 400 --offers 200

# Générer les recommandations Top-K (format de soumission du hackathon)
python src/recommend.py --top-k 10   # -> data/recommendations_top10.csv
python src/recommend.py --top-k 5    # -> data/recommendations_top5.csv

# Évaluer contre la vérité terrain (Precision@K, Recall@K, NDCG@K)
python src/evaluate.py data/recommendations_top10.csv

# Lancer l'application web
streamlit run app.py
```

L'application s'ouvre sur http://localhost:8501 avec trois onglets :
**Je cherche un emploi**, **Je recrute**, **Tableau de bord**.

## 🗂️ Données

### Dataset officiel ACPE

`src/prepare_data.py` fusionne les fichiers officiels en tables uniques :

- **Offres_ACPE.xlsx** — 2 535 offres (référence, intitulé, secteur, entreprise, lieu, contrat) ;
- **Offres_ACPE_Extensions.xlsx** — 143 offres enrichies (description, profil, compétences) ;
- **Appariement_Demandeurs_Offres.xlsx** — 41 285 demandeurs associés à 3 offres chacun.

Traitements appliqués : déduplication, normalisation des lieux, extraction par
expressions régulières du niveau d'études (« Bac+2/3 » → BAC+2) et de l'expérience
minimale (« 2 à 5 ans » → 2) depuis le champ Profil, et extraction automatique des
compétences par référentiel pour les offres sans champ Compétences
(1 350 offres sur 2 531 obtiennent ainsi des compétences structurées).

### Profils demandeurs (Demandeurs.xlsx)

Les 41 298 profils réels (41 285 après déduplication) sont convertis en
`data/demandeurs_acpe.csv` : département décodé depuis le préfixe du matricule
(PPBZV → Brazzaville, PPPNR → Pointe-Noire…), niveaux d'études projetés sur une
échelle ordonnée (« Bac +3 » → Licence), texte de profil composé à partir du
métier visé, de la qualification, de la filière, des secteurs et du diplôme.
L'extraction par référentiel détecte des compétences pour 29 620 profils (72 %).

## 🏆 Résultats officiels (41 285 demandeurs réels × 2 531 offres)

| Métrique | @5 | @10 |
|---|---|---|
| Precision (max. théorique 0,60 / 0,30) | 0,366 | 0,230 |
| Recall | 0,610 | **0,768** |
| NDCG | 0,597 | 0,669 |

77 % des offres jugées pertinentes par l'ACPE figurent dans le Top-10 du moteur.
Poids du profil « classement » calibrés par validation sur 4 000 demandeurs
(`RANKING_WEIGHTS` dans `src/matching_engine.py`) ; le profil « conseiller »
hybride reste utilisé dans l'application pour l'explicabilité.

## 🏁 Conformité au guide du hackathon

| Exigence du guide | Où |
|---|---|
| Préparation / exploration / nettoyage documentés | `src/prepare_data.py` + ce README |
| Moteur d'appariement (méthode libre, justifiée) | `src/matching_engine.py` (hybride, voir tableau des poids) |
| Score de compatibilité expliqué (variables contributrices) | Sous-scores affichés : compétences, texte, sémantique + compétences communes/manquantes |
| Recommandations Top-5 / Top-10, format `candidate_id, rank, job_id, score` | `src/recommend.py` |
| Évaluation Precision@5/10, Recall@5/10, NDCG@5/10 | `src/evaluate.py` (contre `data/appariement.csv`) |
| Tableau de bord décisionnel (candidats, offres, secteurs, métiers, taux moyen, géographie, stats reco) | Onglet « Tableau de bord » de l'app |
| Interface de démonstration (Streamlit) | `app.py` |
| Bonus 1 — recherche en langage naturel | Onglet demandeur : texte libre → offres pertinentes sans mots-clés exacts |
| Bonus 2 — skill gap | Colonne « compétences manquantes » de chaque recommandation |

## 📈 Pistes d'amélioration

Extraction de CV PDF (OCR), apprentissage des poids du score à partir des
recrutements réussis (learning-to-rank), prise en compte du lingala/kituba,
API REST (FastAPI) pour intégration au SI de l'ACPE, et déduplication des offres.

## 👤 Auteur

Glenn Matondo — Hackathon IndabaX Congo 2026 × ACPE.
