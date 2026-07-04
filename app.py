"""EmploiMatch Congo — Streamlit prototype (Hackathon IndabaX Congo 2026 x ACPE).

Interface en français. Lancer avec :  streamlit run app.py
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from matching_engine import MatchingEngine, EMBEDDINGS_AVAILABLE, extract_skills  # noqa: E402
from skills_taxonomy import CITIES, EDUCATION_LEVELS  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"

st.set_page_config(page_title="EmploiMatch Congo", page_icon="🇨🇬", layout="wide")


@st.cache_resource(show_spinner="Chargement du moteur d'appariement…")
def load_engine():
    acpe = DATA_DIR / "offres_acpe.csv"
    offers = pd.read_csv(acpe if acpe.exists() else DATA_DIR / "offres.csv")
    dem_acpe = DATA_DIR / "demandeurs_acpe.csv"
    if dem_acpe.exists():
        # Échantillon de profils réels pour garder la démo fluide
        seekers = (pd.read_csv(dem_acpe)
                   .sample(n=5000, random_state=42)
                   .reset_index(drop=True))
        real_seekers = True
    else:
        seekers = pd.read_csv(DATA_DIR / "demandeurs.csv")
        real_seekers = False
    return MatchingEngine(offers, seekers), acpe.exists(), real_seekers


engine, REAL_DATA, REAL_SEEKERS = load_engine()

st.title("🇨🇬 EmploiMatch Congo")
st.caption(
    "Système intelligent d'appariement entre demandeurs d'emploi et offres d'emploi — "
    "Hackathon IndabaX Congo 2026 × ACPE. "
    + ("Offres : dataset officiel ACPE (2 531 offres réelles). "
       if REAL_DATA else "Offres : données synthétiques de démonstration. ")
    + ("Candidats : 5 000 profils réels ACPE (échantillon anonymisé). "
       if REAL_SEEKERS else "Candidats : profils synthétiques. ")
    + ("Mode : hybride (compétences + TF-IDF + embeddings sémantiques)."
       if EMBEDDINGS_AVAILABLE else
       "Mode : compétences + TF-IDF (installez sentence-transformers pour le mode sémantique).")
)

tab_seeker, tab_employer, tab_stats = st.tabs(
    ["🔎 Je cherche un emploi", "🏢 Je recrute", "📊 Tableau de bord"]
)

# ------------------------------------------------------------------ seeker
with tab_seeker:
    st.subheader("Trouvez les offres qui correspondent à votre profil")
    mode = st.radio("Source du profil", ["Saisir mon CV librement", "Choisir un profil de démonstration"],
                    horizontal=True)

    if mode == "Saisir mon CV librement":
        col1, col2 = st.columns([2, 1])
        with col1:
            cv_text = st.text_area(
                "Collez le texte de votre CV ou décrivez votre profil",
                height=200,
                placeholder="Ex. : Comptable avec 4 ans d'expérience, maîtrise de la comptabilité OHADA, "
                            "de l'audit et du logiciel Sage. Licence en gestion…",
            )
        with col2:
            ville = st.selectbox("Votre ville", CITIES)
            exp = st.number_input("Années d'expérience", 0, 40, 2)
            edu = st.selectbox("Niveau d'études", EDUCATION_LEVELS, index=3)
            top_n = st.slider("Nombre de résultats", 5, 30, 10)

        if st.button("🔍 Trouver mes offres", type="primary", disabled=not cv_text.strip()):
            detected = extract_skills(cv_text)
            if detected:
                st.info("Compétences détectées dans votre CV : " + ", ".join(detected))
            else:
                st.warning("Aucune compétence du référentiel détectée — l'appariement reposera "
                           "sur la similarité de texte.")
            results = engine.match_offers_for_seeker(
                cv_text=cv_text,
                seeker_meta={"ville": ville, "annees_experience": exp, "niveau_etudes": edu},
                top_n=top_n,
            )
            st.dataframe(results, use_container_width=True, hide_index=True)
    else:
        seekers = engine.seekers
        label = seekers["id_demandeur"] + " — " + seekers["nom"] + " (" + seekers["titre_professionnel"] + ", " + seekers["ville"] + ")"
        choice = st.selectbox("Profil de démonstration", label)
        idx = int(label[label == choice].index[0])
        with st.expander("Voir le CV"):
            st.write(seekers.iloc[idx]["cv_texte"])
        top_n = st.slider("Nombre de résultats ", 5, 30, 10)
        if st.button("🔍 Trouver les offres", type="primary"):
            results = engine.match_offers_for_seeker(seeker_idx=idx, top_n=top_n)
            st.dataframe(results, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- employer
with tab_employer:
    st.subheader("Recherche de candidats en langage naturel")
    query = st.text_input(
        "Décrivez le profil recherché",
        placeholder="Ex. : Je cherche un candidat en comptabilité avec une mobilité nationale",
    )
    if query.strip():
        st.dataframe(engine.search_candidates(query, top_n=10),
                     use_container_width=True, hide_index=True)
    st.divider()
    st.subheader("Trouvez les meilleurs candidats pour votre offre")
    offers = engine.offers
    label_o = offers["id_offre"] + " — " + offers["titre"] + " @ " + offers["entreprise"] + " (" + offers["ville"] + ")"
    choice_o = st.selectbox("Sélectionnez une offre", label_o)
    idx_o = int(label_o[label_o == choice_o].index[0])
    with st.expander("Voir l'offre complète"):
        st.write(offers.iloc[idx_o]["description"])
    top_n_c = st.slider("Nombre de candidats", 5, 30, 10)
    if st.button("👥 Classer les candidats", type="primary"):
        results = engine.match_seekers_for_offer(idx_o, top_n=top_n_c)
        st.dataframe(results, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------- stats
with tab_stats:
    st.subheader("Tableau de bord décisionnel — conseillers ACPE")
    recs_path = DATA_DIR / "recommendations_top10.csv"
    recs = pd.read_csv(recs_path) if recs_path.exists() else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Candidats", len(engine.seekers))
    c2.metric("Offres d'emploi", len(engine.offers))
    c3.metric("Localités couvertes", engine.offers["ville"].nunique())
    c4.metric("Taux moyen de compatibilité",
              f"{recs['score'].mean() * 100:.1f} %" if recs is not None else "—")

    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**Secteurs les plus représentés (offres)**")
        st.bar_chart(engine.offers["domaine"].value_counts().head(12))
        st.write("**Répartition géographique des offres**")
        st.bar_chart(engine.offers["ville"].value_counts().head(10))
    with col_b:
        st.write("**Métiers les plus demandés**")
        st.bar_chart(engine.offers["titre"].str.strip().str.title().value_counts().head(12))
        st.write("**Répartition géographique des candidats**")
        st.bar_chart(engine.seekers["ville"].value_counts().head(10))

    if recs is not None:
        st.divider()
        st.write("**Statistiques sur les recommandations générées** "
                 f"({recs['candidate_id'].nunique()} candidats × Top-{recs['rank'].max()})")
        c5, c6, c7 = st.columns(3)
        c5.metric("Recommandations générées", len(recs))
        c6.metric("Score médian", f"{recs['score'].median() * 100:.1f} %")
        c7.metric("Offres distinctes recommandées", recs["job_id"].nunique())
        st.write("**Distribution des scores de compatibilité**")
        hist = pd.cut(recs["score"] * 100, bins=range(0, 105, 10)).value_counts().sort_index()
        hist.index = [f"{i.left}-{i.right}%" for i in hist.index]
        st.bar_chart(hist)
    else:
        st.info("Lancez `python src/recommend.py --top-k 10` pour générer les "
                "recommandations et compléter ce tableau de bord.")


st.divider()
st.caption(
    "Code source : [github.com/glennmatondo64-prog/emploi-match]"
    "(https://github.com/glennmatondo64-prog/emploi-match) — "
    "Hackathon IndabaX Congo 2026 × ACPE."
)
