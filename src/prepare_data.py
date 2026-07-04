"""Préparation du dataset officiel ACPE (Hackathon IndabaX Congo 2026).

Entrées (data/raw/) :
  - Offres_ACPE.xlsx                  2 535 offres (métadonnées)
  - Offres_ACPE_Extensions.xlsx       143 offres enrichies (description, profil, compétences)
  - Appariement_Demandeurs_Offres.xlsx 41 285 demandeurs -> 3 offres (vérité terrain / format de soumission)

Sorties (data/) :
  - offres_acpe.csv     table unique et propre des offres, schéma du moteur
  - appariement.csv     copie CSV de la vérité terrain

Usage : python src/prepare_data.py
"""
import re
from pathlib import Path

import pandas as pd

from skills_taxonomy import EDUCATION_LEVELS
from matching_engine import extract_skills, normalize

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT = Path(__file__).resolve().parent.parent / "data"

EDU_MAP = {
    5: "Master (BAC+5)", 4: "Master (BAC+5)", 3: "Licence (BAC+3)",
    2: "BAC+2 (BTS/DUT)", 1: "BAC+2 (BTS/DUT)",
}


def parse_experience_min(profil: str):
    """'Expérience de 2 à 5 ans' -> 2 ; retourne None si absent."""
    if not isinstance(profil, str):
        return None
    m = re.search(r"(\d+)\s*(?:à\s*\d+\s*)?ans?", profil.lower())
    return int(m.group(1)) if m else None


def parse_education_min(profil: str):
    """'Bac+2/3 en maintenance' -> 'BAC+2 (BTS/DUT)' ; None si absent."""
    if not isinstance(profil, str):
        return None
    p = profil.lower()
    m = re.search(r"bac\s*\+\s*(\d)", p)
    if m:
        return EDU_MAP.get(int(m.group(1)), "Licence (BAC+3)")
    if re.search(r"\bbac\b", p):
        return "BAC"
    return None


def clean_city(lieu: str) -> str:
    return str(lieu).strip().title()


def main():
    offres = pd.read_excel(RAW / "Offres_ACPE.xlsx")
    ext = pd.read_excel(RAW / "Offres_ACPE_Extensions.xlsx")
    appar = pd.read_excel(RAW / "Appariement_Demandeurs_Offres.xlsx")

    offres = offres.drop_duplicates(subset="Référence offre", keep="first")
    ext = ext.rename(columns={"Référence": "Référence offre"})
    df = offres.merge(
        ext[["Référence offre", "Description", "Profil", "Compétences", "Type Contrat"]],
        on="Référence offre", how="left",
    )

    # Texte riche de l'offre (pour TF-IDF / embeddings)
    def build_text(r):
        parts = [str(r["Intitule"]), f"Secteur : {r['Secteur activité']}."]
        for col in ("Description", "Profil", "Compétences"):
            if isinstance(r[col], str) and r[col].strip():
                parts.append(r[col].strip())
        return " ".join(parts)

    df["description"] = df.apply(build_text, axis=1)

    # Compétences requises : champ Compétences des extensions si présent,
    # sinon extraction automatique par référentiel sur le texte de l'offre.
    def build_skills(r):
        if isinstance(r["Compétences"], str) and r["Compétences"].strip():
            items = re.split(r"[;,]", r["Compétences"])
            return "; ".join(normalize(i) for i in items if i.strip())
        return "; ".join(extract_skills(r["description"]))

    out = pd.DataFrame({
        "id_offre": df["Référence offre"],
        "titre": df["Intitule"].astype(str).str.strip(),
        "entreprise": df["Entreprise"].astype(str).str.strip(),
        "ville": df["Lieu"].map(clean_city),
        "domaine": df["Secteur activité"].astype(str).str.strip(),
        "type_contrat": df["Type contrat"].fillna(df["Type Contrat"]).fillna("Non précisé"),
        "niveau_etudes_min": df["Profil"].map(parse_education_min),
        "experience_min": df["Profil"].map(parse_experience_min),
        "competences_requises": df.apply(build_skills, axis=1),
        "description": df["description"],
    })
    out.to_csv(OUT / "offres_acpe.csv", index=False)
    appar.to_csv(OUT / "appariement.csv", index=False)

    if (RAW / "Demandeurs.xlsx").exists():
        convert_demandeurs()

    n_skills = (out["competences_requises"].str.len() > 0).sum()
    print(f"{len(out)} offres -> {OUT / 'offres_acpe.csv'} "
          f"({n_skills} avec compétences, {out['experience_min'].notna().sum()} avec expérience min)")
    print(f"{len(appar)} appariements -> {OUT / 'appariement.csv'}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Conversion du fichier officiel des demandeurs (Demandeurs.xlsx)
# ---------------------------------------------------------------------------

DEPT_FROM_MATRICULE = {
    "BZV": "Brazzaville", "PNR": "Pointe-Noire", "KOU": "Kouilou",
    "BOU": "Bouenza", "NIA": "Niari", "LEK": "Lékoumou", "PLA": "Plateaux",
    "CUV": "Cuvette", "CVO": "Cuvette-Ouest", "SAN": "Sangha",
    "LIK": "Likouala", "POO": "Pool",
}

NIVEAU_MAP = {
    "aucun": "Sans diplôme", "primaire": "Sans diplôme",
    "secondaire 1": "BEPC", "bac": "BAC", "autre": "BAC",
    "post-secondaire – professionnel": "BAC+2 (BTS/DUT)",
    "bac +3": "Licence (BAC+3)", "bac +4/+5 et plus": "Master (BAC+5)",
}

_SKIP_VALUES = {"non déclaré", "non declare", "divers", "nan", "", "non", "oui"}


def _keep(v):
    return isinstance(v, str) and v.strip().lower() not in _SKIP_VALUES


def convert_demandeurs():
    """Demandeurs.xlsx officiel -> data/demandeurs_acpe.csv (schéma du moteur)."""
    src = RAW / "Demandeurs.xlsx"
    d = pd.read_excel(src)
    d.columns = [c.strip() for c in d.columns]
    d = d.drop_duplicates(subset="Matricule", keep="first")

    def city(matricule):
        return DEPT_FROM_MATRICULE.get(str(matricule)[2:5], "")

    def niveau(row):
        return NIVEAU_MAP.get(str(row["niveau_etude"]).strip().lower(), "BAC")

    def cv_text(row):
        parts = []
        if _keep(row.get("Métier visé / Qualification visée")):
            parts.append(f"Métier visé : {row['Métier visé / Qualification visée']}.")
        if _keep(row.get("Qualification")):
            parts.append(f"Qualification : {row['Qualification']}.")
        if _keep(row.get("qualification_metier")):
            parts.append(str(row["qualification_metier"]) + ".")
        if _keep(row.get("Filière / Spécialité")):
            parts.append(f"Filière : {row['Filière / Spécialité']}.")
        if _keep(row.get("Secteur demandé")):
            parts.append(f"Secteur recherché : {row['Secteur demandé']}.")
        if _keep(row.get("secteur_metier")):
            parts.append(f"Secteur métier : {row['secteur_metier']}.")
        sect_act = row.get("Secteur d'activité")
        if _keep(sect_act):
            parts.append(f"Secteur d'activité : {sect_act}.")
        if _keep(row.get("Diplome")):
            parts.append(f"Diplôme : {row['Diplome']}.")
        return " ".join(parts) if parts else "Profil non renseigné."

    texts = d.apply(cv_text, axis=1)
    out = pd.DataFrame({
        "id_demandeur": d["Matricule"],
        "nom": d["Matricule"],                       # anonymisé
        "age": pd.to_numeric(d["Age"], errors="coerce").fillna(0).astype(int),
        "ville": d["Matricule"].map(city),
        "domaine": d["secteur_metier"].fillna(d["Secteur d'activité"]).fillna(""),
        "titre_professionnel": d["Métier visé / Qualification visée"].fillna(
            d["Qualification"]).fillna("Non renseigné"),
        "niveau_etudes": d.apply(niveau, axis=1),
        "annees_experience": 0,                       # non fourni par l'ACPE
        "competences": [";".join(extract_skills(t)) for t in texts],
        "soft_skills": "",
        "langues": "français",
        "cv_texte": texts,
    })
    out.to_csv(OUT / "demandeurs_acpe.csv", index=False)
    n_sk = (out["competences"].str.len() > 0).sum()
    print(f"{len(out)} demandeurs réels -> {OUT / 'demandeurs_acpe.csv'} ({n_sk} avec compétences détectées)")
