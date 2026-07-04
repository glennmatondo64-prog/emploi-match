"""Synthetic dataset generator — Congolese job market.

Generates realistic French-language job seeker profiles and job offers,
saved to data/demandeurs.csv and data/offres.csv.

Usage:
    python src/generate_data.py [--seekers 400] [--offers 200] [--seed 42]
"""
import argparse
import random
from pathlib import Path

import pandas as pd

from skills_taxonomy import (
    SKILLS_TAXONOMY, SOFT_SKILLS, CITIES, CITY_WEIGHTS,
    EDUCATION_LEVELS, CONTRACT_TYPES, LANGUAGES,
)

FIRST_NAMES = [
    "Chris", "Destin", "Gloire", "Merveille", "Prince", "Dieuveil", "Exaucé",
    "Grâce", "Bénédicte", "Ruth", "Esther", "Naomi", "Divine", "Sarah",
    "Jean", "Pierre", "Paul", "Antoine", "Serge", "Rodrigue", "Armel",
    "Clarisse", "Pamela", "Nadège", "Sylvie", "Chancelle", "Duval", "Fortuné",
]
LAST_NAMES = [
    "Makosso", "Nguesso", "Mabiala", "Moukala", "Nkouka", "Bemba", "Okemba",
    "Samba", "Loemba", "Tchicaya", "Malonga", "Bouity", "Ngoma", "Mavoungou",
    "Ondongo", "Itoua", "Elenga", "Akouala", "Mbochi", "Koumba", "Bakala",
]
COMPANY_PREFIXES = [
    "Société", "Groupe", "Compagnie", "Entreprise", "Cabinet", "Agence",
]
COMPANY_ROOTS = [
    "Congo Services", "du Fleuve", "Mayombe", "Alima", "Kouilou", "Sangha",
    "Batéké", "Niari", "Loango", "M'Foa", "Djoué", "Léfini", "Plateaux",
]
JOB_TITLES = {
    "Informatique / IT": ["Développeur web", "Technicien informatique", "Data Analyst", "Administrateur réseau", "Chef de projet IT"],
    "Pétrole / Gaz / Énergie": ["Technicien HSE", "Opérateur de production", "Mécanicien industriel", "Superviseur QHSE", "Soudeur qualifié"],
    "Banque / Finance": ["Comptable", "Auditeur junior", "Chargé de clientèle", "Analyste financier", "Agent de microfinance"],
    "BTP / Construction": ["Conducteur de travaux", "Ingénieur génie civil", "Topographe", "Dessinateur projeteur", "Chef de chantier"],
    "Logistique / Transport": ["Agent de transit", "Magasinier", "Responsable logistique", "Déclarant en douane", "Gestionnaire de flotte"],
    "Santé": ["Infirmier diplômé d'État", "Technicien de laboratoire", "Sage-femme", "Pharmacien assistant", "Agent de santé communautaire"],
    "Éducation / Formation": ["Enseignant", "Formateur professionnel", "Encadreur pédagogique", "Répétiteur"],
    "Commerce / Vente / Marketing": ["Commercial terrain", "Community manager", "Caissier", "Responsable des ventes", "Chargé de marketing"],
    "Agriculture / Agroalimentaire": ["Technicien agricole", "Agronome", "Chef d'exploitation", "Agent de transformation"],
    "Administration / RH / Juridique": ["Assistant RH", "Secrétaire de direction", "Gestionnaire de paie", "Assistant juridique", "Chargé des marchés publics"],
    "Télécommunications": ["Technicien fibre optique", "Ingénieur radio", "Technicien de maintenance télécom", "Superviseur réseau"],
    "Hôtellerie / Restauration": ["Cuisinier", "Réceptionniste", "Serveur", "Gouvernante", "Chef pâtissier"],
}


def _pick_city(rng):
    return rng.choices(CITIES, weights=CITY_WEIGHTS, k=1)[0]


def _company_name(rng):
    return f"{rng.choice(COMPANY_PREFIXES)} {rng.choice(COMPANY_ROOTS)}"


def generate_seekers(n, rng):
    rows = []
    domains = list(SKILLS_TAXONOMY)
    for i in range(1, n + 1):
        domain = rng.choice(domains)
        # ~20% of seekers have a secondary domain (career changers)
        skill_pool = list(SKILLS_TAXONOMY[domain])
        if rng.random() < 0.2:
            second = rng.choice([d for d in domains if d != domain])
            skill_pool += rng.sample(SKILLS_TAXONOMY[second], k=min(2, len(SKILLS_TAXONOMY[second])))
        skills = rng.sample(skill_pool, k=min(rng.randint(3, 7), len(skill_pool)))
        soft = rng.sample(SOFT_SKILLS, k=rng.randint(2, 4))
        exp = max(0, int(rng.gauss(5, 4)))
        edu = rng.choices(EDUCATION_LEVELS, weights=[5, 10, 25, 22, 20, 15, 3], k=1)[0]
        langs = ["français"] + rng.sample(LANGUAGES[1:], k=rng.randint(1, 3))
        city = _pick_city(rng)
        name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        title = rng.choice(JOB_TITLES[domain])
        cv_text = (
            f"{title} avec {exp} ans d'expérience dans le domaine {domain.lower()}. "
            f"Niveau d'études : {edu}. Compétences principales : {', '.join(skills)}. "
            f"Qualités : {', '.join(soft)}. Langues parlées : {', '.join(langs)}. "
            f"Basé(e) à {city}, mobilité possible sur le territoire national."
        )
        rows.append({
            "id_demandeur": f"D{i:04d}",
            "nom": name,
            "age": rng.randint(20, 55),
            "ville": city,
            "domaine": domain,
            "titre_professionnel": title,
            "niveau_etudes": edu,
            "annees_experience": exp,
            "competences": "; ".join(skills),
            "soft_skills": "; ".join(soft),
            "langues": "; ".join(langs),
            "cv_texte": cv_text,
        })
    return pd.DataFrame(rows)


def generate_offers(n, rng):
    rows = []
    domains = list(SKILLS_TAXONOMY)
    for i in range(1, n + 1):
        domain = rng.choice(domains)
        title = rng.choice(JOB_TITLES[domain])
        req_skills = rng.sample(SKILLS_TAXONOMY[domain], k=min(rng.randint(3, 6), len(SKILLS_TAXONOMY[domain])))
        soft = rng.sample(SOFT_SKILLS, k=2)
        exp_min = rng.choices([0, 1, 2, 3, 5, 8], weights=[15, 20, 25, 20, 15, 5], k=1)[0]
        edu_min = rng.choices(EDUCATION_LEVELS[:6], weights=[5, 10, 25, 25, 25, 10], k=1)[0]
        city = _pick_city(rng)
        contract = rng.choices(CONTRACT_TYPES, weights=[35, 35, 12, 10, 8], k=1)[0]
        salary = rng.choice([150, 200, 250, 300, 400, 500, 700, 900, 1200]) * 1000
        company = _company_name(rng)
        description = (
            f"{company} recrute un(e) {title} à {city} en {contract}. "
            f"Missions : intervenir sur des activités de {domain.lower()} et contribuer aux objectifs de l'équipe. "
            f"Profil recherché : {edu_min} minimum, au moins {exp_min} an(s) d'expérience. "
            f"Compétences requises : {', '.join(req_skills)}. "
            f"Qualités attendues : {', '.join(soft)}."
        )
        rows.append({
            "id_offre": f"O{i:04d}",
            "titre": title,
            "entreprise": company,
            "ville": city,
            "domaine": domain,
            "type_contrat": contract,
            "niveau_etudes_min": edu_min,
            "experience_min": exp_min,
            "salaire_fcfa": salary,
            "competences_requises": "; ".join(req_skills),
            "soft_skills": "; ".join(soft),
            "description": description,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seekers", type=int, default=400)
    parser.add_argument("--offers", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    seekers = generate_seekers(args.seekers, rng)
    offers = generate_offers(args.offers, rng)
    seekers.to_csv(data_dir / "demandeurs.csv", index=False)
    offers.to_csv(data_dir / "offres.csv", index=False)
    print(f"{len(seekers)} demandeurs -> {data_dir / 'demandeurs.csv'}")
    print(f"{len(offers)} offres -> {data_dir / 'offres.csv'}")


if __name__ == "__main__":
    main()
