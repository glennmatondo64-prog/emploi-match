"""Référentiel de compétences pour le marché du travail congolais.

Référence centrale utilisée par le générateur de données et le moteur d'appariement.
"""

SKILLS_TAXONOMY = {
    "Informatique / IT": [
        "python", "java", "javascript", "php", "sql", "développement web",
        "développement mobile", "administration réseau", "cybersécurité",
        "maintenance informatique", "bases de données", "linux",
        "analyse de données", "machine learning", "excel avancé",
    ],
    "Pétrole / Gaz / Énergie": [
        "forage", "production pétrolière", "hse", "maintenance industrielle",
        "instrumentation", "électromécanique", "soudure", "tuyauterie",
        "inspection technique", "gestion de plateforme", "qhse",
    ],
    "Banque / Finance": [
        "comptabilité", "audit", "analyse financière", "gestion de trésorerie",
        "fiscalité", "contrôle de gestion", "microfinance", "conformité bancaire",
        "gestion des risques", "sage comptabilité", "ohada",
    ],
    "BTP / Construction": [
        "génie civil", "topographie", "conduite de travaux", "dessin technique",
        "autocad", "métré", "béton armé", "voirie", "électricité bâtiment",
        "plomberie", "menuiserie", "maçonnerie",
    ],
    "Logistique / Transport": [
        "gestion de stock", "transit", "douane", "manutention portuaire",
        "planification logistique", "conduite poids lourds", "gestion de flotte",
        "supply chain", "magasinage",
    ],
    "Santé": [
        "soins infirmiers", "pharmacie", "laboratoire médical", "sage-femme",
        "santé publique", "kinésithérapie", "imagerie médicale", "nutrition",
    ],
    "Éducation / Formation": [
        "enseignement", "pédagogie", "formation professionnelle",
        "conception de programmes", "encadrement scolaire", "alphabétisation",
    ],
    "Commerce / Vente / Marketing": [
        "vente", "négociation commerciale", "marketing digital",
        "gestion de clientèle", "community management", "étude de marché",
        "merchandising", "caisse", "service client",
    ],
    "Agriculture / Agroalimentaire": [
        "agronomie", "élevage", "transformation alimentaire", "maraîchage",
        "gestion coopérative", "pêche", "sylviculture",
    ],
    "Administration / RH / Juridique": [
        "gestion des ressources humaines", "paie", "droit du travail",
        "secrétariat", "gestion administrative", "rédaction juridique",
        "passation de marchés", "archivage",
    ],
    "Télécommunications": [
        "fibre optique", "réseaux mobiles", "transmission", "bss/oss",
        "maintenance télécom", "voip", "radiofréquence",
    ],
    "Hôtellerie / Restauration": [
        "cuisine", "service en salle", "réception hôtelière", "housekeeping",
        "gestion hôtelière", "pâtisserie",
    ],
}

SOFT_SKILLS = [
    "travail en équipe", "communication", "leadership", "rigueur",
    "autonomie", "adaptabilité", "gestion du temps", "esprit d'analyse",
    "sens de l'organisation", "prise d'initiative",
]

CITIES = [
    "Brazzaville", "Pointe-Noire", "Dolisie", "Nkayi", "Ouesso",
    "Owando", "Madingou", "Impfondo", "Sibiti", "Kinkala",
]

CITY_WEIGHTS = [0.38, 0.32, 0.08, 0.05, 0.04, 0.04, 0.03, 0.02, 0.02, 0.02]

EDUCATION_LEVELS = [
    "Sans diplôme", "BEPC", "BAC", "BAC+2 (BTS/DUT)",
    "Licence (BAC+3)", "Master (BAC+5)", "Doctorat",
]

CONTRACT_TYPES = ["CDI", "CDD", "Stage", "Consultance", "Intérim"]

LANGUAGES = ["français", "lingala", "kituba", "anglais"]

ALL_SKILLS = sorted({s for skills in SKILLS_TAXONOMY.values() for s in skills})
