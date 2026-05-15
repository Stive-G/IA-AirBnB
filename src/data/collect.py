"""
Téléchargement des données AirBnB depuis insideairbnb.com — Juin 2025

Ce module s'occupe d'une seule responsabilité : aller chercher les fichiers
.csv.gz sur internet et les décompresser dans data/raw/<ville>/.
"""
import gzip
import shutil
import urllib.request
from pathlib import Path

# URLs des fichiers détaillés (79 colonnes) pour chaque ville — juin 2025.
# On utilise les "detailed listings" et non les "visualisation listings" (18 colonnes)
# car ils contiennent les scores de reviews, les infos hôte, etc.
URLS = {
    "lyon": {
        "listings_detail": (
            "https://data.insideairbnb.com/france/auvergne-rhone-alpes/"
            "lyon/2025-06-15/data/listings.csv.gz"
        ),
    },
    "paris": {
        "listings_detail": (
            "https://data.insideairbnb.com/france/ile-de-france/"
            "paris/2025-06-06/data/listings.csv.gz"
        ),
    },
    "bordeaux": {
        "listings_detail": (
            "https://data.insideairbnb.com/france/nouvelle-aquitaine/"
            "bordeaux/2025-06-15/data/listings.csv.gz"
        ),
    },
}

# Date de snapshot pour chaque ville (utile pour traçabilité)
DATES = {"lyon": "2025-06-15", "paris": "2025-06-06", "bordeaux": "2025-06-15"}

# En-tête HTTP pour simuler un navigateur — certains serveurs bloquent les requêtes
# sans User-Agent (erreur 403 Forbidden)
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def _download(url: str, dest: Path) -> None:
    """
    Télécharge un fichier depuis `url` et le sauvegarde dans `dest`.

    On utilise urllib.request (bibliothèque standard Python, pas besoin de requests).
    timeout=120 : on attend max 2 minutes avant d'abandonner (fichiers > 100 Mo).
    Le fichier est écrit en mode binaire 'wb' car c'est un .gz compressé.
    """
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        f.write(r.read())


def download_all(raw_dir: str = "data/raw") -> None:
    """
    Télécharge et décompresse les fichiers CSV pour les 3 villes.

    Étapes pour chaque ville :
    1. Crée le dossier data/raw/<ville>/ si nécessaire
    2. Vérifie si le CSV existe déjà (évite de re-télécharger)
    3. Télécharge le .gz (~20-170 Mo selon la ville)
    4. Décompresse avec gzip + shutil.copyfileobj (streaming, économe en RAM)
    5. Supprime le .gz devenu inutile

    Pourquoi gzip.open + shutil.copyfileobj ?
    → C'est la méthode recommandée pour décompresser sans charger tout en mémoire.
    """
    for city, files in URLS.items():
        city_dir = Path(raw_dir) / city
        city_dir.mkdir(parents=True, exist_ok=True)

        for name, url in files.items():
            csv_dest = city_dir / f"{name}.csv"

            # Si le CSV est déjà là, on saute (idempotent)
            if csv_dest.exists():
                print(f"[SKIP] {city}/{name}.csv déjà présent")
                continue

            gz_dest = city_dir / f"{name}.csv.gz"
            print(f"[DL]   {city}/{name}.csv.gz ...")
            _download(url, gz_dest)

            print(f"       décompression ...")
            with gzip.open(gz_dest, "rb") as gz, open(csv_dest, "wb") as out:
                shutil.copyfileobj(gz, out)  # copie par blocs, pas tout en RAM

            gz_dest.unlink()  # supprime le .gz intermédiaire
            print(f"       -> {csv_dest.stat().st_size:,} octets")

    print("Téléchargement terminé.")


if __name__ == "__main__":
    download_all()
