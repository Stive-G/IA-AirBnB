import gzip
import shutil
import urllib.request
from pathlib import Path


URLS = {
    "paris": "https://data.insideairbnb.com/france/ile-de-france/paris/2025-06-06/data/listings.csv.gz",
    "lyon": "https://data.insideairbnb.com/france/auvergne-rhone-alpes/lyon/2025-06-15/data/listings.csv.gz",
    "bordeaux": "https://data.insideairbnb.com/france/nouvelle-aquitaine/bordeaux/2025-06-15/data/listings.csv.gz",
}


def download_file(url, destination):
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response:
        with open(destination, "wb") as file:
            file.write(response.read())


def unzip_file(gz_path, csv_path):
    with gzip.open(gz_path, "rb") as compressed:
        with open(csv_path, "wb") as output:
            shutil.copyfileobj(compressed, output)


def download_all(raw_dir="data/raw"):
    for city, url in URLS.items():
        # Un dossier par ville pour éviter de mélanger les fichiers
        city_dir = Path(raw_dir) / city
        city_dir.mkdir(parents=True, exist_ok=True)

        gz_path = city_dir / "listings.csv.gz"
        csv_path = city_dir / "listings_detail.csv"

        if csv_path.exists():
            print(city, ": fichier déjà présent")
            continue

        print(city, ": téléchargement")
        download_file(url, gz_path)

        # Le fichier récupéré est compressé on le transforme en csv
        print(city, ": décompression")
        unzip_file(gz_path, csv_path)

    print("Téléchargement terminé")


if __name__ == "__main__":
    download_all()
