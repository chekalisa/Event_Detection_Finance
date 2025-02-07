from fastapi import APIRouter, Query, HTTPException, UploadFile, File, status
from fastapi.responses import FileResponse

from app.services import (
    process_news,  
    process_clustering, 
    process_outliers, 
    process_vectorization, 
    process_silhouette,
    clusters_without_outliers,
    preprocess_tweets,
    process_tweets_vectorization,
    tweets_assignment,
    alert_generation
)
from datetime import datetime
import csv
import requests
import time
import logging
import os

# Configuration du logging pour webscrapping
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


UPLOAD_DIR = "uploaded_data"
SCRAPED_TWEETS_DIR = "scraped_tweets"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SCRAPED_TWEETS_DIR, exist_ok=True)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", summary="Health Check", description="Checks si le API répond et marche.")
def health_check():
    return {"status": "OK", "message": "API est en train de marcher"}

@router.post("/scrape/stocktwits", status_code=status.HTTP_200_OK, summary="Scrape StockTwits", description="Scrape StockTwits data pour BTC.X et sauvegarde à CSV.")
async def scrape_stocktwits_endpoint(target_date: str, filename: str = "btc_tweets.csv"):
    """
    Web scraping endpoint avec date format DD/MM/YYYY.
    """
    try:
        # Convertir la date en objet datetime
        target_date_obj = datetime.strptime(target_date, "%d/%m/%Y")
        logger.info(f"Scraping pour la date {target_date_obj} avec filename {filename}")

        # Exécuter la fonction de scraping
        result = scrape_stocktwits(target_date_obj, filename_prefix=filename.split(".")[0])
        return result

    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Format de date invalide. utilisez DD/MM/YYYY."
        )
    except Exception as e:
        logger.error(f"Scraping echoué: {e}")
        raise HTTPException(status_code=500, detail=f"Scraping echoué: {str(e)}")


def scrape_stocktwits(target_date: datetime, filename_prefix: str):
    """Webscrapping des tweets."""

    def fetch_tweets(max_id=None):
        url = "https://api.stocktwits.com/api/2/streams/symbol/BTC.X.json"
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
        params = {"limit": 50}
        if max_id:
            params["max"] = max_id

        retries = 5
        while retries > 0:
            try:
                response = requests.get(url, headers=headers, params=params, timeout=10)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    logger.warning("Limite de taux atteinte (429). Pause de 60 secondes...")
                    time.sleep(60)
                    retries -= 1
                else:
                    logger.error(f"Erreur. Status code: {response.status_code}")
                    retries -= 1
            except requests.RequestException as e:
                logger.error(f"Request echoué: {e}")
                retries -= 1

        return None

    def save_tweets_to_csv(tweets, filename):
        logger.info(f"Sauvegarde {len(tweets)} des tweets à {filename}")
        with open(filename, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Date", "Message", "Likes", "Followers", "Sentiment"])
            for tweet in tweets:
                writer.writerow(tweet)
        logger.info(f"File sauvegardé: {filename}")

    # 📌 Chemin du fichier pour stocker les tweets
    filename = os.path.join(SCRAPED_TWEETS_DIR, f"{filename_prefix}_{target_date.strftime('%Y-%m-%d')}.csv")

    all_tweets = []
    max_id = None
    total_calls = 0
    earliest_time = None

    while True:
        data = fetch_tweets(max_id)
        total_calls += 1
        if not data or "messages" not in data:
            break

        logger.info(f"Processing {len(data.get('messages', []))} messages.")
        for item in data.get("messages", []):
            try:
                created_at = item.get("created_at", None)
                if created_at:
                    created_at = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
                else:
                    created_at = None

                if created_at and created_at.date() == target_date.date():
                    likes = item.get("likes", {}).get("total", None)
                    user = item.get("user", {})
                    followers = user.get("followers", None) if isinstance(user, dict) else None
                    entities = item.get("entities", {})
                    sentiment = (
                        entities.get("sentiment", {}).get("basic", None)
                        if isinstance(entities, dict) and entities.get("sentiment")
                        else None
                    )
                    tweet = [
                        created_at.strftime("%Y-%m-%d %H:%M:%S") if created_at else None,
                        item.get("body", None),
                        likes,
                        followers,
                        sentiment
                    ]
                    logger.info(f"Ajout de tweet créé à {created_at}")
                    all_tweets.append(tweet)
                    if earliest_time is None or created_at < earliest_time:
                        earliest_time = created_at

                elif created_at and created_at.date() < target_date.date():
                    logger.info("Tous les tweets ont été colléctés.")
                    save_tweets_to_csv(all_tweets, filename=filename)
                    return {
                        "message": "Scraping completé",
                        "total_calls": total_calls,
                        "earliest_tweet_time": earliest_time.strftime("%Y-%m-%d %H:%M:%S") if earliest_time else None,
                        "filename": filename
                    }
            except Exception as e:
                logger.error(f"Erreur: {repr(e)}")
                continue

        if "cursor" in data and "max" in data["cursor"]:
            max_id = data["cursor"]["max"]
        else:
            break

    save_tweets_to_csv(all_tweets, filename=filename)
    return {
        "message": "Scraping completé",
        "total_calls": total_calls,
        "earliest_tweet_time": earliest_time.strftime("%Y-%m-%d %H:%M:%S") if earliest_time else None,
        "filename": filename
    }

@router.post("/upload/news", summary="Chargement des News Articles", description="Charge un Excel file qui contient news articles.")
def upload_news(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, "news.xlsx")
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())
    return {"message": f"{file.filename} Chargé comme news.xlsx"}

@router.post("/upload/bitcoin", summary="Chargement de Bitcoin Data", description="Charge un CSV file qui contient Bitcoin prix data.")
def upload_bitcoin(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, "Bitcoin.csv")
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())
    return {"message": f"{file.filename} chargé comme Bitcoin.csv"}

@router.post("/upload/tweets", summary="Upload Tweets Data", description="Uploads a CSV file containing Tweets.")
def upload_tweets(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, "tweets.csv")
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())
    return {"message": f"{file.filename} chargé comme tweets.csv"}

@router.post("/process/news_bitcoin", summary="Traitement des données News et Bitcoin", description="Traite les données de news et de Bitcoin téléchargées afin d'extraire des caractéristiques pertinentes.")
def process_all():
    return process_news()

@router.post("/process/vectorization", summary="Vectorisation des articles de presse", description="Vectorise les articles de presse en utilisant Word2Vec et génère des embeddings de documents.")
def vectorization():
    return process_vectorization()

@router.post("/process/silhouette", summary="Calcul des scores de Silhouette", description="Calcule les scores de silhouette pour une gamme de nombres de clusters afin de déterminer le clustering optimal.")
def silhouette_analysis():
    return process_silhouette()

@router.post("/process/clustering", summary="Réalisation du clustering", description="Regroupe les articles de presse en clusters via un clustering hiérarchique avec un nombre de clusters spécifié.")
def clustering(k: int = Query(3, description="Nombre de clusters à créer.")):
    return process_clustering(k)

@router.post("/process/outliers", summary="Suppression des valeurs aberrantes", description="Supprime les valeurs aberrantes des données groupées en fonction d'un seuil de similarité cosinus.")
def process_outlier_removal(threshold: float = Query(0.9, description="Seuil de similarité cosinus pour supprimer les valeurs aberrantes.")):
    return process_outliers(threshold)

@router.post("/process/clusters_without_outliers", summary="Clusters sans valeurs aberrantes", description="Génère des graphiques des clusters sans valeurs aberrantes et calcule les scores TF-IDF pour chaque cluster.")
def clusters_no_outliers():
    return clusters_without_outliers()

@router.post("/process/tweets", summary="Prétraitement des Tweets", description="Nettoie et tokenize le dataset de Tweets téléchargé.")
def preprocess_tweets_router():
    return preprocess_tweets()

@router.post("/process/tweets_vectorization", summary="Vectorisation des Tweets", description="Génère des embeddings vectoriels pour les tweets en utilisant un modèle Word2Vec.")
def tweets_vectorization():
    return process_tweets_vectorization()

@router.post("/process/tweets_assignment", summary="Assignation des Tweets aux clusters", description="Assigne les tweets aux clusters de news en fonction de la similarité cosinus.")
def assignment():
    return tweets_assignment()

@router.post("/process/alert_generation", summary="Génération d'alertes", description="Génère des alertes en fonction de la proportion de tweets assignés par jour.")
def generation():
    return alert_generation()

@router.get("/images/cluster", summary="Affiche le graphique des mots les plus important de cluster qui détècte notre événement (Trump election)")
def get_cluster_image():
    image_path = os.path.join("graphs", "cluster_1_top_terms.png")
    if os.path.exists(image_path):
        return FileResponse(image_path, media_type="image/png")
    else:
        return {"error": "Image non trouvée"}

@router.get("/images/alert", summary="Affiche le graphique de Alert Generation")
def get_cluster_image():
    image_path = os.path.join("graphs", "alert_generation.png")
    if os.path.exists(image_path):
        return FileResponse(image_path, media_type="image/png")
    else:
        return {"error": "Image non trouvée"}