from fastapi import FastAPI
from app.routes import router

app = FastAPI(
    title="'Event Detection' API pour Finance",
    description="API pour traitement et data analysis",
    version="1.0.0"
)

app.include_router(router)

@app.get("/", summary="API Overview", description="Message avec la description des endpoints")
def index():
    return {
        "message": "'Event Detection' API pour Finance",
        "documentation": {
            "Swagger Docs": "/docs",
            "ReDoc Docs": "/redoc"
        },
       
        "endpoints": {
            "/health": "Vérifier l'état du système",
            "/upload/news": "Charger des articles de presse (format Excel)",
            "/upload/bitcoin": "Charger les données de prix du Bitcoin (format CSV)",
            "/upload/tweets": "Charger le dataset de Tweets (format CSV)",
            "/scrape/stocktwits": "Scraper les données de StockTwits pour BTC.X",
            "/process/news": "Traiter les articles de presse et intégrer les données Bitcoin",
            "/process/vectorization": "Vectoriser les articles de presse en utilisant Word2Vec",
            "/process/silhouette": "Calculer les scores de silhouette pour le clustering",
            "/process/clustering": "Effectuer un clustering hiérarchique sur les articles de presse",
            "/process/outliers": "Supprimer les valeurs aberrantes en fonction de la similarité cosinus",
            "/process/clusters_without_outliers": "Analyser les clusters sans valeurs aberrantes à l'aide de TF-IDF",
            "/process/tweets": "Prétraiter les Tweets chargés (nettoyage, tokenisation)",
            "/process/tweets_vectorization": "Convertir les tweets en représentations vectorielles",
            "/process/tweets_assignment": "Assigner les tweets aux clusters en utilisant la similarité cosinus",
            "/process/alert_generation": "Générer des alertes en fonction des assignations de tweets"
        }
    }