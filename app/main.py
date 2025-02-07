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
                "/health": "Check system status",
                "/upload/news": "Upload news articles (Excel format)",
                "/upload/bitcoin": "Upload Bitcoin price data (CSV format)",
                "/upload/tweets": "Upload the Tweets dataset (CSV format)",
                "/scrape/stocktwits": "Scrape StockTwits data for BTC.X",
                "/process/news": "Process news articles and integrate Bitcoin data (Tokenization, creation of matrix, sorting of tokens)",
                "/process/vectorization": "Vectorize news articles using Word2Vec",
                "/process/silhouette": "Compute silhouette scores for clustering evaluation",
                "/process/clustering": "Perform hierarchical clustering on news articles",
                "/process/outliers": "Remove outliers based on cosine similarity",
                "/process/clusters_without_outliers": "Analyze clusters without outliers using TF-IDF",
                "/process/tweets": "Preprocess uploaded Tweets (cleaning, tokenization, matrix construction)",
                "/process/tweets_vectorization": "Convert Tweets into vector representations",
                "/process/tweets_assignment": "Assign Tweets to clusters using cosine similarity",
                "/process/alert_generation": "Generate alerts based on Tweet assignments",
                "/images/cluster": "Display the graph with the most important words for cluster that detects out event (Trump election)",
                "/images/alert": "Display the alert generation graph"
        }
    }