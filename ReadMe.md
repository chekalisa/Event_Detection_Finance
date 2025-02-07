# Finance Event Detection

## Introduction

Bienvenue dans notre projet de détection d'événements financiers, qui explore l'impact des actualités et des discussions sociales sur les variations de prix du Bitcoin.

Ce projet repose sur l'analyse combinée de news financières et de messages sociaux (StockTwits, Twitter) pour identifier les événements majeurs et leur influence sur les marchés financiers.


## Comment Exécuter

Pour exécuter le FastAPI, deux possibilités : 

1. Vous pouvez lancer en local le FastAPI en lançant les commandes suivantes : 
   
   1.  Créer et lancer un environement virtuel: 
       ```bash
       python -m venv .venv
       source .venv/bin/activate
       ```

   2. Charger poetry : 
       ```bash
       pip install poetry
       ```

   3. Charger les librairies :

    ```bash
    poetry install
    ```

   4. lancer fastAPI :

    ```bash
    uvicorn app.main:app --reload
    ```
    

2. Sinon vous pouvez accéder au FfastApi via le docker file de ce repository en suivant ces étapes : 
    
    1. Créer l'image Docker
       ```bash
       docker build -t fastapi-app .
       ```
    En créant l'image docker, un environnement virtuel va se créer et les dépendances vont se télécharger directement. 
    
    2. Lancer l'image Docker :
        ```bash
        docker run -p 8000:8000 fastapi-app
        ```


Cette commande va vous adresser vers le FastAPI avec les endpoins suivants  : 
- health: Vérifier l'état du système,
- /scrape/stocktwits: Scraper les données de StockTwits pour BTC.X,
- /upload/news: Charger des articles de presse (format Excel),
- /upload/bitcoin: Charger les données de prix du Bitcoin (format CSV),
- /upload/tweets: Charger le dataset de Tweets (format CSV),
- /process/news: Traiter les articles de presse et intégrer les données Bitcoin,
- /process/vectorization": Vectoriser les articles de presse en utilisant Word2Vec,
- /process/silhouette: Calculer les scores de silhouette pour le clustering,
- /process/clustering: Effectuer un clustering hiérarchique sur les articles de presse,
- /process/outliers: Supprimer les valeurs aberrantes en fonction de la similarité cosinus,
- /process/clusters_without_outliers: Analyser les clusters sans valeurs aberrantes à l'aide de TF-IDF,
- /process/tweets: Prétraiter les Tweets chargés (nettoyage, tokenisation),
- /process/tweets_vectorization: Convertir les tweets en représentations vectorielles,
- /process/tweets_assignment: Assigner les tweets aux clusters en utilisant la similarité cosinus,
- /process/alert_generation: Générer des alertes en fonction des assignations de tweets

    3.  Accéder à l'application FastAPI :
        http://localhost:8000
    
    4. Utiliser l'interface interactive Swagger UI pour tester l'API :
        http://localhost:8000/docs

## Information sur le code
L’endpoint de web scraping peut prendre un temps considérable à s’exécuter. Pour simplifier son utilisation, un dossier contenant tous les tweets déjà webscrapés est mis à disposition.

Pour charger les données, utilisez les endpoints "/upload/news", "/upload/bitcoin" et "/upload/tweets", disponibles dans le dossier /data. Ensuite, exécutez chaque endpoint successivement.

À la fin, vous pourrez accéder aux deux endpoints finaux qui affichent les graphiques des résultats essentiels.

## Auteurs

- [Alisa Chekalina](https://github.com/chekalisa)
- [Lia Gasparin](https://github.com/LiaGasparin)
- [Carmen Cristea](https://github.com/CarmenParis)