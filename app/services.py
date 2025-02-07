
######################################################################
#                 Imports et création des répertoires                #
######################################################################


from fastapi import HTTPException, Query
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os
import shutil
import re
from gensim.models import Word2Vec
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.spatial.distance import cosine
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# Les ressources NLTK 
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')



PROCESSED_DIR_NEWS = "processed_data_news"
os.makedirs(PROCESSED_DIR_NEWS, exist_ok=True)

PROCESSED_DIR_TWEETS = "processed_data_tweets"
os.makedirs(PROCESSED_DIR_TWEETS, exist_ok=True)

GRAPHS = "graphs"
os.makedirs(GRAPHS, exist_ok=True)


######################################################################
#                               Articles                             #
######################################################################


def preprocess_message_news(message):

    """Nettoie et tokenise les articles de presse."""

    if not isinstance(message, str):
        return []
    
    message = message.lower()
    message = re.sub(r'\d+', '', message)
    tokens = word_tokenize(message)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    cleaned_tokens = [token for token in lemmatized_tokens if len(token) > 1]
    
    return cleaned_tokens



def perform_clustering(data, k, linkage_criterion, metric):

    """Crée des clusters en utilisant AgglomerativeClustering."""

    clustering = AgglomerativeClustering(n_clusters=k, metric=metric, linkage=linkage_criterion)
    clusters = clustering.fit_predict(data)
    return clusters



def process_news():
    """
    Nettoie, tokenise et filtre les articles, intègre les variations Bitcoin, 
    et génère des fichiers optimisés.
    """
    print("INFO: Début du processus de traitement.")

    news_file = os.path.join('uploaded_data', "news.xlsx")
    bitcoin_file = os.path.join('uploaded_data', "Bitcoin.csv")

    if not os.path.exists(news_file) or not os.path.exists(bitcoin_file):
        print("INFO: Fichiers requis non trouvés.")
        raise HTTPException(status_code=400, detail="Les fichiers des articles et du Bitcoin doivent être ajoutés en premier.")
    
    print("INFO: Chargement des fichiers en cours...")
    news_df = pd.read_excel(news_file)
    
    required_columns = {'Date', 'Titre', 'Article'}
    if not required_columns.issubset(news_df.columns):
        raise HTTPException(status_code=400, detail=f"Le fichier doit contenir {required_columns}")

    print("INFO: Prétraitement des articles de presse...")
    news_df['Date'] = pd.to_datetime(news_df['Date']).dt.date
    
    news_df['Processed_News'] = news_df['Article'].apply(preprocess_message_news)
    processed_news_path = os.path.join(PROCESSED_DIR_NEWS, "news_tokenized.csv")

    news_df.to_csv(processed_news_path, index=False)
    print(f"INFO: Base avec tokens sauvegardée à {processed_news_path}")
    
    print("INFO: Création de la matrice des tokens...")
    unique_tokens = sorted(set(token for tokens in news_df['Processed_News'] for token in tokens))
    token_matrix = pd.DataFrame(0, index=news_df.index, columns=unique_tokens)

    for i, tokens in enumerate(news_df['Processed_News']):
        token_matrix.loc[news_df.index[i], list(set(tokens))] = 1

    final_df = pd.concat([news_df[['Date', 'Titre']], token_matrix], axis=1)
    final_df.fillna(0, inplace=True)

    matrix_news_path = os.path.join(PROCESSED_DIR_NEWS, "matrix_news.csv")
    final_df.to_csv(matrix_news_path, index=False)
    print(f"INFO: Matrice des tokens sauvegardée à {matrix_news_path}")

    print("INFO: Début de filtrage des tokens... ...")
    matrix_news = final_df.copy()
    token_columns = matrix_news.columns.difference(['Date', 'Titre'])
    total_articles = len(matrix_news)

    token_frequencies = matrix_news[token_columns].sum(axis=0)

    tokens_to_remove = token_frequencies[
        (token_frequencies > 0.75 * total_articles) | (token_frequencies < 0.2 * total_articles)
    ].index.tolist()

    filtered_matrix = matrix_news.drop(columns=tokens_to_remove)

    print("INFO: Sauvegarde de matrice filtré...")
    filtred_matrix_news_path = os.path.join(PROCESSED_DIR_NEWS, "matrix_news_filtred.csv")
    filtered_matrix.to_csv(filtred_matrix_news_path, index=False)

    print(f"INFO: Matrice filtrée sauvegardée à {filtred_matrix_news_path}")

    print("INFO: Chargement et traitement des données Bitcoin...")
    bitcoin_df = pd.read_csv(bitcoin_file)
    bitcoin_df.columns = bitcoin_df.columns.str.replace('"', '')
    bitcoin_df = bitcoin_df.map(lambda x: str(x).replace('"', ''))
    bitcoin_df['Date'] = pd.to_datetime(bitcoin_df['Date'], format='%d/%m/%Y')
    bitcoin_df['Variation %'] = bitcoin_df['Variation %'].str.replace(',', '.').str.rstrip('%').astype(float)

    bitcoin_df['Next_Date'] = bitcoin_df['Date'] - pd.Timedelta(days=1)
    variation_mapping = bitcoin_df[['Next_Date', 'Variation %']].rename(columns={'Next_Date': 'Date'})

    filtered_matrix['Date'] = pd.to_datetime(filtered_matrix['Date'])
    filtered_matrix = filtered_matrix.merge(variation_mapping, on='Date', how='left')

    token_columns = filtered_matrix.columns.difference(['Date', 'Titre', 'Variation %'])
    filtered_matrix[token_columns] = filtered_matrix[token_columns].astype(float)

    print("INFO: Application des variations Bitcoin aux tokens...")
    for index, row in filtered_matrix.iterrows():
        variation = row['Variation %']
        filtered_matrix.loc[index, token_columns] = filtered_matrix.loc[index, token_columns].where(filtered_matrix.loc[index, token_columns] != 1, variation)

    filtred_matrix_news_path_variation = os.path.join(PROCESSED_DIR_NEWS, "matrix_news_with_variation.csv")
    filtered_matrix.to_csv(filtred_matrix_news_path_variation, index=False)

    print("INFO: Calcul des moyennes des tokens...")
    average_per_token = filtered_matrix[token_columns].mean(axis=0, skipna=True)
    average_per_token_sorted = average_per_token.sort_values()

    n_tokens = int(len(average_per_token_sorted) * 0.25)
    selected_tokens = pd.concat([
        average_per_token_sorted.head(n_tokens),  
        average_per_token_sorted.tail(n_tokens)  
    ])

    selected_tokens_path = os.path.join(PROCESSED_DIR_NEWS, "selected_tokens.csv")
    selected_tokens.to_csv(selected_tokens_path, header=["Average"])


    selected_tokens = pd.read_csv(selected_tokens_path, index_col=0)

    selected_tokens_list = selected_tokens.index.tolist()
    columns_to_keep = ['Date', 'Titre'] + selected_tokens_list

    print("INFO: Chargement de la matrice initiale des tokens...")


    matrix_news=pd.read_csv(matrix_news_path)

    print("INFO: Filtrage de la matrice des tokens...")
    filtered_matrix_news = matrix_news[columns_to_keep]
    filtered_matrix_news_path_by_tokens = os.path.join(PROCESSED_DIR_NEWS, "matrix_news_filtered_by_tokens.csv")
    filtered_matrix_news.to_csv(filtered_matrix_news_path_by_tokens, index=False)

    print("INFO: Traitement terminé.")

    return {"message": "Processing complete", "output_files": [matrix_news_path, filtred_matrix_news_path, filtred_matrix_news_path_variation, selected_tokens_path, filtered_matrix_news_path_by_tokens]}




def process_vectorization():

    """
    Vectorise les articles en utilisant Word2Vec, génère des représentations numériques 
    et sauvegarde les vecteurs ainsi que le modèle entraîné.
    """

    print("INFO: Début de la vectorisation des articles.")


    filtered_matrix_path = os.path.join(PROCESSED_DIR_NEWS, "matrix_news_filtered_by_tokens.csv")
    if not os.path.exists(filtered_matrix_path):
        print("ERROR: Fichier filtré introuvable.")
        raise HTTPException(status_code=400, detail="Le fichier de matrice filtrée est introuvable.")

    print("INFO: Chargement de la matrice des tokens sélectionnés...")
    filtered_matrix_news = pd.read_csv(filtered_matrix_path)


    if filtered_matrix_news.empty:
        print("ERROR: Le fichier chargé est vide.")
        raise HTTPException(status_code=400, detail="La matrice filtrée est vide.")
    
    print("INFO: Préparation des données pour Word2Vec...")
    sentences = []
    for _, row in filtered_matrix_news.iterrows():
        article_tokens = row[2:].index[row[2:] == 1].tolist()
        sentences.append(article_tokens)

    if not sentences or all(len(sentence) == 0 for sentence in sentences):
        print("ERROR: Aucune donnée valide pour entraîner Word2Vec.")
        raise HTTPException(status_code=400, detail="Aucune donnée valide pour entraîner le modèle Word2Vec.")


    print("INFO: Entraînement du modèle Word2Vec...")
    model = Word2Vec(sentences, vector_size=200, window=5, min_count=1, workers=4)

    model_path = os.path.join(PROCESSED_DIR_NEWS, "word2vec_model_news.model")
    model.save(model_path)
    print(f"INFO: Modèle Word2Vec sauvegardé à {model_path}")



    
    def get_article_vector(article_tokens):

        """
        Calcule le vecteur moyen d'un article en agrégeant les vecteurs Word2Vec de ses tokens.
        Retourne un vecteur nul si aucun token valide n'est trouvé.
        """

        vectors = [model.wv[token] for token in article_tokens if token in model.wv]
        return np.mean(vectors, axis=0) if len(vectors) > 0 else np.zeros(model.vector_size)

    print("INFO: Calcul des vecteurs pour chaque article...")
    article_vectors = []
    for _, row in filtered_matrix_news.iterrows():
        article_tokens = row[2:].index[row[2:] == 1].tolist()
        article_vector = get_article_vector(article_tokens)
        article_vectors.append(article_vector)

    article_vectors_df = pd.DataFrame(article_vectors)
    article_vectors_df['Date'] = filtered_matrix_news['Date']
    article_vectors_df['Titre'] = filtered_matrix_news['Titre']


    output_file = os.path.join(PROCESSED_DIR_NEWS, "article_vectors.csv")
    article_vectors_df.to_csv(output_file, index=False)

    print(f"INFO: vectorization terminée")
    return {
        "message": "Vectorisation complète",
        "output_files": [output_file, model_path]
    }




def process_silhouette():

    """
    Calcule et analyse les scores de silhouette pour déterminer le nombre optimal de clusters.
    Génère un graphique des scores et le sauvegarde.
    """

    print("INFO: Début du calcul des scores de silouhette.")

    vectors_path = os.path.join(PROCESSED_DIR_NEWS, "article_vectors.csv")

    print("INFO: Chargement des vecteurs d'articles...")

    article_vectors_df = pd.read_csv(vectors_path)

    article_vectors_only = article_vectors_df.iloc[:, :-2].values 

    article_vectors_only = article_vectors_df.iloc[:, :-2].values  

    range_n_clusters = range(2, 11)
    silhouette_scores = []

    for n_clusters in range_n_clusters:
        agglo_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        cluster_labels = agglo_clustering.fit_predict(article_vectors_only)  
        silhouette_avg = silhouette_score(article_vectors_only, cluster_labels)  
        silhouette_scores.append(silhouette_avg)
        print(f"Pour {n_clusters} clusters, le score de silhouette est de {silhouette_avg:.4f}")

    print("INFO: Sauvegarde du graphique...")
    silhouette_plot_path = os.path.join(GRAPHS, "silhouette_scores.png")
    plt.figure(figsize=(10, 6))
    plt.plot(range_n_clusters, silhouette_scores, marker='o')
    plt.xlabel("Nombre de clusters")
    plt.ylabel("Score de silhouette")
    plt.title("Méthode de silhouette pour déterminer le nombre optimal de clusters")
    plt.grid()
    plt.savefig(silhouette_plot_path)
    plt.close()
    print(f"Graphique des scores de silhouette sauvegardé : {silhouette_plot_path}")

    return {
        "message": "Score de silhouette calculé",
        "output_files": {
            "silhouette_plot": silhouette_plot_path
        }
    }




def process_clustering(k: int = Query(3, description="Nombre de clusters")):

    """
    Effectue un clustering hiérarchique des articles, ajuste les petits clusters, 
    et génère les fichiers des résultats et des centroïdes.
    """

    vectors_path = os.path.join(PROCESSED_DIR_NEWS, "article_vectors.csv")

    print("INFO: Chargement des vecteurs d'articles...")

    if not os.path.exists(vectors_path) :
        print("INFO: Fichiers requis non trouvés.")
        raise HTTPException(status_code=400, detail="Les fichiers des vecteurs doivent être chargés en premier.")
    
    article_vectors_df = pd.read_csv(vectors_path)

    vector_columns = article_vectors_df.select_dtypes(include=[np.number]).columns
    article_vectors_only = article_vectors_df[vector_columns]

    print(f"INFO: Début du clustering avec {k} clusters...")
    
    linkage_criterion = 'ward' 
    metric = 'euclidean'  
    
    article_vectors_df['Cluster'] = perform_clustering(article_vectors_only, k, linkage_criterion, metric)
    
    cluster_sizes = article_vectors_df['Cluster'].value_counts()
    print(f"Taille des clusters initiaux :\n{cluster_sizes}")

    # Supprimer les petits clusters et relancer le clustering, pour gérer les clusters qui sont très petits, par exemple cluster à 1 ou 2 articles
    min_cluster_size = 5
    while (cluster_sizes < min_cluster_size).any():
        small_clusters = cluster_sizes[cluster_sizes < min_cluster_size].index
        print(f"Clusters trop petits (< {min_cluster_size}) : {small_clusters}")
       
        article_vectors_df = article_vectors_df[~article_vectors_df['Cluster'].isin(small_clusters)]
        article_vectors_only = article_vectors_df[vector_columns]
        
        if len(article_vectors_df) < k:
            print("Nombre insuffisant d'articles pour relancer le clustering avec le même nombre de clusters. Fin de la boucle.")
            break
        article_vectors_df['Cluster'] = perform_clustering(article_vectors_only, k, linkage_criterion, metric)
        cluster_sizes = article_vectors_df['Cluster'].value_counts()
        print(f"Taille des clusters après filtrage :\n{cluster_sizes}")

    print("INFO: Calcul des centroides (médiane par cluster)...")
    centroids = []
    for cluster in sorted(article_vectors_df['Cluster'].unique()):
        cluster_vectors = article_vectors_only[article_vectors_df['Cluster'] == cluster]
        centroid = np.median(cluster_vectors, axis=0)
        centroids.append(centroid)

    centroids_df = pd.DataFrame(centroids, columns=[f"Dim_{i+1}" for i in range(article_vectors_only.shape[1])])
    centroids_df['Cluster'] = sorted(article_vectors_df['Cluster'].unique())

    print("INFO:Sauvegarder les résultats du clustering")

    clustering_output_file = os.path.join(PROCESSED_DIR_NEWS, "article_vectors_with_clusters.csv")
    centroids_output_file = os.path.join(PROCESSED_DIR_NEWS,"cluster_centroids.csv")

    article_vectors_df.to_csv(clustering_output_file, index=False)
    centroids_df.to_csv(centroids_output_file, index=False)

    return {
        "message": f"Clustering terminé avec {k} clusters",
        "parameters_used": {
            "k": k
        },
        "output_files": {
            "clustering": clustering_output_file,
            "centroids": centroids_output_file
        }
    }




def process_outliers(threshold: float = Query(0.9, description="Seuil de similarité cosinus pour filtrer les articles")):

    """
    Filtre les articles considérés comme outliers en fonction de leur similarité cosinus 
    avec leur centroïde de cluster et met à jour les centroïdes.
    """

    print("INFO: Chargement des fichiers de clustering et des centroids...")
    
    clustering_output_file = os.path.join(PROCESSED_DIR_NEWS, "article_vectors_with_clusters.csv")
    centroids_output_file = os.path.join(PROCESSED_DIR_NEWS, "cluster_centroids.csv")
    


    if not os.path.exists(clustering_output_file) or not os.path.exists(centroids_output_file):
        print("INFO: Fichiers requis non trouvés.")
        raise HTTPException(status_code=400, detail="Les fichiers des vecteurs et clusters associés, ainsi que le fichier des centroides doivent être chargés")
    
    article_vectors_df = pd.read_csv(clustering_output_file)
    centroids_df = pd.read_csv(centroids_output_file)

    print("INFO: Calcul de la similarité cosinus entre chaque article et son centroïde")
    
    articles_to_keep = []
    similarities = []
    
    for _, row in article_vectors_df.iterrows():
        cluster_id = row['Cluster']
        article_vector = row.iloc[:-3].values  
        centroid_row = centroids_df[centroids_df['Cluster'] == cluster_id]
        
        if not centroid_row.empty:
            centroid_vector = centroid_row.iloc[0, :-1].values
            similarity = 1 - cosine(article_vector, centroid_vector)
            similarities.append(similarity)
            if similarity >= threshold:
                articles_to_keep.append(row)
    
    print(f"INFO: {len(similarities)} similarités cosinus calculées.")
    print(f"Min: {min(similarities):.4f}, Max: {max(similarities):.4f}, Moyenne: {np.mean(similarities):.4f}")
    
    filtered_articles_df = pd.DataFrame(articles_to_keep).reset_index(drop=True)
    
    new_centroids = []
    for cluster in sorted(filtered_articles_df['Cluster'].unique()):
        cluster_vectors = filtered_articles_df[filtered_articles_df['Cluster'] == cluster].iloc[:, :-3].values
        new_centroid = np.median(cluster_vectors, axis=0)
        new_centroids.append(new_centroid)
    
    new_centroids_df = pd.DataFrame(new_centroids, columns=[f"Dim_{i+1}" for i in range(filtered_articles_df.shape[1] - 3)])
    new_centroids_df['Cluster'] = sorted(filtered_articles_df['Cluster'].unique())
    
    filtered_output_file = os.path.join(PROCESSED_DIR_NEWS, "article_vectors_with_clusters_without_outliers.csv")
    new_centroids_output_file = os.path.join(PROCESSED_DIR_NEWS, "new_cluster_centroids.csv")
    
    filtered_articles_df.to_csv(filtered_output_file, index=False)
    new_centroids_df.to_csv(new_centroids_output_file, index=False)
    
    print("INFO: Suppression des outliers terminée.")
    
    return {
        "message": "Suppression des outliers terminée",
        "threshold_used": threshold,
        "output_files": {
            "articles_avant_filtrage": len(article_vectors_df),
            "articles_après_filtrage": len(filtered_articles_df),
            "fichier_articles_filtrés": filtered_output_file,
            "fichier_nouveaux_centroids": new_centroids_output_file
        }
    }




def clusters_without_outliers():

    """
    Associe les articles filtrés à leurs clusters, applique TF-IDF aux tokens,
    et génère des fichiers de résultats et des graphiques des termes importants par cluster.
    """

    print("INFO: Chargement des fichiers...")

    tokens_output_file = os.path.join(PROCESSED_DIR_NEWS, "matrix_news_filtered_by_tokens.csv")
    clusters_output_file = os.path.join(PROCESSED_DIR_NEWS, "article_vectors_with_clusters_without_outliers.csv")

    if not os.path.exists(tokens_output_file) or not os.path.exists(clusters_output_file):
        print("INFO: Fichiers requis non trouvés.")
        raise HTTPException(status_code=400, detail="La matrice avec les tokens ou les vecteurs sans outliers doivent être téléversés en premier.")
    
    tokens_df = pd.read_csv(tokens_output_file)
    clusters_df = pd.read_csv(clusters_output_file)

    tokens_df = tokens_df[tokens_df['Titre'].isin(clusters_df['Titre'])]

    merged_df = tokens_df.merge(clusters_df[['Titre', 'Cluster']], on='Titre', how='left')

    token_columns = merged_df.columns.difference(['Date', 'Titre', 'Cluster'])

    print("INFO: Calcul des scores TF-IDF...")
    cluster_tfidf_results = {}
    cluster_graphs = {}

    for cluster in merged_df['Cluster'].unique():
        print(f"Cluster {cluster}")

        cluster_df = merged_df[merged_df['Cluster'] == cluster]

        tfidf_transformer = TfidfTransformer()
        tfidf_matrix = tfidf_transformer.fit_transform(cluster_df[token_columns])

        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=token_columns)
        tfidf_df['Titre'] = cluster_df['Titre']

        cluster_mean_tfidf = tfidf_df[token_columns].mean().sort_values(ascending=False)

    
        cluster_tfidf_results[cluster] = cluster_mean_tfidf

        print(f"Top termes importants pour le Cluster {cluster}:")
        print(cluster_mean_tfidf.head(10))

    
        cluster_path = os.path.join(GRAPHS, f"cluster_{cluster}_top_terms.png")

        plt.figure(figsize=(10, 6))
        cluster_mean_tfidf.head(10).plot(kind='bar', color='skyblue')
        plt.title(f"Cluster {cluster} - Top termes importants")
        plt.xlabel("Termes")
        plt.ylabel("Score TF-IDF moyen")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(cluster_path)
        plt.close()

        cluster_graphs[f"Cluster {cluster}"] = cluster_path

    
    tf_idf_output_file = os.path.join(PROCESSED_DIR_NEWS, "tfidf_per_cluster.csv")
    all_clusters_df = pd.concat(
        [pd.DataFrame({f"Cluster_{cluster}": tfidf}) for cluster, tfidf in cluster_tfidf_results.items()],
        axis=1
    )
    all_clusters_df.to_csv(tf_idf_output_file)

    print("INFO : Calcul des TF-IDF terminé.")

    return {
        "message": "Calcul terminé",
        "output_files": {
            "Moyennes TF-IDF": tf_idf_output_file,
            "Cluster Graphs": cluster_graphs
        }
    }



######################################################################
#                             TWEETS                                 #
######################################################################



def preprocess_message_tweets(message):
        
        """
        Nettoie et tokenise un tweet en supprimant les stopwords, les chiffres, 
        et les répétitions, puis applique la lemmatisation.
        """

        if not isinstance(message, str): 
            return []

        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer() 

        message = message.lower()
        message = re.sub(r'\d+', '', message)
        
        tokens = word_tokenize(message)
        filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
        
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
        
        cleaned_tokens = [token for token in lemmatized_tokens if len(token) > 2 and 'btc' not in token and 'bitcoin' not in token]
        cleaned_tokens = [token for token in cleaned_tokens if not re.search(r'(.)\1{2,}', token)]
        
        return cleaned_tokens if cleaned_tokens else None  




def preprocess_tweets():
    
    """
    Nettoie, tokenise et vectorise les tweets en créant une matrice de tokens, 
    puis filtre les tokens selon un lexique issu des articles de presse.
    """

    tweet_file = "uploaded_data/tweets.csv"
    tweets_df = pd.read_csv(tweet_file)
    tweets_df = tweets_df.drop_duplicates()
    tweets_df["Titre"] = tweets_df["Message"].str.replace("$BTC.X", "", regex=False)

    required_columns = {'Date', 'Message', "Titre"}
    if not required_columns.issubset(tweets_df.columns):
        raise ValueError(f"Le fichier CSV doit contenir les colonnes : {required_columns}.")


    tweets_df['Date'] = pd.to_datetime(tweets_df['Date']).dt.date

    tweets_df['Processed_Message'] = tweets_df['Message'].apply(preprocess_message_tweets)

    tweets_df = tweets_df.dropna(subset=['Processed_Message'])

    tweets_df = tweets_df[tweets_df['Processed_Message'].apply(lambda x: len(x) > 0)]

    output_folder_tweets_processed = os.path.join(PROCESSED_DIR_TWEETS,"tweets_tokenized.csv")
    tweets_df.to_csv(output_folder_tweets_processed, index=False)


    unique_tokens = sorted(set(token for tokens in tweets_df['Processed_Message'] for token in tokens))
    token_matrix = pd.DataFrame(0, index=tweets_df.index, columns=unique_tokens)

    for i, tokens in enumerate(tweets_df['Processed_Message']):
        token_matrix.loc[tweets_df.index[i], list(set(tokens))] = 1

    final_df = pd.concat([tweets_df[['Date', 'Titre']], token_matrix], axis=1)

    final_df.fillna(0, inplace=True)

    output_folder_matrix = os.path.join(PROCESSED_DIR_TWEETS,"matrix_tweets.csv")
    final_df.to_csv(output_folder_matrix, index=False)

    lexique_output_file = os.path.join(PROCESSED_DIR_NEWS, "selected_tokens.csv")
    
    news_lexicon = pd.read_csv(lexique_output_file, index_col=0) 

    news_tokens_list = news_lexicon.index.tolist()

    columns_to_keep = ['Date', 'Titre'] + [token for token in final_df.columns if token in news_tokens_list]

    filtered_matrix_tweets = final_df[columns_to_keep]

    folder_filtered_matrix_tweets = os.path.join(PROCESSED_DIR_TWEETS,"matrix_tweets_filtered_by_news_tokens.csv")
    filtered_matrix_tweets.to_csv(folder_filtered_matrix_tweets, index=False)
    
    return {
        "message": "Prétraitement terminé",
        "output_files": {
            "Tweets tokenisés": output_folder_tweets_processed,
            "Matrice des tokens": output_folder_matrix,
            "Matrice filtrée": folder_filtered_matrix_tweets
        }
    }




def process_tweets_vectorization():
    
    """
    Charge la matrice filtrée des tweets et applique le modèle Word2Vec pour générer 
    leurs représentations vectorielles, puis sauvegarde les résultats.
    """

    print("INFO: Chargement de la matrice filtrée des tweets...")

    folder_filtered_matrix_tweets = os.path.join(PROCESSED_DIR_TWEETS, "matrix_tweets_filtered_by_news_tokens.csv")
    if not os.path.exists(folder_filtered_matrix_tweets):
        print("INFO: Fichiers requis non trouvés.")
        raise HTTPException(status_code=400, detail="Le fichier des tweets filtrés doit être généré.")

    filtered_matrix_tweets = pd.read_csv(folder_filtered_matrix_tweets)

    folder_word2vec_model = os.path.join(PROCESSED_DIR_NEWS, "word2vec_model_news.model")
    if not os.path.exists(folder_word2vec_model):
        print("INFO: Modèle Word2Vec non trouvé.")
        raise HTTPException(status_code=400, detail="Le modèle Word2Vec doit être tout d'abord entraîné.")

    print("INFO: Chargement du modèle Word2Vec...")
    model = Word2Vec.load(folder_word2vec_model)



    
    def get_tweet_vector(tweet_tokens):

        """
        Calcule le vecteur moyen d'un tweet en agrégeant les vecteurs Word2Vec de ses tokens.
        Retourne un vecteur nul si aucun token valide n'est trouvé.
        """
         
        vectors = [model.wv[token] for token in tweet_tokens if token in model.wv]
        return np.mean(vectors, axis=0) if len(vectors) > 0 else np.zeros(model.vector_size)

    print("INFO: Calcul des vecteurs pour chaque tweet...")

    tweet_vectors = []
    for _, row in filtered_matrix_tweets.iterrows():
        tweet_tokens = row[2:].index[row[2:] == 1].tolist()  
        tweet_vector = get_tweet_vector(tweet_tokens)  
        tweet_vectors.append(tweet_vector)

    tweet_vectors_df = pd.DataFrame(tweet_vectors)
    tweet_vectors_df['Date'] = filtered_matrix_tweets['Date']
    tweet_vectors_df['Titre'] = filtered_matrix_tweets['Titre']

    print("INFO: Sauvegarde des vecteurs des tweets...")

    folder_vectors_tweets = os.path.join(PROCESSED_DIR_TWEETS, "tweet_vectors.csv")
    tweet_vectors_df.to_csv(folder_vectors_tweets, index=False)

    print("INFO: Vectorisation des tweets terminée.")

    return {
        "message": "Vectorisation des tweets terminée",
        "output_files": {
            "Vecteurs des tweets": folder_vectors_tweets
        }
    }

    


def tweets_assignment():

    """
    Assigne les tweets aux clusters les plus similaires en utilisant la similarité cosinus 
    avec les centroïdes des articles, puis sauvegarde les résultats.
    """

    print("INFO: Chargement des fichiers de vecteurs...")

    folder_vectors_tweets = os.path.join(PROCESSED_DIR_TWEETS, "tweet_vectors.csv")
    folder_news_vectors = os.path.join(PROCESSED_DIR_NEWS, "article_vectors_with_clusters_without_outliers.csv")
    folder_centroids = os.path.join(PROCESSED_DIR_NEWS, "new_cluster_centroids.csv")

    for file in [folder_vectors_tweets, folder_news_vectors, folder_centroids]:
        if not os.path.exists(file):
            print(f"INFO: Fichier manquant -> {file}")
            raise HTTPException(status_code=400, detail=f"Le fichier {file} doit être généré d'abord.")

    tweet_vectors_df = pd.read_csv(folder_vectors_tweets)
    news_vectors_df = pd.read_csv(folder_news_vectors)
    centroids_df = pd.read_csv(folder_centroids)

    tweet_vectors_df['Date'] = pd.to_datetime(tweet_vectors_df['Date'])
    news_vectors_df['Date'] = pd.to_datetime(news_vectors_df['Date'])


    tweet_vectors_only = tweet_vectors_df.iloc[:, :-2].values
    centroid_vectors = centroids_df.iloc[:, :-1].values

    print("INFO: Calcul de la similarité cosinus...")

    similarity_matrix = cosine_similarity(tweet_vectors_only, centroid_vectors)

    closest_clusters = similarity_matrix.argmax(axis=1)
    max_similarities = similarity_matrix.max(axis=1)

    similarity_threshold = 0.55

    tweet_vectors_df['Cluster'] = np.where(max_similarities >= similarity_threshold, closest_clusters, -1)
    tweet_vectors_df['Similarity'] = max_similarities

    assigned_tweets_df = tweet_vectors_df[tweet_vectors_df['Cluster'] != -1]
    daily_clustered_tweets = assigned_tweets_df.groupby(['Date', 'Cluster'])['Titre'].apply(list).reset_index()

    print("INFO: Sauvegarde des tweets assignés aux clusters...")

    output_file_assigned_tweets = os.path.join(PROCESSED_DIR_TWEETS, "tweet_vectors_with_clusters.csv")
    assigned_tweets_df.to_csv(output_file_assigned_tweets, index=False)

    output_file_daily_tweets = os.path.join(PROCESSED_DIR_TWEETS, "daily_clustered_tweets.csv")
    daily_clustered_tweets.to_csv(output_file_daily_tweets, index=False)

    print("INFO: Attribution des tweets aux clusters terminée.")

    return {
        "message": "Attribution des tweets aux clusters terminée",
        "output_files": {
            "Tweets assignés aux clusters": output_file_assigned_tweets,
            "Tweets groupés par jour et cluster": output_file_daily_tweets
        }
    }




def alert_generation():
    
    """
    Analyse le pourcentage de tweets assignés aux clusters par jour, 
    génère un graphique d'alerte et sauvegarde les résultats.
    """

    print("INFO: Chargement des fichiers de tweets...")

    folder_vectors_tweets = os.path.join(PROCESSED_DIR_TWEETS, "tweet_vectors.csv")
    output_file_assigned_tweets = os.path.join(PROCESSED_DIR_TWEETS, "tweet_vectors_with_clusters.csv")

    for file in [folder_vectors_tweets, output_file_assigned_tweets]:
        if not os.path.exists(file):
            print(f"INFO: Fichier manquant -> {file}")
            raise HTTPException(status_code=400, detail=f"Le fichier {file} doit être généré d'abord.")

    all_tweets_df = pd.read_csv(folder_vectors_tweets)
    assigned_tweets_df = pd.read_csv(output_file_assigned_tweets)

    all_tweets_df['Date'] = pd.to_datetime(all_tweets_df['Date'])
    assigned_tweets_df['Date'] = pd.to_datetime(assigned_tweets_df['Date'])

    print("INFO: Calcul du pourcentage de tweets assignés...")

    total_tweets_per_day = all_tweets_df.groupby('Date').size().reset_index(name='Total_Tweets')
    assigned_tweets_per_day = assigned_tweets_df.groupby('Date').size().reset_index(name='Assigned_Tweets')

    association_stats = total_tweets_per_day.merge(assigned_tweets_per_day, on="Date", how="left")

    association_stats['Assigned_Tweets'].fillna(0, inplace=True)

    association_stats['Percentage_Assigned'] = (association_stats['Assigned_Tweets'] / association_stats['Total_Tweets']) * 100

    print("INFO: Génération du graphique d'alerte...")

    association_stat_path = os.path.join(GRAPHS, "alert_generation.png")

    plt.figure(figsize=(12, 6))
    plt.plot(association_stats['Date'], association_stats['Percentage_Assigned'], marker='o', linestyle='-', color='b', label="Assigned Percentage")
    plt.axhline(y=35, color='r', linestyle='--', label="Seuil 35%")
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xlabel("Date")
    plt.ylabel("Pourcentage de tweets associés (%)")
    plt.title("Évolution du pourcentage de tweets associés aux clusters")
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.legend()
    plt.savefig(association_stat_path)
    plt.close()

    print("INFO: Génération des alertes terminée.")

    return {
        "message": "Génération des alertes terminée",
        "output_files": {
            "Graphique d'alerte": association_stat_path
        }
    }
